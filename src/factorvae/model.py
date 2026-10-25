"""
FactorVAE: A Probabilistic Dynamic Factor Model Based on Variational
Autoencoder for Predicting Cross-Sectional Stock Returns.

Source type: paper-reproduction-oriented implementation.

This file keeps the feature extractor, factor encoder, factor decoder, and
factor predictor as one reviewable PyTorch module.

Expected input:
    x: Tensor [batch, num_stocks, lookback, num_features]
    returns: optional Tensor [batch, num_stocks, 1]
    mask: Tensor [batch, num_stocks, 1]

Outputs:
    If returns is None: Tensor [batch, num_stocks, 1] prediction from the prior factor predictor.
    If returns is provided: dict containing reconstruction distribution, prediction,
    posterior/prior Gaussian parameters, and the VAE loss.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """GRU feature extractor for stock characteristics."""

    def __init__(self, num_features: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.normalize = nn.LayerNorm(num_features)
        self.linear = nn.Linear(num_features, num_features)
        self.activation = nn.LeakyReLU()
        self.gru = nn.GRU(num_features, hidden_size, num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, stock, seq_len, feature]
        batch_size, stock_num, seq_len, feature_num = x.shape
        x = x.reshape(batch_size * stock_num, seq_len, feature_num)
        out = self.normalize(x)
        out = self.activation(self.linear(out))
        stock_latent, _ = self.gru(out)
        return stock_latent[:, -1, :].reshape(batch_size, stock_num, -1)


class FactorEncoder(nn.Module):
    """Posterior q(z|x,y): portfolio layer followed by Gaussian mapping."""

    def __init__(self, num_factors: int, num_portfolios: int, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_portfolios)
        # Softmax is over the stock dimension so each portfolio is a cross-section.
        self.softmax = nn.Softmax(dim=1)
        self.linear_mu = nn.Linear(num_portfolios, num_factors)
        self.linear_sigma = nn.Linear(num_portfolios, num_factors)
        self.softplus = nn.Softplus()

    def mapping_layer(self, portfolio_return: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        portfolio_return = portfolio_return.squeeze(-1)
        mean = self.linear_mu(portfolio_return)
        sigma = self.softplus(self.linear_sigma(portfolio_return)) + 1e-6
        return mean, sigma

    def forward(
        self,
        stock_latent: torch.Tensor,
        returns: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.linear(stock_latent)
        valid = mask.to(dtype=torch.bool).expand_as(logits)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        weights = self.softmax(logits)  # [batch, stock, portfolio]
        returns = returns * mask.to(dtype=returns.dtype)
        portfolio_return = weights.transpose(1, 2).matmul(returns)  # [batch, portfolio, 1]
        return self.mapping_layer(portfolio_return)


class AlphaLayer(nn.Module):
    """Stock-specific alpha distribution p(alpha_i|x_i)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.LeakyReLU()
        self.mu_layer = nn.Linear(hidden_size, 1)
        self.sigma_layer = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()

    def forward(self, stock_latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.activation(self.linear1(stock_latent))
        alpha_mu = self.mu_layer(h)
        alpha_sigma = self.softplus(self.sigma_layer(h)) + 1e-6
        return alpha_mu, alpha_sigma


class BetaLayer(nn.Module):
    """Factor exposure beta_i for each stock."""

    def __init__(self, hidden_size: int, num_factors: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_factors)

    def forward(self, stock_latent: torch.Tensor) -> torch.Tensor:
        return self.linear(stock_latent)


class FactorDecoder(nn.Module):
    """Return decoder y_i = alpha_i + beta_i z."""

    def __init__(self, hidden_size: int, num_factors: int):
        super().__init__()
        self.alpha_layer = AlphaLayer(hidden_size)
        self.beta_layer = BetaLayer(hidden_size, num_factors)

    @staticmethod
    def reparameterize(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(sigma)
        return mu + eps * sigma

    def distribution(
        self,
        stock_latent: torch.Tensor,
        factor_mu: torch.Tensor,
        factor_sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_mu, alpha_sigma = self.alpha_layer(stock_latent)
        beta = self.beta_layer(stock_latent)

        factor_mu = factor_mu.unsqueeze(-1)
        factor_sigma = factor_sigma.clamp_min(1e-6).unsqueeze(-1)
        mu = alpha_mu + beta.matmul(factor_mu)
        sigma = torch.sqrt(alpha_sigma.pow(2) + beta.pow(2).matmul(factor_sigma.pow(2)) + 1e-6)
        return mu, sigma

    def forward(
        self,
        stock_latent: torch.Tensor,
        factor_mu: torch.Tensor,
        factor_sigma: torch.Tensor,
        sample: bool = True,
    ) -> torch.Tensor:
        mu, sigma = self.distribution(stock_latent, factor_mu, factor_sigma)
        return self.reparameterize(mu, sigma) if sample else mu


class AttentionLayer(nn.Module):
    """Cosine attention pooling used by the prior factor predictor."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_size))
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, stock_latent: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        key = self.key_layer(stock_latent)
        value = self.value_layer(stock_latent)
        query = self.query.view(1, 1, -1).expand_as(key)
        score = F.cosine_similarity(key, query, dim=-1, eps=1e-8)
        score = F.relu(score)
        if mask is not None:
            stock_mask = mask.squeeze(-1).to(dtype=torch.bool) if mask.ndim == 3 else mask.to(dtype=torch.bool)
            score = score.masked_fill(~stock_mask, 0.0)
        attention = score / score.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return attention.unsqueeze(1).matmul(value).squeeze(1)


class FactorPredictor(nn.Module):
    """Prior p(z|x) estimated from stock latent representations."""

    def __init__(self, hidden_size: int, num_factors: int):
        super().__init__()
        self.num_factors = num_factors
        self.attention_layers = nn.ModuleList([AttentionLayer(hidden_size) for _ in range(num_factors)])
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.LeakyReLU()
        self.mu_layer = nn.Linear(hidden_size, 1)
        self.sigma_layer = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        stock_latent: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        contexts = [layer(stock_latent, mask=mask) for layer in self.attention_layers]
        h_multi = torch.stack(contexts, dim=1)
        h_multi = self.activation(self.linear(h_multi))
        pred_mu = self.mu_layer(h_multi).squeeze(-1)
        pred_sigma = self.softplus(self.sigma_layer(h_multi)).squeeze(-1) + 1e-6
        return pred_mu, pred_sigma


class FactorVAE(nn.Module):
    """End-to-end FactorVAE model."""

    def __init__(
        self,
        num_features: int,
        num_factors: int = 96,
        num_portfolios: int = 128,
        hidden_size: int = 64,
        num_gru_layers: int = 1,
        kl_weight: float = 1.0,
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.feature_extractor = FeatureExtractor(num_features, hidden_size, num_gru_layers)
        self.factor_encoder = FactorEncoder(num_factors, num_portfolios, hidden_size)
        self.factor_decoder = FactorDecoder(hidden_size, num_factors)
        self.factor_predictor = FactorPredictor(hidden_size, num_factors)

    @staticmethod
    def kl_divergence(mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
        sigma1 = sigma1.clamp_min(1e-6)
        sigma2 = sigma2.clamp_min(1e-6)
        kl = torch.log(sigma2 / sigma1) + (sigma1.pow(2) + (mu1 - mu2).pow(2)) / (2 * sigma2.pow(2)) - 0.5
        return kl.sum()


    def prediction(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        sample: bool = False,
    ) -> torch.Tensor:
        stock_latent = self.feature_extractor(x)
        pred_mu, pred_sigma = self.factor_predictor(stock_latent, mask=mask)
        return self.factor_decoder(stock_latent, pred_mu, pred_sigma, sample=sample).float()


    def forward(
        self,
        x: torch.Tensor,
        returns: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        stock_latent = self.feature_extractor(x)
        if mask is None and returns is not None:
            mask = torch.ones_like(returns)

        pred_mu, pred_sigma = self.factor_predictor(stock_latent, mask=mask)
        prediction = self.factor_decoder(stock_latent, pred_mu, pred_sigma, sample=False).float()

        if returns is None:
            return prediction

        loss_mask = mask.to(dtype=torch.float32)
        factor_mu, factor_sigma = self.factor_encoder(stock_latent, returns, mask=loss_mask)
        reconstruction_mu, reconstruction_sigma = self.factor_decoder.distribution(
            stock_latent,
            factor_mu,
            factor_sigma,
        )
        returns = returns.float()
        pred_mu = pred_mu.float()
        pred_sigma = pred_sigma.float()
        factor_mu = factor_mu.float()
        factor_sigma = factor_sigma.float()
        reconstruction_mu = reconstruction_mu.float()
        reconstruction_sigma = reconstruction_sigma.float()
        standardized = (returns - reconstruction_mu) / reconstruction_sigma.clamp_min(1e-6)
        
        nll = 0.5 * (
            standardized.pow(2)
            + 2.0 * torch.log(reconstruction_sigma.clamp_min(1e-6))
            + math.log(2.0 * math.pi)
        )
        
        valid_count = loss_mask.sum().clamp_min(1.0)
        reconstruction_loss = (nll * loss_mask).sum() / valid_count
        reconstruction_mse = ((reconstruction_mu - returns).pow(2) * loss_mask).sum() / valid_count
        kl = self.kl_divergence(factor_mu, factor_sigma, pred_mu, pred_sigma) / factor_mu.numel()
        loss = reconstruction_loss + self.kl_weight * kl

        return {
            "loss": loss,
            "negative_log_likelihood": reconstruction_loss,
            "reconstruction_mse": reconstruction_mse,
            "kl_loss": kl,
            "reconstruction": reconstruction_mu,
            "reconstruction_sigma": reconstruction_sigma,
            "prediction": prediction,
            "posterior_mu": factor_mu,
            "posterior_sigma": factor_sigma,
            "prior_mu": pred_mu,
            "prior_sigma": pred_sigma,
        }


Model = FactorVAE
