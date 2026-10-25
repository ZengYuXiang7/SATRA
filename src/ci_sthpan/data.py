from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from rsr.data import RSRData, make_data_args as make_rsr_data_args


HYPERGRAPH_ALIASES = {
    "auto": "correlation_top16",
    "correlation": "correlation_top16",
    "corr": "correlation_top16",
    "correlation_top16": "correlation_top16",
    "sector_industry": "rsr_industry",
    "industry": "rsr_industry",
    "rsr_industry": "rsr_industry",
    "wiki": "rsr_wiki_neighbor",
    "wikidata": "rsr_wiki_neighbor",
    "rsr_wiki": "rsr_wiki_neighbor",
    "rsr_wiki_neighbor": "rsr_wiki_neighbor",
    "sp500_sector": "sp500_sector",
    "estimate_distance": "estimate_leadlag_distance",
    "estimate_pearson": "estimate_leadlag_pearson",
    "estimate_kendall": "estimate_leadlag_kendall",
    "estimate_mutual_information": "estimate_leadlag_mutual_information",
}


def make_data_args(data, **overrides) -> Namespace:
    hypergraph_name = overrides.pop("hypergraph_name", "auto")
    relation_name = overrides.get("relation_name")
    if relation_name is None:
        resolved = HYPERGRAPH_ALIASES.get(str(hypergraph_name), str(hypergraph_name))
        if resolved.startswith("correlation_top"):
            overrides["relation_name"] = resolved
        elif resolved.startswith("rsr_industry"):
            overrides["relation_name"] = "rsr_industry"
        elif resolved.startswith("rsr_wiki"):
            overrides["relation_name"] = "rsr_wiki"
        elif resolved.startswith("sp500_sector"):
            overrides["relation_name"] = "sp500_sector"
        else:
            overrides["relation_name"] = "correlation_top16"
    args = make_rsr_data_args(data, **overrides)
    args.hypergraph_name = hypergraph_name
    return args


class CISTHPANData(RSRData):
    """Unified stock windows plus a CI-STHPAN hyperedge_index tensor."""

    def __init__(self, args):
        super().__init__(args)
        self.hypergraph_name = self._resolve_hypergraph_name()
        self.hyperedge_path = self._resolve_hyperedge_path()
        self.hyperedge_index = self._load_hyperedge_index()
        self.hyperedge_count = int(self.hyperedge_index[1].max() + 1) if self.hyperedge_index.size else 0
        self.input_transform_metadata.update(
            {
                "hypergraph_name": self.hypergraph_name,
                "hyperedge_path": str(self.hyperedge_path),
                "hyperedge_shape": list(self.hyperedge_index.shape),
                "hyperedge_count": self.hyperedge_count,
                "hypergraph_source": self._hypergraph_source_from_manifest(),
                "paper_note": (
                    "CI-STHPAN paper builds channel-independent DTW hypergraphs. "
                    "This baseline defaults to leakage-free correlation_top16 hyperedges "
                    "prepared from the active training segment unless --hypergraph_name is set."
                ),
            }
        )

    def _resolve_hypergraph_name(self):
        requested = str(getattr(self.args, "hypergraph_name", "auto") or "auto")
        return HYPERGRAPH_ALIASES.get(requested, requested)

    def _resolve_hyperedge_path(self):
        root = Path(self.args.relation_root) / self.market
        candidates = [
            root / f"{self.hypergraph_name}_hyperedge_index.npy",
            root / f"{self.hypergraph_name}_neighbor_hyperedge_index.npy",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            f"hyperedge_index file not found for {self.hypergraph_name!r}; tried "
            + ", ".join(str(path) for path in candidates)
        )

    def _load_hyperedge_index(self):
        edge = np.asarray(np.load(self.hyperedge_path), dtype=np.int64)
        if edge.ndim != 2 or edge.shape[0] != 2:
            raise ValueError(f"hyperedge_index must be [2,E], got {edge.shape}")
        if edge.size:
            if int(edge[0].min()) < 0 or int(edge[0].max()) >= self.stock_num:
                raise ValueError(
                    f"hyperedge node ids must be in [0,{self.stock_num}); got "
                    f"[{int(edge[0].min())},{int(edge[0].max())}]"
                )
            if int(edge[1].min()) < 0:
                raise ValueError("hyperedge ids must be non-negative")
        return edge

    def _hypergraph_source_from_manifest(self):
        try:
            import json

            manifest = json.loads((Path(self.args.relation_root) / "manifest.json").read_text(encoding="utf-8"))
            key = self.hyperedge_path.name.removesuffix(".npy")
            return manifest["datasets"][self.market]["files"][key].get("source")
        except Exception:
            return None

    def hyperedge_tensor(self, device=None):
        tensor = torch.as_tensor(self.hyperedge_index, dtype=torch.long)
        if device is not None:
            tensor = tensor.to(device)
        return tensor


__all__ = ["CISTHPANData", "make_data_args"]
