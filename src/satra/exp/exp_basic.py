import os

import torch


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = None

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        device_type = self.args.device_type
        logger = self.args.obs_logger
        if device_type == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device(f"cuda:{self.args.gpu}")
            if logger is not None:
                logger.log(f"Use GPU: cuda:{self.args.gpu}")
        else:
            device = torch.device("cpu")
            if logger is not None:
                logger.log("Use CPU")
        return device
