from argparse import Namespace

import numpy as np

from rsr.data import RSRData, make_data_args as make_rsr_data_args


def make_data_args(data, **overrides) -> Namespace:
    overrides.setdefault("relation_name", "auto")
    overrides.setdefault("input_features", "stock_all")
    return make_rsr_data_args(data, **overrides)


class TRAData(RSRData):
    """Unified stock windows with ordered loaders for causal TRA memory updates."""

    def _debug_train_offsets(self):
        offsets = self.train_offsets
        if self.args.debug:
            max_windows = int(self.args.debug_train_steps) * int(self.args.batch_size)
            offsets = offsets[:max_windows]
        return np.asarray(offsets, dtype=np.int64)

    def _debug_eval_offsets(self, start_index, end_index):
        offsets = np.asarray(list(self.iter_eval_offsets(start_index, end_index)), dtype=np.int64)
        if self.args.debug:
            offsets = offsets[: int(self.args.debug_eval_steps)]
        return offsets

    def train_loader(self, epoch=0):
        return self._loader(
            offsets=self._debug_train_offsets(),
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=bool(self.args.train_drop_last),
            return_offset=True,
            seed=int(self.args.random_seed) + int(epoch),
        )

    def train_memory_loader(self):
        return self._loader(
            offsets=self._debug_train_offsets(),
            batch_size=1,
            shuffle=False,
            drop_last=False,
            return_offset=True,
            seed=None,
        )

    def causal_eval_loader(self, start_index, end_index):
        return self._loader(
            offsets=self._debug_eval_offsets(start_index, end_index),
            batch_size=1,
            shuffle=False,
            drop_last=False,
            return_offset=True,
            seed=None,
        )


__all__ = ["TRAData", "make_data_args"]
