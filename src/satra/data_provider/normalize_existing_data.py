import argparse
import json
import os
import pickle
import shutil

import numpy as np

from satra.data_provider.generate_data import INPUT_NORM_METHOD, INPUT_NORM_SCOPE, RAW_NORM_LAYOUT, normalize_eod_features


def load_pickle(path):
    with open(path, "rb") as f:
        return np.asarray(pickle.load(f), dtype=np.float32)


def dump_pickle(path, array):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(array.astype(np.float32), f, protocol=pickle.HIGHEST_PROTOCOL)


def dump_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def normalize_market(source_root, output_root, market, args):
    src_dir = os.path.join(source_root, f"{market}_ohlcv")
    out_dir = os.path.join(output_root, f"{market}_ohlcv")
    with open(os.path.join(src_dir, "feature_config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)

    eod_data = load_pickle(os.path.join(src_dir, "eod_data.pkl"))
    feature_names = list(config["feature_names"])
    original_feature_names = list(config["original_feature_names"])
    feature_args = dict(config)
    feature_args["input_norm_method"] = INPUT_NORM_METHOD
    feature_args["input_norm_scope"] = INPUT_NORM_SCOPE
    feature_args["raw_norm_layout"] = RAW_NORM_LAYOUT
    normalized, input_normalization = normalize_eod_features(
        eod_data,
        feature_names,
        original_feature_names,
        market,
        feature_args,
    )

    if args.clean_output and os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    dump_pickle(os.path.join(out_dir, "eod_data.pkl"), normalized)
    for name in ["mask_data.pkl", "observed_mask_data.pkl", "gt_data.pkl", "price_data.pkl"]:
        shutil.copy2(os.path.join(src_dir, name), os.path.join(out_dir, name))
    config["input_norm_method"] = INPUT_NORM_METHOD
    config["input_norm_scope"] = INPUT_NORM_SCOPE
    config["raw_norm_layout"] = RAW_NORM_LAYOUT
    config["input_normalization"] = input_normalization
    config["normalization_source_root"] = source_root
    dump_json(os.path.join(out_dir, "feature_config.json"), config)
    return {
        "eod_data": normalized.shape,
        "feature_dim": int(normalized.shape[2]),
        "input_normalization": input_normalization,
        "min": float(np.min(normalized)),
        "max": float(np.max(normalized)),
        "mean": float(np.mean(normalized)),
        "std": float(np.std(normalized)),
    }


def main():
    parser = argparse.ArgumentParser(description="Normalize existing *_ohlcv datasets after feature generation.")
    parser.add_argument("--source-root", default="data/processed")
    parser.add_argument("--output-root", default="data/processed")
    parser.add_argument("--markets", nargs="+", default=["NASDAQ", "NYSE", "SP500"], choices=["NASDAQ", "NYSE", "SP500"])
    parser.add_argument("--clean-output", action="store_true")
    args = parser.parse_args()

    for market in args.markets:
        stats = normalize_market(args.source_root, args.output_root, market, args)
        print(f"{market} normalized to {os.path.join(args.output_root, market + '_ohlcv')}: {stats}")


if __name__ == "__main__":
    main()
