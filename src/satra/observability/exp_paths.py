from pathlib import Path


def build_exp_paths(args):
    timestamp = args.exp_timestamp
    exp_dir = Path(args.results) / args.data / args.model / timestamp
    paths = {
        "exp_dir": exp_dir,
        "details_dir": exp_dir / "details",
        "curves_dir": exp_dir / "curves",
        "checkpoints_dir": exp_dir / "checkpoints",
        "native_dir": exp_dir / "native",
        "config_path": exp_dir / "config.json",
        "summary_path": exp_dir / "summary_all_runs.json",
        "run_md_path": exp_dir / "run.md",
        "project_run_md_path": Path("results/run.md"),
        "model_detail_path": exp_dir / "model_detail.md",
    }
    paths["project_run_md_path"].parent.mkdir(parents=True, exist_ok=True)
    for key in ["details_dir", "curves_dir", "checkpoints_dir", "native_dir"]:
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def run_paths(paths, run_id):
    return {
        "detail_log": paths["details_dir"] / f"run{run_id}.log",
        "history": paths["exp_dir"] / f"history_run{run_id}.json",
        "curve": paths["curves_dir"] / f"training_curve_run{run_id}.png",
        "best_ckpt": paths["checkpoints_dir"] / f"best_model_run{run_id}.pt",
        "last_ckpt": paths["checkpoints_dir"] / f"last_model_run{run_id}.pt",
    }
