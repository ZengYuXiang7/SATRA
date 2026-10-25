import torch
import platform


def build_runtime_protocol(
    args,
    precision_effective,
    amp_effective,
    tf32_effective,
    compile_effective,
    compile_reason,
    optimizer_name="AdamW",
    fused_effective=None,
    fused_disabled_reason=None,
    clip_ratio=None,
    max_grad_norm_seen=None,
):
    return {
        "precision": {
            "requested": args.precision,
            "effective": precision_effective,
            "amp_requested": args.precision == "bf16_amp_tf32",
            "amp_effective": bool(amp_effective),
            "autocast_dtype": "bfloat16" if amp_effective else None,
            "loss_dtype": "float32",
            "tf32_requested": args.precision in {"tf32", "bf16_amp_tf32"},
            "tf32_effective": bool(tf32_effective),
            "reason": "cuda_bf16_amp_tf32" if precision_effective == "bf16_amp_tf32" else precision_effective,
        },
        "compile": {
            "requested": bool(args.torch_compile),
            "effective": bool(compile_effective),
            "mode": args.torch_compile_mode,
            "reason": compile_reason,
        },
        "optimizer": {
            "name": optimizer_name,
            "fused_requested": args.device_type == "cuda",
            "fused_effective": fused_effective,
            "disabled_reason": fused_disabled_reason,
        },
        "dataloader": {
            "pin_memory_requested": bool(args.pin_memory_requested),
            "pin_memory_effective": bool(args.pin_memory),
        },
        "grad_clip": {
            "max_grad_norm": float(args.max_grad_norm),
            "error_if_nonfinite": True,
            "clip_ratio": clip_ratio,
            "max_grad_norm_seen": max_grad_norm_seen,
        },
        "drop_last": {
            "train_requested": bool(args.train_drop_last),
            "train_effective": bool(args.train_drop_last),
            "valid_effective": bool(args.valid_drop_last),
            "test_effective": False,
            "semantics": "stock time-window iteration; train/valid may drop tail batches for compile-stable shapes, test keeps all eval offsets",
        },
    }


def get_runtime_info(
    args,
    precision_effective,
    amp_effective,
    tf32_effective,
    compile_effective,
    compile_reason,
    fused_effective=None,
    fused_disabled_reason=None,
):
    runtime_protocol = build_runtime_protocol(
        args,
        precision_effective,
        amp_effective,
        tf32_effective,
        compile_effective,
        compile_reason,
        fused_effective=fused_effective,
        fused_disabled_reason=fused_disabled_reason,
    )
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": False,
        "device": "cpu",
        "gpu_name": None,
        "runtime_protocol": runtime_protocol,
    }
    info.update(
        {
            "precision_requested": runtime_protocol["precision"]["requested"],
            "precision_effective": runtime_protocol["precision"]["effective"],
            "amp_effective": runtime_protocol["precision"]["amp_effective"],
            "tf32_effective": runtime_protocol["precision"]["tf32_effective"],
            "torch_compile_requested": runtime_protocol["compile"]["requested"],
            "torch_compile_effective": runtime_protocol["compile"]["effective"],
            "torch_compile_mode": runtime_protocol["compile"]["mode"],
            "torch_compile_reason": runtime_protocol["compile"]["reason"],
            "max_grad_norm": runtime_protocol["grad_clip"]["max_grad_norm"],
        }
    )
    if bool(args.use_gpu):
        idx = int(args.gpu)
        info["device"] = f"cuda:{idx}"
        info["gpu_name"] = torch.cuda.get_device_name(idx)
    return info


def model_param_count(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "trainable_m": trainable / 1_000_000.0}
