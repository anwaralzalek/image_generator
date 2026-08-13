#!/usr/bin/env python3
"""Pre-download the three image-model profiles into the Hugging Face cache.

The repositories publish ordinary fp16/fp32 source weights. AnimeGen converts
the supported denoiser and text-encoder linear weights to INT8 with Quanto when
it loads a profile; the source cache is shared by local and Docker runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    from animegen.config import ModelProfile, Settings, load_settings
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from animegen.config import ModelProfile, Settings, load_settings


COMPONENT_PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "feature_extractor/*",
    "safety_checker/*",
    "tokenizer/*",
    "tokenizer_2/*",
    "tokenizer_3/*",
    "text_encoder/*",
    "text_encoder_2/*",
    "text_encoder_3/*",
    "unet/*",
    "vae/*",
]


def _ignore_patterns(profile: ModelProfile) -> list[str]:
    """Avoid duplicate root checkpoints and unused full-precision variants."""
    root_checkpoints = {
        "cagliostrolab/animagine-xl-4.0": [
            "animagine-xl-4.0.safetensors",
            "animagine-xl-4.0-opt.safetensors",
        ],
        "Lykon/dreamshaper-xl-v2-turbo": [
            "DreamShaperXL_Turbo_V2-SFW.safetensors",
            "DreamShaperXL_Turbo_v2.safetensors",
            "DreamShaperXL_Turbo_v2_1.safetensors",
        ],
        "dreamlike-art/dreamlike-anime-1.0": [
            "dreamlike-anime-1.0.ckpt",
            "dreamlike-anime-1.0.safetensors",
        ],
    }
    ignored = list(root_checkpoints.get(profile.repo_id, []))
    # Component SafeTensors are required; legacy pickle files are both unsafe
    # and incompatible with the supported CUDA 12.1 PyTorch build.
    ignored.extend(["*.bin", "*.ckpt"])
    if profile.variant == "fp16":
        ignored.extend(
            [
                "unet/diffusion_pytorch_model.safetensors",
                "vae/diffusion_pytorch_model.safetensors",
                "text_encoder/model.safetensors",
                "text_encoder_2/model.safetensors",
            ]
        )
    return ignored


def fetch_profile(
    key: str, profile: ModelProfile, token: str | None, force: bool
) -> bool:
    """Cache one Diffusers repository without redundant single-file weights."""
    from huggingface_hub import snapshot_download

    print(
        f"[get ] {key}: {profile.name} ({profile.repo_id}; "
        f"estimated {profile.estimate})"
    )
    kwargs: dict[str, Any] = {
        "repo_id": profile.repo_id,
        "allow_patterns": COMPONENT_PATTERNS,
        "ignore_patterns": _ignore_patterns(profile),
        "token": token,
    }
    if force:
        kwargs["force_download"] = True
    try:
        path = snapshot_download(**kwargs)
    except Exception as exc:  # noqa: BLE001 - report every failed profile
        print(f"[fail] {profile.repo_id}: {type(exc).__name__}: {exc}")
        return False
    print(f"[ok  ] cached at {path}")
    return True


def fetch_vae(settings: Settings, token: str | None, force: bool) -> bool:
    """Cache the stable fp16 SDXL VAE shared by best and balanced profiles."""
    from huggingface_hub import snapshot_download

    repo = settings.model.vae_repo
    print(f"[get ] shared SDXL VAE: {repo}")
    kwargs: dict[str, Any] = {
        "repo_id": repo,
        "allow_patterns": ["*.json", "*.safetensors"],
        "token": token,
    }
    if force:
        kwargs["force_download"] = True
    try:
        path = snapshot_download(**kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"[fail] {repo}: {type(exc).__name__}: {exc}")
        return False
    print(f"[ok  ] cached at {path}")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download_models.py",
        description="Pre-download one or all INT8 runtime image-model profiles.",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=["all", "best", "balanced", "fast"],
        help="Profile to download; repeat as needed (default: all).",
    )
    parser.add_argument("--token", help="Hugging Face token for gated repositories.")
    parser.add_argument("--config", help="Settings file (default: config/settings.yaml).")
    parser.add_argument("--skip-vae", action="store_true", help="Skip the shared SDXL VAE.")
    parser.add_argument("--force", action="store_true", help="Re-download cached files.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config) if args.config else load_settings()
    requested = args.model or ["all"]
    keys = list(settings.model.profiles) if "all" in requested else requested

    ok = True
    for key in keys:
        _, profile = settings.image_model(key)
        ok &= fetch_profile(key, profile, args.token, args.force)
    if not args.skip_vae and any(
        settings.image_model(key)[1].architecture == "sdxl" for key in keys
    ):
        ok &= fetch_vae(settings, args.token, args.force)

    if ok:
        print("\nModels cached. Runtime loading enforces INT8 weight quantization.")
        return 0
    print("\nOne or more model downloads failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
