#!/usr/bin/env python3
"""Pre-download every weight the generator needs.

A live demo must never hit the network: run this once, ahead of time.

Two artefacts are fetched:

1. The DreamShaper XL v2 Turbo single-file checkpoint (~6.5 GB) into ``models/``.
   No URL is hardcoded -- Civitai downloads may require an account token and the
   links rotate -- so pass ``--url`` (or set ``model.checkpoint_url`` in
   ``config/settings.yaml``). If the fetch fails, manual instructions are printed.
2. The ``madebyollin/sdxl-vae-fp16-fix`` VAE, pulled into the Hugging Face cache.

The script is idempotent: existing files are skipped unless ``--force`` is given.

Usage:
    python scripts/download_models.py --url https://example.com/dreamshaper.safetensors
    python scripts/download_models.py --url "https://civitai.com/api/download/models/XXXX" --token $CIVITAI_TOKEN
    python scripts/download_models.py --skip-checkpoint   # VAE only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    from animegen.config import Settings, load_settings
except ModuleNotFoundError:  # running from a clone without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from animegen.config import Settings, load_settings

CHUNK_SIZE = 1024 * 1024
MIN_PLAUSIBLE_CHECKPOINT_BYTES = 100 * 1024 * 1024


def human(num_bytes: float) -> str:
    """Render a byte count as a short human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}GB"


def manual_instructions(destination: Path, reason: str) -> str:
    """Build the copy-pasteable fallback shown when an automatic fetch fails."""
    return "\n".join(
        [
            "",
            "=" * 72,
            f"Automatic checkpoint download failed: {reason}",
            "",
            "Download DreamShaper XL v2 Turbo manually instead:",
            "  1. Open https://civitai.com/models/112902 (DreamShaper XL)",
            "     and pick the 'v2 Turbo DPM++ SDE' .safetensors file,",
            "     or use a Hugging Face mirror such as",
            "     https://huggingface.co/Lykon/dreamshaper-xl-v2-turbo",
            "  2. Log in if the site asks for it (Civitai gates some downloads).",
            f"  3. Save the file as:  {destination}",
            "  4. Re-run this script to fetch the VAE:",
            "     python scripts/download_models.py --skip-checkpoint",
            "",
            "Civitai API tokens work too:",
            "  python scripts/download_models.py --url <api-download-url> --token <token>",
            "=" * 72,
            "",
        ]
    )


def download_file(url: str, destination: Path, token: str | None = None) -> None:
    """Stream ``url`` to ``destination`` with a progress line.

    The download lands in a ``.part`` file and is renamed only on success, so an
    interrupted run never leaves a truncated checkpoint that loads as garbage.

    Raises:
        RuntimeError: If the response is too small to be a real checkpoint.
    """
    import requests

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    part_file = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=30, headers=headers) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        if total and total < MIN_PLAUSIBLE_CHECKPOINT_BYTES:
            raise RuntimeError(
                f"response is only {human(total)}; the URL probably returned an "
                "HTML login page rather than the checkpoint"
            )

        downloaded = 0
        with part_file.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    print(
                        f"\r  {human(downloaded)} / {human(total)} ({pct:5.1f}%)",
                        end="",
                        flush=True,
                    )
                else:
                    print(f"\r  {human(downloaded)}", end="", flush=True)
        print()

    part_file.replace(destination)


def fetch_checkpoint(settings: Settings, args: argparse.Namespace) -> bool:
    """Download the SDXL checkpoint. Returns True when it is present afterwards."""
    models_dir = Path(args.models_dir) if args.models_dir else settings.paths.models_dir
    destination = (models_dir / settings.model.checkpoint_filename).resolve()

    if destination.is_file() and not args.force:
        print(f"[skip] checkpoint already present: {destination} "
              f"({human(destination.stat().st_size)})")
        return True

    url = args.url or settings.model.checkpoint_url
    if not url:
        print(manual_instructions(destination, "no --url given and no checkpoint_url in settings"))
        return False

    print(f"[get ] checkpoint -> {destination}")
    try:
        download_file(url, destination, token=args.token)
    except Exception as exc:  # noqa: BLE001 - any failure ends in manual instructions
        print(manual_instructions(destination, f"{type(exc).__name__}: {exc}"))
        return False

    print(f"[ok  ] checkpoint saved ({human(destination.stat().st_size)})")
    return True


def fetch_vae(settings: Settings, args: argparse.Namespace) -> bool:
    """Pull the fp16-fix VAE into the Hugging Face cache. Returns True on success."""
    repo = settings.model.vae_repo
    print(f"[get ] VAE {repo}")
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError:
        print("[fail] huggingface_hub is not installed; run: pip install -e .")
        return False

    kwargs: dict[str, Any] = {
        "repo_id": repo,
        "allow_patterns": ["*.json", "*.safetensors"],
    }
    if args.force:
        kwargs["force_download"] = True
    try:
        path = snapshot_download(**kwargs)
    except Exception as exc:  # noqa: BLE001 - report and let the caller decide
        print(f"[fail] could not download {repo}: {type(exc).__name__}: {exc}")
        print("       Check your network, then re-run with --skip-checkpoint.")
        return False

    print(f"[ok  ] VAE cached at {path}")
    return True


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="download_models.py",
        description=(
            "Pre-download the DreamShaper XL v2 Turbo checkpoint and the "
            "fp16-fix VAE so the demo never downloads live."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/download_models.py --url https://host/dreamshaper.safetensors\n"
            "  python scripts/download_models.py --skip-checkpoint\n"
        ),
    )
    parser.add_argument(
        "--url",
        "--source",
        dest="url",
        help="Direct download URL for the checkpoint (overrides model.checkpoint_url).",
    )
    parser.add_argument(
        "--token",
        help="Bearer token for gated hosts such as Civitai.",
    )
    parser.add_argument(
        "--models-dir",
        help="Destination directory for the checkpoint (default: paths.models_dir).",
    )
    parser.add_argument(
        "--config",
        help="Settings file to read (default: config/settings.yaml).",
    )
    parser.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help="Do not fetch the checkpoint; only pull the VAE.",
    )
    parser.add_argument(
        "--skip-vae",
        action="store_true",
        help="Do not fetch the VAE; only get the checkpoint.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the files already exist.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config) if args.config else load_settings()

    ok = True
    if not args.skip_checkpoint:
        ok &= fetch_checkpoint(settings, args)
    if not args.skip_vae:
        ok &= fetch_vae(settings, args)

    if ok:
        print("\nAll model files are in place. Next: animegen generate \"...\"")
        return 0
    print("\nSome downloads did not complete; see the instructions above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
