"""Command-line interface.

Examples:
    animegen generate "American teenagers having fun at a party"
    animegen generate "rooftop party at dusk" --images 4 --seed 1234 --warmup
    animegen generate "beach party" --no-llm --size 1024x1024 --steps 8
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from animegen import __version__
from animegen.config import Settings, load_settings
from animegen.core.orchestrator import Orchestrator, RunResult

LOGGER = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Generate semi-3D (2.5D) anime images locally: an Ollama model expands "
        "your idea, DreamShaper XL v2 Turbo renders it on 8 GB of VRAM."
    ),
)


def setup_logging(verbose: bool = False) -> None:
    """Configure root logging for CLI use."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if not verbose:
        # diffusers/transformers are chatty about weight conversion at INFO.
        for noisy in ("diffusers", "transformers", "httpx", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_size(value: str, settings: Settings) -> tuple[int, int]:
    """Parse a ``WIDTHxHEIGHT`` string into pixel dimensions.

    Args:
        value: Size such as ``832x1216``.
        settings: Active settings, used to warn about untested sizes.

    Returns:
        A ``(width, height)`` tuple.

    Raises:
        typer.BadParameter: If the format is wrong or the dimensions are not
            positive multiples of 8 (SDXL's latent stride).
    """
    parts = value.lower().replace(" ", "").split("x")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise typer.BadParameter(
            f"'{value}' is not a WIDTHxHEIGHT size, e.g. 832x1216"
        )
    width, height = (int(part) for part in parts)
    if width <= 0 or height <= 0 or width % 8 or height % 8:
        raise typer.BadParameter(
            f"'{value}': width and height must be positive multiples of 8"
        )
    if value not in settings.generation.allowed_sizes:
        LOGGER.warning(
            "Size %s is outside the tested set %s; expect slower renders or OOM",
            value,
            ", ".join(settings.generation.allowed_sizes),
        )
    return width, height


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"animegen {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    ctx: typer.Context,
    config: Annotated[
        Optional[Path],
        typer.Option("--config", help="Settings file (default: config/settings.yaml)."),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable debug logging.")
    ] = False,
    _version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Load settings once and hand them to the subcommands."""
    setup_logging(verbose)
    ctx.obj = load_settings(config)


@app.command()
def generate(
    ctx: typer.Context,
    prompt: Annotated[
        str, typer.Argument(help='Your idea, e.g. "American teenagers having fun at a party".')
    ],
    images: Annotated[
        int, typer.Option("--images", "-n", min=1, help="How many images to render.")
    ] = 1,
    seed: Annotated[
        Optional[int],
        typer.Option("--seed", help="Seed of the first image (random when omitted)."),
    ] = None,
    size: Annotated[
        Optional[str],
        typer.Option("--size", help="WIDTHxHEIGHT, e.g. 832x1216 or 1024x1024."),
    ] = None,
    steps: Annotated[
        Optional[int], typer.Option("--steps", help="Denoising steps (Turbo range 4-8).")
    ] = None,
    guidance: Annotated[
        Optional[float], typer.Option("--guidance", help="Classifier-free guidance scale.")
    ] = None,
    no_llm: Annotated[
        bool, typer.Option("--no-llm", help="Skip Ollama; render the raw prompt.")
    ] = False,
    warmup: Annotated[
        bool,
        typer.Option(
            "--warmup",
            help="Render a throwaway image first so the real one is not the slow one.",
        ),
    ] = False,
) -> None:
    """Generate images from PROMPT and save them under outputs/."""
    settings: Settings = ctx.obj or load_settings()
    width, height = parse_size(size, settings) if size else (None, None)

    orchestrator = Orchestrator(settings=settings)
    if warmup:
        typer.echo("Warming up the pipeline...")
        orchestrator.warmup()

    result = orchestrator.run(
        prompt=prompt,
        images=images,
        seed=seed,
        width=width,
        height=height,
        steps=steps,
        guidance=guidance,
        use_llm=not no_llm,
    )
    _print_summary(result)


def _print_summary(result: RunResult) -> None:
    """Print a short, demo-friendly report of a run."""
    typer.echo("")
    typer.echo(f"Prompt     : {result.original_prompt}")
    typer.echo(f"Enhanced   : {result.enhanced_prompt}")
    if not result.llm_enabled:
        typer.echo("LLM        : skipped (--no-llm)")
    elif result.used_fallback:
        typer.echo("LLM        : unavailable, used the raw prompt")
    else:
        typer.echo(f"LLM        : ok ({result.llm_duration_s:.1f}s)")
    typer.echo(
        f"Timing     : sd {result.sd_duration_s:.1f}s, total {result.total_duration_s:.1f}s"
    )
    typer.echo("Images     :")
    for image in result.images:
        typer.echo(f"  seed {image.seed:<12} {image.path}")


@app.command()
def ui(
    ctx: typer.Context,
    host: Annotated[
        str, typer.Option("--host", help="Interface to bind.")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="TCP port.")] = 7860,
    share: Annotated[
        bool, typer.Option("--share", help="Expose a public gradio.live tunnel.")
    ] = False,
    warmup: Annotated[
        Optional[bool],
        typer.Option(
            "--warmup/--no-warmup",
            help="Render a throwaway image at startup (default: ANIMEGEN_UI_WARMUP).",
        ),
    ] = None,
) -> None:
    """Serve the Gradio demo interface."""
    from animegen.ui.app import DemoApp

    settings: Settings = ctx.obj or load_settings()
    DemoApp(settings=settings).launch(host=host, port=port, share=share, warmup=warmup)


@app.command()
def info(ctx: typer.Context) -> None:
    """Show the active configuration and whether Ollama is reachable."""
    from animegen.llm.enhancer import OllamaEnhancer

    settings: Settings = ctx.obj or load_settings()
    checkpoint = settings.checkpoint_path
    ollama_up = OllamaEnhancer(settings=settings).is_available()

    typer.echo(f"animegen {__version__}")
    typer.echo(f"checkpoint : {checkpoint} "
               f"({'found' if checkpoint.is_file() else 'MISSING'})")
    typer.echo(f"vae        : {settings.model.vae_repo}")
    typer.echo(
        f"defaults   : {settings.generation.width}x{settings.generation.height}, "
        f"{settings.generation.steps} steps, guidance {settings.generation.guidance_scale}, "
        f"{settings.generation.scheduler}"
    )
    typer.echo(
        f"ollama     : {settings.ollama.host} ({settings.ollama.model}) "
        f"{'up' if ollama_up else 'DOWN -> --no-llm fallback'}"
    )
    typer.echo(f"outputs    : {settings.outputs_dir}")


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
