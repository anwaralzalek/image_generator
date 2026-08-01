"""Command-line interface.

Examples:
    animegen generate "American teenagers having fun at a party"
    animegen generate "rooftop party at dusk" --images 4 --seed 1234
    animegen generate "beach party" --no-llm --size 1024x1024 --steps 8
"""

from __future__ import annotations

import logging
from typing import Annotated, Optional

import typer

from animegen import __version__
from animegen.config import (
    COMPUTE_DTYPE,
    LINEAR_WEIGHT_DTYPE,
    MAX_IMAGES,
    QUANTIZATION_BACKEND,
    SEED_MAX,
    Settings,
    load_settings,
    parse_dimensions,
)
from animegen.core.orchestrator import Orchestrator, RunResult

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Generate 3D anime images locally with three selectable "
        "image-quality tiers, INT8 linear weights, and optional Ollama prompt "
        "enhancement."
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


def parse_size(value: str) -> tuple[int, int]:
    try:
        return parse_dimensions(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"animegen {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    ctx: typer.Context,
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
    try:
        ctx.obj = load_settings()
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid configuration: {exc}") from None


@app.command()
def generate(
    ctx: typer.Context,
    prompt: Annotated[
        str,
        typer.Argument(
            help='Your idea, e.g. "American teenagers having fun at a party".'
        ),
    ],
    images: Annotated[
        int,
        typer.Option(
            "--images", "-n", min=1, max=MAX_IMAGES, help="How many images to render."
        ),
    ] = 1,
    seed: Annotated[
        Optional[int],
        typer.Option(
            "--seed",
            min=0,
            max=SEED_MAX,
            help="Seed of the first image (random when omitted).",
        ),
    ] = None,
    size: Annotated[
        Optional[str],
        typer.Option("--size", help="WIDTHxHEIGHT, e.g. 832x1216 or 1024x1024."),
    ] = None,
    steps: Annotated[
        Optional[int],
        typer.Option(
            "--steps",
            min=1,
            help="Denoising steps (clamped to the model's tested range).",
        ),
    ] = None,
    guidance: Annotated[
        Optional[float],
        typer.Option("--guidance", min=0, help="Classifier-free guidance scale."),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option("--model", help="Image model: best, balanced, or fast."),
    ] = None,
    no_llm: Annotated[
        bool, typer.Option("--no-llm", help="Skip Ollama; render the raw prompt.")
    ] = False,
) -> None:
    """Generate images from PROMPT and save them under outputs/."""
    settings: Settings = ctx.obj or load_settings()
    try:
        model_key, _ = settings.image_model(model)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--model") from None
    width, height = parse_size(size) if size else (None, None)

    orchestrator = Orchestrator(settings=settings)
    try:
        result = orchestrator.run(
            prompt=prompt,
            images=images,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            model=model_key,
            use_llm=not no_llm,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from None
    _print_summary(result)


def _print_summary(result: RunResult) -> None:
    """Print a short, demo-friendly report of a run."""
    typer.echo("")
    typer.echo(f"Prompt     : {result.original_prompt}")
    typer.echo(f"Enhanced   : {result.enhanced_prompt}")
    if result.images:
        model_settings = result.images[0].metadata.get("settings", {})
        typer.echo(
            f"Model      : {model_settings.get('model', '?')} "
            f"({model_settings.get('linear_weight_dtype', '?')} linear weights, "
            f"estimated {model_settings.get('estimated_render_time', '?')})"
        )
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
    port: Annotated[
        int, typer.Option("--port", min=1, max=65535, help="TCP port.")
    ] = 7860,
) -> None:
    """Serve the Gradio demo interface."""
    try:
        import gradio  # noqa: F401
    except ImportError:
        raise typer.BadParameter(
            'UI dependencies are missing; install with pip install -e ".[ui]"'
        ) from None
    from animegen.ui.app import DemoApp

    settings: Settings = ctx.obj or load_settings()
    DemoApp(settings=settings).launch(host=host, port=port)


@app.command()
def info(ctx: typer.Context) -> None:
    """Show the active configuration and whether Ollama is reachable."""
    from animegen.llm.enhancer import OllamaEnhancer

    settings: Settings = ctx.obj or load_settings()
    ollama_up = OllamaEnhancer(settings=settings).is_available()

    typer.echo(f"animegen {__version__}")
    typer.echo(
        f"precision  : {LINEAR_WEIGHT_DTYPE} linear weights, "
        f"{COMPUTE_DTYPE} compute ({QUANTIZATION_BACKEND})"
    )
    typer.echo("image models:")
    for key, profile in settings.profiles.items():
        marker = " (default)" if key == settings.default_model else ""
        typer.echo(
            f"  {key:<8} {profile.name}{marker} | {profile.quality} | "
            f"{profile.default_size}, {profile.steps} steps | ~{profile.estimate}"
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
