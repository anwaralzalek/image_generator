"""Gradio demo interface.

The UI is deliberately thin: it parses widget values and calls the same
:class:`~animegen.core.orchestrator.Orchestrator` the CLI uses, so what the
audience sees on screen is exactly what ``animegen generate`` produces.

Two constraints shape this module:

* **8 GB of VRAM**: the queue runs a single worker and a lock serialises
  generations, because two concurrent SDXL runs OOM the card instantly.
* **Demo latency**: the pipeline is loaded once at startup (optionally with a
  warmup render, toggled by ``ANIMEGEN_UI_WARMUP``) rather than on the first
  click.

``gradio`` is imported lazily so this module -- and its handler logic -- can be
imported and tested without it.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Any

from animegen.config import Settings, get_settings
from animegen.core.orchestrator import Orchestrator, RunResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    import gradio as gr

LOGGER = logging.getLogger(__name__)

DEFAULT_PROMPT = "American teenagers having fun at a party"
WARMUP_ENV_VAR = "ANIMEGEN_UI_WARMUP"
#: Beyond this the queue is telling the audience the demo is stuck.
QUEUE_MAX_SIZE = 8

_TRUTHY = {"1", "true", "yes", "on"}


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean-ish environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUTHY


def parse_seed(value: str | int | None) -> int | None:
    """Turn the seed textbox content into a seed.

    Args:
        value: Widget content; blank means "random".

    Returns:
        The seed, or None for a random one.

    Raises:
        ValueError: If the text is not a non-negative integer.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = value.strip()
    if not text:
        return None
    try:
        seed = int(text)
    except ValueError:
        raise ValueError(f"Seed must be a whole number, got '{text}'") from None
    if seed < 0:
        raise ValueError("Seed must be zero or positive")
    return seed


def parse_size(value: str) -> tuple[int, int]:
    """Turn a ``WIDTHxHEIGHT`` dropdown value into pixel dimensions.

    Raises:
        ValueError: If the value is not a valid size.
    """
    parts = value.lower().replace(" ", "").split("x")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"'{value}' is not a WIDTHxHEIGHT size")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0 or width % 8 or height % 8:
        raise ValueError(f"'{value}': dimensions must be positive multiples of 8")
    return width, height


class DemoApp:
    """Stateful holder for the Gradio demo.

    Args:
        settings: Application settings; process-wide settings when omitted.
        orchestrator: Orchestrator to drive; a default one when omitted.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        orchestrator: Orchestrator | None = None,
    ) -> None:
        self._settings: Settings = settings or get_settings()
        self._orchestrator = orchestrator or Orchestrator(settings=self._settings)
        # SDXL on 8 GB tolerates exactly one run at a time.
        self._lock = threading.Lock()

    @property
    def settings(self) -> Settings:
        """The settings backing this demo."""
        return self._settings

    @property
    def orchestrator(self) -> Orchestrator:
        """The orchestrator shared with the CLI."""
        return self._orchestrator

    def startup(self, warmup: bool | None = None) -> None:
        """Load the pipeline before the first click.

        A failure here (missing checkpoint, no GPU) is logged rather than
        raised: the UI still starts and reports the error on first use, which
        beats a dead terminal five minutes before a demo.

        Args:
            warmup: Force the warmup render on/off; defaults to the
                ``ANIMEGEN_UI_WARMUP`` environment variable.
        """
        should_warm = env_flag(WARMUP_ENV_VAR) if warmup is None else warmup
        try:
            if should_warm:
                LOGGER.info("Loading pipeline with warmup render")
                self._orchestrator.warmup()
            else:
                LOGGER.info("Loading pipeline")
                self._orchestrator.generator.load()
        except Exception as exc:  # noqa: BLE001 - startup must not kill the server
            LOGGER.warning(
                "Pipeline not ready at startup (%s: %s); it will load on first use",
                type(exc).__name__,
                exc,
            )

    def generate(
        self,
        prompt: str,
        images: int = 1,
        seed_text: str = "",
        size: str = "832x1216",
        use_llm: bool = True,
    ) -> tuple[list[tuple[Any, str]], str, str]:
        """Run one generation and format it for the widgets.

        Args:
            prompt: Prompt textbox content.
            images: Image-count slider value.
            seed_text: Seed textbox content; blank means random.
            size: Size dropdown value.
            use_llm: State of the "Use LLM enhancement" checkbox.

        Returns:
            ``(gallery_items, details_markdown, status_line)``.

        Raises:
            ValueError: On invalid widget input (blank prompt, bad seed/size).
        """
        if not prompt or not prompt.strip():
            raise ValueError("Enter a prompt first")

        seed = parse_seed(seed_text)
        width, height = parse_size(size)

        # Serialise runs: the queue already limits concurrency to 1, but a
        # second entry point (or a queue misconfiguration) must not OOM the GPU.
        with self._lock:
            result = self._orchestrator.run(
                prompt=prompt,
                images=int(images),
                seed=seed,
                width=width,
                height=height,
                use_llm=use_llm,
            )

        return self._gallery_items(result), self._details(result), self._status(result)

    @staticmethod
    def _gallery_items(result: RunResult) -> list[tuple[Any, str]]:
        """Gallery entries: the saved file plus a seed caption."""
        return [(str(image.path), f"seed {image.seed}") for image in result.images]

    def _details(self, result: RunResult) -> str:
        """Markdown for the expandable panel: prompts, seeds, settings."""
        first = result.images[0].metadata["settings"] if result.images else {}
        seeds = ", ".join(str(seed) for seed in result.seeds)
        lines = [
            f"**Your prompt**  \n{result.original_prompt}",
            "",
            f"**Prompt sent to SDXL**  \n{result.enhanced_prompt}",
            "",
            f"**Negative prompt**  \n{first.get('negative_prompt', '')}",
            "",
            f"**Seeds**: {seeds}",
            "",
            f"**Settings**: {first.get('model', '?')} | "
            f"{first.get('width', '?')}x{first.get('height', '?')} | "
            f"{first.get('steps', '?')} steps | "
            f"guidance {first.get('guidance_scale', '?')} | "
            f"{first.get('scheduler', '?')}",
            "",
            "**Files**",
            *[f"- `{image.path.name}` (+ sidecar `{image.metadata_path.name}`)"
              for image in result.images],
        ]
        return "\n".join(lines)

    def _status(self, result: RunResult) -> str:
        """One-line status: image count, LLM state and stage timings."""
        if not result.llm_enabled:
            llm_state = "LLM off"
        elif result.used_fallback:
            llm_state = "LLM unreachable, used the raw prompt"
        else:
            llm_state = f"LLM {result.llm_duration_s:.1f}s"
        return (
            f"{len(result.images)} image(s) in {result.total_duration_s:.1f}s "
            f"({llm_state}, SDXL {result.sd_duration_s:.1f}s) - "
            f"saved to {self._settings.outputs_dir}"
        )

    def build(self) -> "gr.Blocks":
        """Assemble the Blocks UI. Requires ``gradio`` to be installed."""
        import gradio as gr

        gen_cfg = self._settings.generation
        default_size = f"{gen_cfg.width}x{gen_cfg.height}"
        sizes = list(gen_cfg.allowed_sizes) or [default_size]

        def on_generate(
            prompt: str, images: float, seed_text: str, size: str, use_llm: bool
        ) -> tuple[list[tuple[Any, str]], str, str]:
            try:
                return self.generate(prompt, int(images), seed_text, size, use_llm)
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
            except FileNotFoundError as exc:
                raise gr.Error(str(exc)) from exc

        # No theme= here: Gradio 6 moved it to launch(), and warning-free
        # startup across 4.x/5.x/6.x matters more than a custom palette.
        with gr.Blocks(title="Anime Party Generator") as demo:
            gr.Markdown(
                "# Anime Party Generator\n"
                "Local 2.5D anime images: llama3.2 expands your idea on the CPU, "
                "DreamShaper XL v2 Turbo renders it on the GPU."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    prompt = gr.Textbox(
                        label="Prompt",
                        value=DEFAULT_PROMPT,
                        lines=3,
                        placeholder="Describe the scene in plain English",
                    )
                    with gr.Row():
                        images = gr.Slider(
                            label="Images",
                            minimum=1,
                            maximum=gen_cfg.max_images_per_run,
                            step=1,
                            value=gen_cfg.images_per_run,
                        )
                        seed = gr.Textbox(
                            label="Seed", placeholder="blank = random", value=""
                        )
                    with gr.Row():
                        size = gr.Dropdown(
                            label="Size",
                            choices=sizes,
                            value=default_size if default_size in sizes else sizes[0],
                        )
                        use_llm = gr.Checkbox(label="Use LLM enhancement", value=True)
                    generate_button = gr.Button("Generate", variant="primary")
                with gr.Column(scale=3):
                    gallery = gr.Gallery(
                        label="Results", columns=2, height=560, preview=True
                    )
                    status = gr.Markdown("")
                    with gr.Accordion("Prompt details and seeds", open=False):
                        details = gr.Markdown("")

            inputs = [prompt, images, seed, size, use_llm]
            outputs = [gallery, details, status]
            # concurrency_limit=1: one SDXL run at a time on an 8 GB card.
            generate_button.click(
                on_generate, inputs=inputs, outputs=outputs, concurrency_limit=1
            )
            prompt.submit(
                on_generate, inputs=inputs, outputs=outputs, concurrency_limit=1
            )

        demo.queue(max_size=QUEUE_MAX_SIZE, default_concurrency_limit=1)
        return demo

    def launch(
        self,
        host: str = "127.0.0.1",
        port: int = 7860,
        share: bool = False,
        warmup: bool | None = None,
        **launch_kwargs: Any,
    ) -> None:
        """Load the model, then serve the UI (blocking).

        Args:
            host: Interface to bind.
            port: TCP port.
            share: Request a public gradio.live tunnel.
            warmup: Override the ``ANIMEGEN_UI_WARMUP`` toggle.
            **launch_kwargs: Passed through to ``Blocks.launch``.
        """
        self.startup(warmup=warmup)
        demo = self.build()
        LOGGER.info("Serving the demo on http://%s:%d", host, port)
        demo.launch(
            server_name=host, server_port=port, share=share, **launch_kwargs
        )


def build_demo(settings: Settings | None = None) -> "gr.Blocks":
    """Build the Blocks app (used by ``gradio`` reload mode and tests)."""
    return DemoApp(settings=settings).build()


def main() -> None:
    """Serve the demo with default settings."""
    DemoApp().launch()


if __name__ == "__main__":
    main()
