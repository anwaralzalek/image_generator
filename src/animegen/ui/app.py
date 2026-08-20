"""Single-worker Gradio UI backed by the shared orchestrator."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from animegen.config import (
    LINEAR_WEIGHT_DTYPE,
    MAX_IMAGES,
    SEED_MAX,
    Settings,
    get_settings,
    parse_dimensions as parse_size,
)
from animegen.core.orchestrator import Orchestrator, RunResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    import gradio as gr

LOGGER = logging.getLogger(__name__)

DEFAULT_PROMPT = "American teenagers having fun at a party"
QUEUE_MAX_SIZE = 8


def parse_seed(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Seed must be a whole number")
    text = str(value).strip()
    if not text:
        return None
    try:
        seed = int(text)
    except ValueError:
        raise ValueError(f"Seed must be a whole number, got '{text}'") from None
    if not 0 <= seed <= SEED_MAX:
        raise ValueError(f"Seed must be between 0 and {SEED_MAX}")
    return seed


class DemoApp:
    def __init__(
        self,
        settings: Settings | None = None,
        orchestrator: Orchestrator | None = None,
    ) -> None:
        self._settings: Settings = settings or get_settings()
        self._orchestrator = orchestrator or Orchestrator(settings=self._settings)
        self._lock = threading.Lock()

    def generate(
        self,
        prompt: str,
        images: int | float = 1,
        seed_text: str = "",
        size: str | None = None,
        use_llm: bool = True,
        model: str | None = None,
    ) -> tuple[list[tuple[Any, str]], str, str]:
        if not prompt or not prompt.strip():
            raise ValueError("Enter a prompt first")

        seed = parse_seed(seed_text)
        if isinstance(images, bool) or images != int(images):
            raise ValueError("Images must be a whole number")
        images = int(images)
        if not 1 <= images <= MAX_IMAGES:
            raise ValueError(f"Images must be between 1 and {MAX_IMAGES}")
        _, profile = self._settings.image_model(model)
        selected_size = size or profile.default_size
        width, height = parse_size(selected_size)
        if selected_size.lower().replace(" ", "") not in profile.allowed_sizes:
            raise ValueError(
                f"{profile.name} supports: {', '.join(profile.allowed_sizes)}"
            )

        with self._lock:
            result = self._orchestrator.run(
                prompt=prompt,
                images=images,
                seed=seed,
                width=width,
                height=height,
                model=model,
                use_llm=use_llm,
            )

        gallery = [(str(image.path), f"seed {image.seed}") for image in result.images]
        return gallery, self._details(result), self._status(result)

    def _details(self, result: RunResult) -> str:
        first = result.images[0].metadata["settings"] if result.images else {}
        seeds = ", ".join(str(seed) for seed in result.seeds)
        lines = [
            f"**Your prompt**  \n{result.original_prompt}",
            "",
            f"**Prompt sent to image model**  \n{result.enhanced_prompt}",
            "",
            f"**Negative prompt**  \n{first.get('negative_prompt', '')}",
            "",
            f"**Seeds**: {seeds}",
            "",
            f"**Settings**: {first.get('model', '?')} | "
            f"{first.get('width', '?')}x{first.get('height', '?')} | "
            f"{first.get('steps', '?')} steps | "
            f"guidance {first.get('guidance_scale', '?')} | "
            f"{first.get('scheduler', '?')} | "
            f"{first.get('linear_weight_dtype', '?')} linear weights",
            "",
            f"**Estimated render time**: {first.get('estimated_render_time', '?')} "
            "per image after model load",
            "",
            "**Files**",
            *[
                f"- `{image.path.name}` (+ sidecar `{image.metadata_path.name}`)"
                for image in result.images
            ],
        ]
        return "\n".join(lines)

    def _status(self, result: RunResult) -> str:
        if not result.llm_enabled:
            llm_state = "LLM off"
        elif result.used_fallback:
            llm_state = "LLM unreachable, used the raw prompt"
        else:
            llm_state = f"LLM {result.llm_duration_s:.1f}s"
        return (
            f"{len(result.images)} image(s) in {result.total_duration_s:.1f}s "
            f"({llm_state}, image model {result.sd_duration_s:.1f}s) - "
            f"saved to {self._settings.outputs_dir}"
        )

    def _model_description(self, model: str | None = None) -> str:
        key, profile = self._settings.image_model(model)
        return (
            f"**{profile.name}** — {profile.quality}; {profile.parameter_count} "
            f"parameters; recommended {profile.default_size} at {profile.steps} steps; "
            f"estimated **{profile.estimate}** per image. "
            f"Supported linear weights: **{LINEAR_WEIGHT_DTYPE.upper()}**. "
            f"Profile: `{key}`."
        )

    def _model_selection(
        self, model: str | None = None
    ) -> tuple[str, tuple[str, ...], str]:
        _, profile = self._settings.image_model(model)
        return (
            self._model_description(model),
            profile.allowed_sizes,
            profile.default_size,
        )

    def build(self) -> "gr.Blocks":
        import gradio as gr

        default_key, default_profile = self._settings.image_model()
        default_size = default_profile.default_size
        model_choices = [
            (
                f"{profile.quality}: {profile.name} (~{profile.estimate})",
                key,
            )
            for key, profile in self._settings.profiles.items()
        ]

        def on_model_change(selected: str) -> tuple[str, Any]:
            description, choices, value = self._model_selection(selected)
            return description, gr.Dropdown(choices=choices, value=value)

        def on_generate(
            prompt: str,
            images: float,
            seed_text: str,
            model: str,
            size: str,
            use_llm: bool,
            progress=gr.Progress(track_tqdm=True),
        ) -> tuple[list[tuple[Any, str]], str, str]:
            try:
                _, profile = self._settings.image_model(model)
                progress(
                    0,
                    desc=(
                        f"Checking {profile.name}; uncached model files will "
                        "download here"
                    ),
                )
                result = self.generate(
                    prompt, images, seed_text, size, use_llm, model
                )
                progress(1, desc="Generation complete")
                return result
            except (ValueError, RuntimeError, OSError) as exc:
                raise gr.Error(str(exc)) from exc

        with gr.Blocks(title="Image Generator") as demo:
            gr.Markdown(
                "# Image Generator\n"
                "Choose an image-quality tier with INT8 linear weights; "
                "llama3.2 optionally expands "
                "your idea on the CPU. Model use is subject to the Optional prompt enhancement"
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
                            maximum=MAX_IMAGES,
                            step=1,
                            value=1,
                        )
                        seed = gr.Textbox(
                            label="Seed", placeholder="blank = random", value=""
                        )
                    model = gr.Dropdown(
                        label="Image model", choices=model_choices, value=default_key
                    )
                    model_info = gr.Markdown(self._model_description(default_key))
                    with gr.Row():
                        size = gr.Dropdown(
                            label="Size",
                            choices=default_profile.allowed_sizes,
                            value=default_size,
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

            model.change(
                on_model_change,
                inputs=[model],
                outputs=[model_info, size],
            )
            inputs = [prompt, images, seed, model, size, use_llm]
            outputs = [gallery, details, status]
            generate_button.click(
                on_generate,
                inputs=inputs,
                outputs=outputs,
                concurrency_limit=1,
                show_progress="full",
            )
            prompt.submit(
                on_generate,
                inputs=inputs,
                outputs=outputs,
                concurrency_limit=1,
                show_progress="full",
            )

        demo.queue(max_size=QUEUE_MAX_SIZE, default_concurrency_limit=1)
        return demo

    def launch(
        self,
        host: str = "127.0.0.1",
        port: int = 7860,
        **launch_kwargs: Any,
    ) -> None:
        demo = self.build()
        allowed_paths = list(launch_kwargs.pop("allowed_paths", []) or [])
        outputs_dir = str(self._settings.outputs_dir)
        if outputs_dir not in allowed_paths:
            allowed_paths.append(outputs_dir)
        LOGGER.info("Serving the demo on http://%s:%d", host, port)
        demo.launch(
            server_name=host,
            server_port=port,
            allowed_paths=allowed_paths,
            **launch_kwargs,
        )
