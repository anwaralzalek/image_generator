# Live Demo Runbook

Assessment prompt: **"American teenagers having fun at a party"**

Everything here assumes the install in [README.md](README.md) is done and the models are
already on disk. Nothing in this runbook downloads anything.

---

## Which stack are you demoing?

Pick one **before** rehearsal and stay on it — the two paths hold the model in VRAM
independently, and running both at once is an instant OOM.

| | Local install | Docker |
|---|---|---|
| Start | `animegen ui --warmup` | `docker compose up` |
| Ollama | host service you started | bundled container, pulled at first `up` |
| Pre-flight | `animegen info` | `docker compose run --rm animegen info` |
| Risk on the day | host env drift | GPU not reaching the container |

The commands below are the local ones. For Docker, prefix with
`docker compose run --rm animegen` (drop the `animegen` word itself), e.g.
`docker compose run --rm animegen generate "..." --seed 1234`.

Docker-specific pre-flight, run once at T-30:

```bash
docker compose run --rm --no-deps animegen python -c "import torch; print(torch.cuda.is_available())"
#    -> True. False means the container has no GPU: fix it now, not on stage.

docker compose up -d && docker compose ps
#    -> ollama healthy, ollama-init exited 0, animegen running

curl -s -o /dev/null -w "%{http_code}\n" http://localhost:7860/
#    -> 200
```

## T-30 min — pre-flight

Run these in order. Each line has an expected result; if one does not match, fix it now,
not on stage.

```bash
# 1. Ollama up, model resident in RAM (this also pays the first-load cost)
ollama run llama3.2:3b "say ok"
#    -> prints something; the point is that the model is now loaded

# 2. Everything the app needs, in one line
animegen info
#    -> checkpoint : ...\models\DreamShaperXL_v2_Turbo.safetensors (found)
#    -> ollama     : http://localhost:11434 (llama3.2:3b) up

# 3. GPU is idle (close browsers, Discord, other CUDA apps first)
nvidia-smi
#    -> < 500 MB used, nothing big in the process list

# 4. Full dry run with the real prompt, warmed up
animegen generate "American teenagers having fun at a party" --seed 1234 --warmup
#    -> an image in outputs/, ~6 s after the warmup completes

# 5. Fallback path still works (pretend Ollama is dead)
animegen generate "American teenagers having fun at a party" --no-llm --seed 1234
#    -> renders fine, prints "LLM: skipped (--no-llm)"
```

Then open the outputs folder and actually look at the two images. If they are black, see
the VAE row in the README troubleshooting table — do not start the demo.

---

## Known-good seeds

> **Lock these in during rehearsal.** Seeds are only reproducible for a fixed
> prompt + model + steps + guidance + size, and the *enhanced* prompt changes every time
> the LLM runs (temperature 0.7). The seeds below are the rehearsal starting points;
> replace them with the ones that actually looked good on your machine, and record the
> enhanced prompt from the sidecar next to each.

| # | Seed | Command | Verdict (fill in at rehearsal) |
|---|------|---------|-------------------------------|
| 1 | 1234 | `animegen generate "American teenagers having fun at a party" --seed 1234 --no-llm` | |
| 2 | 2468 | `animegen generate "American teenagers having fun at a party" --seed 2468 --no-llm` | |
| 3 | 13579 | `animegen generate "American teenagers having fun at a party" --seed 13579 --no-llm` | |

**Why `--no-llm` for the locked shots:** it is the only way a seed is truly reproducible.
With the LLM in the loop, the same seed plus a different enhanced prompt gives a
different image. Use the LLM live to *show the enhancement*; use `--no-llm` when you need
a specific picture to appear.

To reproduce an exact shot from a previous run, read its sidecar and replay the final
prompt verbatim:

```bash
# outputs/20260716_143012_seed1234.json holds "final_prompt" and every setting
animegen generate "<final_prompt from the sidecar>" --no-llm --seed 1234
```

`final_prompt` already contains the style suffix; the orchestrator will not append it
twice.

---

## The demo itself

### Take 1 — the CLI, showing the LLM stage (~30 s)

```bash
animegen generate "American teenagers having fun at a party" --warmup
```

Talk over the warmup, then point at the output:

- **Prompt** — what you typed.
- **Enhanced** — what llama3.2 built from it: subjects, count, lighting, camera angle,
  clothing, plus the fixed style suffix the app owns.
- **LLM: ok (0.6s)** — and it ran on the CPU, so the GPU was free for SDXL the whole time.
- **seed 1234 outputs\...png** — every image is reproducible and has a JSON sidecar.

### Take 2 — the UI (~1 min)

```bash
animegen ui --warmup
```

Open http://127.0.0.1:7860. The prompt box is already filled with the assessment prompt.

1. Set **Images** to 2, leave **Seed** blank, click **Generate**.
2. While it runs: the queue is single-worker — one generation at a time, because 8 GB is
   8 GB.
3. Open **Prompt details and seeds** to show the exact prompt SDXL received and the seeds.
4. Uncheck **Use LLM enhancement**, generate again, and compare: the enhancement is what
   turns a one-liner into a composed scene.

### If someone asks "why is it fast?"

DreamShaper XL **v2 Turbo** at 6 steps with guidance 2.0, not 30 steps at guidance 7.5.
The Karras-sigma DPM++ scheduler holds up at that step count.

---

## If Ollama misbehaves

| What you see | What to do |
|---|---|
| `LLM: unavailable, used the raw prompt` | Nothing — it already fell back and rendered. Say so out loud; the fallback is the feature. |
| First call takes longer than usual | The model was cold. `ollama run llama3.2:3b "hi"` in another terminal, then retry. |
| Ollama is dead and will not start | Switch to `--no-llm` for the rest of the demo and use the locked seeds; show the enhanced prompt from a rehearsal sidecar instead. |
| Enhanced prompt is nonsense | Regenerate — it is temperature 0.7. If it repeats, `--no-llm` and move on. |
| UI checkbox route | Uncheck **Use LLM enhancement** — same effect as `--no-llm`, no restart needed. |

The demo never depends on Ollama being up. That is by design and is worth stating.

---

## Reset between takes

```bash
# 1. Stop the UI (Ctrl+C in its terminal) — it holds the model in VRAM
# 2. Confirm the GPU is actually free
nvidia-smi
#    -> no python process holding GBs

# 3. Move the previous take's output out of the way (keep it, do not delete)
mkdir outputs\_takes 2>nul
move outputs\*.png outputs\_takes\   &  move outputs\*.json outputs\_takes\
#    Linux/macOS: mkdir -p outputs/_takes && mv outputs/*.png outputs/*.json outputs/_takes/

# 4. Re-warm before the next take
animegen generate "American teenagers having fun at a party" --seed 1234 --no-llm --warmup
```

Docker equivalent:

```bash
docker compose down            # stops the UI and frees VRAM; volumes survive
docker compose up -d           # back up; the llama pull is a no-op the second time
docker compose logs -f animegen
```

`docker compose down` keeps `ollama-models` and `hf-cache`, so a restart costs seconds,
not a re-download. Only `docker compose down -v` throws the models away — do not type
that on demo day.

Checklist before restarting:

- [ ] Only one animegen process running (CLI **or** UI, never both — they each load a full
      copy of the model into VRAM). With Docker this includes a stray
      `docker compose run` container: check `docker compose ps`.
- [ ] `nvidia-smi` shows the GPU idle.
- [ ] Browser tabs with the old Gradio UI are closed (they hold a websocket and a
      WebGL context).
- [ ] `outputs/` is empty enough that the new file is obvious on screen.
- [ ] Terminal cleared, font size up.

---

## Emergency card

| Problem | One-liner |
|---|---|
| OOM mid-demo | `--images 1 --size 832x1216`, close the browser, retry |
| Everything is slow | you skipped `--warmup`; run it once and continue |
| Black images | wrong VAE — `python scripts/download_models.py --skip-checkpoint` |
| Ollama down | add `--no-llm` |
| UI will not start | fall back to the CLI; same code path, same results |
| Docker GPU error | `docker compose down`, run the local install instead — same commands without the compose prefix |
| Bundled Ollama unhealthy | `docker compose restart ollama`, or just add `--no-llm` |
| Nothing works | show `outputs/_takes/` from rehearsal and walk through a sidecar JSON |
