# Auto Featurization Demo Videos

Scripts for generating demo videos showcasing Auto Featurization for Microsoft Fabric.

## Quick Start — Final Demo Video

```bash
pip install playwright edge-tts moviepy nbconvert jupyter
playwright install chromium
python docs/demo/generate_final_video.py
```

Output: `docs/demo/auto_featurization_demo.mp4`

## Video Scripts

### `generate_final_video.py` — Combined Demo Video (Recommended)

Generates the full demo video with three parts:

1. **HTML slides** — Intro, customer ask, platform gap, FeatCopilot capabilities, benchmarks, architecture
2. **Notebook walkthrough** — Scrolls through `examples/featcopilot_demo.ipynb` with narration on key cells
3. **Summary slide** — Key takeaways

```bash
python docs/demo/generate_final_video.py
# Output: docs/demo/auto_featurization_demo.mp4
```

### `generate_fabric_video.py` — Slides Only

Generates a video from `fabric_proposal.html` slides only (no notebook walkthrough).

```bash
python docs/demo/generate_fabric_video.py
# Output: docs/demo/fabric_proposal_demo.mp4
```

### `generate_notebook_video.py` — Notebook Only

Generates a video from synthetic notebook-style slides with code and output.

```bash
python docs/demo/generate_notebook_video.py
# Output: docs/demo/notebook_walkthrough.mp4
```

### `generate_video.py` — Original FeatCopilot Demo

Generates a video from `index.html`, the original FeatCopilot presentation.

```bash
python docs/demo/generate_video.py           # with TTS narration
python docs/demo/generate_video.py --no-audio  # without audio
# Output: docs/demo/featcopilot_demo.mp4
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `playwright` | Captures HTML slides and scrolls notebook as screenshots |
| `edge-tts` | Generates text-to-speech narration (free, no API key) |
| `moviepy` | Assembles images and audio into MP4 video |
| `nbconvert` | Converts `.ipynb` notebook to HTML for rendering |
| `jupyter` | Required by nbconvert |

Install all at once:

```bash
pip install playwright edge-tts moviepy nbconvert jupyter
playwright install chromium
```

## Files

| File | Description |
|------|-------------|
| `fabric_proposal.html` | Reveal.js slides for the Fabric auto featurization demo |
| `index.html` | Reveal.js slides for the general FeatCopilot demo |
| `generate_final_video.py` | Combined video: slides + notebook + summary |
| `generate_fabric_video.py` | Slides-only video from `fabric_proposal.html` |
| `generate_notebook_video.py` | Notebook walkthrough video |
| `generate_video.py` | Original demo video from `index.html` |

## Customizing Narrations

Each script contains narration text as Python constants at the top of the file:

- **`SLIDE_NARRATIONS`** — One entry per HTML slide
- **`NOTEBOOK_NARRATIONS`** — Triggered at specific notebook sections, with `offset_px` to control alignment
- **`SUMMARY_NARRATION`** — Plays over the final summary slide

Edit these strings directly to change what the TTS voice says. The TTS voice (`en-US-GuyNeural`) and rate (`-5%`) can be changed in the `generate_tts` / `generate_narration` functions.

## Viewing Slides in Browser

Open `fabric_proposal.html` or `index.html` directly in a browser, or serve locally:

```bash
cd docs/demo
python -m http.server 8000
# Open http://localhost:8000/fabric_proposal.html
```

Keyboard shortcuts: **Arrow keys** to navigate, **F** for fullscreen, **ESC** for overview.
