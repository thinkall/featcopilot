"""
Generate the final demo video combining HTML slides and notebook walkthrough.

The video has three parts:
1. HTML presentation slides (intro, customer ask, gap, capabilities, benchmarks, architecture)
2. Notebook scrolling walkthrough with narration on key cells
3. Summary slide from the HTML presentation

Requires: playwright, edge-tts, moviepy, jupyter/nbconvert
"""

import asyncio
import subprocess
from pathlib import Path

# Output directories
output_dir = Path("docs/demo/final_video_frames")
output_dir.mkdir(exist_ok=True)

HTML_PATH = Path("docs/demo/fabric_proposal.html")
NOTEBOOK_PATH = Path("examples/featcopilot_demo.ipynb")

# ============================================================
# Narration scripts
# ============================================================

# Narrations for each HTML slide (7 slides: intro through summary)
SLIDE_NARRATIONS = [
    "Welcome to the Auto Featurization demo for Microsoft Fabric. We'll see why auto featurization matters, explore FeatCopilot's capabilities, and walk through a live notebook demo.",
    "Enterprise customers consistently ask for a managed feature store to publish and share engineered features at scale. We're building it. But there's a deeper problem — customers struggle with creating features in the first place. According to the Anaconda State of Data Science report and the dbt Labs 2024 survey, data preparation and feature engineering remain the single largest time sink for data scientists.",
    "We have Lakehouse for raw data, Feature Store in development, and AutoML available. But what about feature creation? Auto Featurization fills this critical gap.",
    "FeatCopilot is an LLM-powered auto feature engineering framework — a fast Tabular Engine, a semantic LLM Engine with GitHub Copilot as the default AI backend, human-readable explanations, and native Feature Store integration.",
    "FeatCopilot delivers over 12% average improvement on text classification with up to 49% best case, and nearly 8% on LLM-powered regression. 12 out of 12 wins on text benchmarks.",
    "The integration flow is straightforward — OneLake, Lakehouse, FeatCopilot, Feature Store, then AutoML. All built on Python, Spark, and scikit-learn compatible APIs.",
    # Summary slide narration is placed after the notebook walkthrough
]

# The summary narration plays after the notebook walkthrough
SUMMARY_NARRATION = (
    "Auto Featurization fills the gap in Fabric's ML pipeline. "
    "GitHub Copilot powers the LLM engine, aligning with our Copilot investment. "
    "Up to 49% improvement on text tasks, 14% combined with AutoML. "
    "Feature Store integration makes it production ready. "
    "Completing the Data Science story in Microsoft Fabric."
)

# Narrations for key notebook sections (keyed by heading or cell index)
# offset_px: scroll offset from heading to trigger narration at the right visual position
NOTEBOOK_NARRATIONS = [
    {
        "trigger": "FeatCopilot Demo",
        "offset_px": 0,
        "text": "Let's now walk through the demo notebook to see FeatCopilot in action.",
    },
    {
        "trigger": "Setup & Imports",
        "offset_px": 400,
        "text": "We import FeatCopilot and configure GitHub Copilot as the default LLM backend. This aligns with our strategy of actively using GitHub Copilot across the developer experience.",
    },
    {
        "trigger": "Dataset: Healthcare",
        "offset_px": 600,
        "text": "We create a synthetic healthcare dataset for diabetes prediction with 2000 samples and 10 clinical features.",
    },
    {
        "trigger": "Baseline Model",
        "offset_px": 600,
        "text": "The baseline with no feature engineering gives 0.6425 accuracy and 0.6353 ROC-AUC. This is our starting point.",
    },
    {
        "trigger": "Feature Engineering with FeatCopilot",
        "offset_px": 700,
        "text": "The tabular engine generates 47 features from 10 originals in under one second, improving ROC-AUC to 0.6751 — a 6.25% improvement.",
    },
    {
        "trigger": "LLM Engine",
        "offset_px": 700,
        "text": "Now we add the LLM engine, powered by GitHub Copilot as the default AI backend. It generates 58 features total including domain-aware ones like metabolic syndrome score. ROC-AUC reaches 0.6742, a 6.12% improvement. This aligns with Bogdan's ask to actively use GitHub Copilot across our tools.",
    },
    {
        "trigger": "Visualize Feature Engineering",
        "offset_px": 500,
        "text": "The comparison chart shows clear progression: baseline at 0.6353, tabular at 0.6751, and LLM at 0.6742 ROC-AUC.",
    },
    {
        "trigger": "AutoML Training with FLAML",
        "offset_px": 700,
        "text": "With FLAML AutoML, LLM-engineered features achieve 0.6767 ROC-AUC — a 3% improvement over AutoML alone at 0.6568.",
    },
    {
        "trigger": "Feature Store Integration",
        "offset_px": 600,
        "text": "Features are saved to Feast feature store for production serving, with automatic materialization for real-time inference.",
    },
    {
        "trigger": "Summary & Key Takeaways",
        "offset_px": 200,
        "text": "That concludes the notebook demo.",
    },
]


async def capture_html_slides():
    """Capture slides from the HTML presentation."""
    from playwright.async_api import async_playwright

    slides_dir = output_dir / "slides"
    slides_dir.mkdir(exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1440, "height": 1080})

        html_path = HTML_PATH.absolute()
        await page.goto(f"file:///{html_path}")
        await page.wait_for_timeout(2000)

        total_slides = await page.evaluate("Reveal.getTotalSlides()")
        print(f"Total HTML slides: {total_slides}")

        for i in range(total_slides):
            await page.evaluate(f"Reveal.slide({i})")
            await page.wait_for_timeout(500)

            # Trigger all fragments
            fragments = await page.evaluate("Reveal.availableFragments().next")
            while fragments:
                await page.evaluate("Reveal.nextFragment()")
                await page.wait_for_timeout(300)
                fragments = await page.evaluate("Reveal.availableFragments().next")

            await page.wait_for_timeout(300)
            screenshot_path = slides_dir / f"slide_{i:02d}.png"
            await page.screenshot(path=str(screenshot_path))
            print(f"  Captured slide {i + 1}/{total_slides}")

        await browser.close()

    return total_slides


async def capture_notebook_scroll():
    """Convert notebook to HTML and capture scrolling frames with narration triggers."""
    from playwright.async_api import async_playwright

    frames_dir = output_dir / "notebook"
    frames_dir.mkdir(exist_ok=True)

    # Convert notebook to HTML
    nb_html_path = output_dir / "notebook_rendered.html"
    print("  Converting notebook to HTML...")
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "html",
            "--no-prompt",
            str(NOTEBOOK_PATH.absolute()),
            "--output-dir",
            str(nb_html_path.parent.absolute()),
            "--output",
            nb_html_path.stem,
        ],
        check=True,
        capture_output=True,
    )

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1440, "height": 1080})
        await page.goto(f"file:///{nb_html_path.absolute()}")
        await page.wait_for_timeout(2000)

        # Inject custom styling for light/white notebook appearance
        await page.evaluate(
            """
            document.body.style.background = '#ffffff';
            document.body.style.color = '#24292e';
            document.body.style.maxWidth = '1400px';
            document.body.style.margin = '0 auto';
            document.body.style.padding = '20px 40px';
            document.body.style.fontFamily = '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif';

            // Style code cells with light theme
            document.querySelectorAll('.jp-InputArea-editor, .highlight, pre').forEach(el => {
                el.style.background = '#f6f8fa';
                el.style.border = '1px solid #d0d7de';
                el.style.borderRadius = '8px';
                el.style.color = '#24292e';
            });

            // Style output cells
            document.querySelectorAll('.jp-OutputArea-output, .output_text, .output').forEach(el => {
                el.style.background = '#ffffff';
                el.style.color = '#24292e';
            });

            // Style headings
            document.querySelectorAll('h1, h2, h3, h4').forEach(el => {
                el.style.color = '#0078d4';
            });

            // Style tables
            document.querySelectorAll('table').forEach(el => {
                el.style.color = '#24292e';
            });
            document.querySelectorAll('th').forEach(el => {
                el.style.background = '#f0f3f6';
                el.style.color = '#24292e';
            });
            document.querySelectorAll('td').forEach(el => {
                el.style.color = '#24292e';
                el.style.borderColor = '#d0d7de';
            });
        """
        )
        await page.wait_for_timeout(500)

        # Get total page height
        total_height = await page.evaluate("document.body.scrollHeight")
        viewport_height = 1080
        print(f"  Notebook total height: {total_height}px")

        # Identify key section positions by searching for heading text
        # Apply offset_px so narration triggers when relevant output is visible
        section_positions = []
        for narr in NOTEBOOK_NARRATIONS:
            trigger = narr["trigger"]
            offset = narr.get("offset_px", 0)
            pos = await page.evaluate(
                f"""
                (() => {{
                    const elements = document.querySelectorAll('h1, h2, h3, h4, p, span');
                    for (const el of elements) {{
                        if (el.textContent.includes('{trigger}')) {{
                            return el.getBoundingClientRect().top + window.scrollY;
                        }}
                    }}
                    return -1;
                }})()
            """
            )
            # Apply offset so narration triggers when output/results are visible
            if pos >= 0:
                pos += offset
            section_positions.append({"trigger": trigger, "position": pos, "narration": narr["text"]})

        # Scroll through notebook, capturing frames
        scroll_step = 300  # pixels per frame
        frame_idx = 0
        current_scroll = 0
        narration_frames = {}  # frame_idx -> narration text
        triggered = set()  # track which narrations already fired

        # Check if we're at a narration trigger point (fire each only once)
        def check_narration_trigger(scroll_pos):
            for idx, sp in enumerate(section_positions):
                if idx not in triggered and sp["position"] >= 0 and abs(scroll_pos - sp["position"]) < scroll_step:
                    triggered.add(idx)
                    return sp["narration"]
            return None

        # Capture the initial view
        screenshot_path = frames_dir / f"frame_{frame_idx:04d}.png"
        await page.screenshot(path=str(screenshot_path))
        trigger_narr = check_narration_trigger(0)
        if trigger_narr:
            narration_frames[frame_idx] = trigger_narr
        frame_idx += 1

        while current_scroll < total_height - viewport_height:
            current_scroll += scroll_step
            await page.evaluate(f"window.scrollTo(0, {current_scroll})")
            await page.wait_for_timeout(100)

            screenshot_path = frames_dir / f"frame_{frame_idx:04d}.png"
            await page.screenshot(path=str(screenshot_path))

            trigger_narr = check_narration_trigger(current_scroll)
            if trigger_narr:
                narration_frames[frame_idx] = trigger_narr
                print(f"  Narration trigger at frame {frame_idx}: {trigger_narr[:50]}...")

            frame_idx += 1

        await browser.close()

    print(f"  Captured {frame_idx} notebook frames")
    return frame_idx, narration_frames


async def generate_tts(text, output_path):
    """Generate TTS audio for a text string."""
    import edge_tts

    voice = "en-US-GuyNeural"
    communicate = edge_tts.Communicate(text, voice, rate="-5%")
    await communicate.save(str(output_path))

    from moviepy.editor import AudioFileClip

    clip = AudioFileClip(str(output_path))
    duration = clip.duration
    clip.close()
    return duration


async def generate_all_audio(total_slides, narration_frames):
    """Generate all TTS audio files."""
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    # Slide narrations (excluding summary which is last)
    slide_durations = []
    slides_to_narrate = total_slides - 1  # last slide (summary) narrated after notebook
    for i in range(slides_to_narrate):
        if i < len(SLIDE_NARRATIONS):
            audio_path = audio_dir / f"slide_{i:02d}.mp3"
            print(f"  Generating audio for slide {i + 1}...")
            dur = await generate_tts(SLIDE_NARRATIONS[i], audio_path)
            slide_durations.append(dur)

    # Notebook narrations
    nb_audio_durations = {}
    for frame_idx, text in narration_frames.items():
        audio_path = audio_dir / f"notebook_{frame_idx:04d}.mp3"
        print(f"  Generating audio for notebook frame {frame_idx}...")
        dur = await generate_tts(text, audio_path)
        nb_audio_durations[frame_idx] = dur

    # Summary slide narration
    summary_audio_path = audio_dir / "summary.mp3"
    print("  Generating audio for summary slide...")
    summary_duration = await generate_tts(SUMMARY_NARRATION, summary_audio_path)

    return slide_durations, nb_audio_durations, summary_duration


def create_final_video(
    total_slides, slide_durations, total_nb_frames, nb_narration_frames, nb_audio_durations, summary_duration
):
    """Assemble the final video from all parts."""
    from moviepy.editor import AudioFileClip, CompositeAudioClip, ImageClip, concatenate_videoclips

    slides_dir = output_dir / "slides"
    nb_dir = output_dir / "notebook"
    audio_dir = output_dir / "audio"

    all_clips = []
    all_audio_clips = []
    current_time = 0.0
    AUDIO_DELAY = 0.8

    # ---- Part 1: HTML slides (all except summary) ----
    slides_to_show = total_slides - 1  # last is summary, shown after notebook
    for i in range(slides_to_show):
        img_path = slides_dir / f"slide_{i:02d}.png"
        audio_path = audio_dir / f"slide_{i:02d}.mp3"

        if img_path.exists() and audio_path.exists() and i < len(slide_durations):
            duration = slide_durations[i] + AUDIO_DELAY + 0.5
            clip = ImageClip(str(img_path)).set_duration(duration)
            all_clips.append(clip)

            audio_clip = AudioFileClip(str(audio_path))
            all_audio_clips.append(audio_clip.set_start(current_time + AUDIO_DELAY))
            current_time += duration
            print(f"Added slide {i + 1} ({duration:.1f}s)")

    # ---- Part 2: Notebook scrolling ----
    # Determine scroll speed: frames without narration are fast, frames with narration pause
    FPS = 24
    SCROLL_FRAME_DURATION = 1.0 / 8  # 8 scroll steps per second (smooth scroll)
    PAUSE_EXTRA = 0.5  # extra pause at narration points (beyond audio duration)

    nb_frame_files = sorted(nb_dir.glob("frame_*.png"))
    i = 0
    while i < len(nb_frame_files):
        img_path = nb_frame_files[i]

        if i in nb_narration_frames and i in nb_audio_durations:
            # Pause on this frame for narration duration
            narr_dur = nb_audio_durations[i] + AUDIO_DELAY + PAUSE_EXTRA
            clip = ImageClip(str(img_path)).set_duration(narr_dur)
            all_clips.append(clip)

            audio_path = audio_dir / f"notebook_{i:04d}.mp3"
            if audio_path.exists():
                audio_clip = AudioFileClip(str(audio_path))
                all_audio_clips.append(audio_clip.set_start(current_time + AUDIO_DELAY))

            current_time += narr_dur
        else:
            # Regular scrolling frame
            clip = ImageClip(str(img_path)).set_duration(SCROLL_FRAME_DURATION)
            all_clips.append(clip)
            current_time += SCROLL_FRAME_DURATION

        i += 1

    print(f"Added {len(nb_frame_files)} notebook frames")

    # ---- Part 3: Summary slide (last HTML slide) ----
    summary_slide_idx = total_slides - 1
    img_path = slides_dir / f"slide_{summary_slide_idx:02d}.png"
    summary_audio_path = audio_dir / "summary.mp3"

    if img_path.exists() and summary_audio_path.exists():
        duration = summary_duration + AUDIO_DELAY + 1.0
        clip = ImageClip(str(img_path)).set_duration(duration)
        all_clips.append(clip)

        audio_clip = AudioFileClip(str(summary_audio_path))
        all_audio_clips.append(audio_clip.set_start(current_time + AUDIO_DELAY))
        current_time += duration
        print(f"Added summary slide ({duration:.1f}s)")

    # ---- Assemble ----
    print("Concatenating all clips...")
    final_video = concatenate_videoclips(all_clips, method="compose")

    print("Compositing audio...")
    final_audio = CompositeAudioClip(all_audio_clips)
    final_video = final_video.set_audio(final_audio)

    output_path = "docs/demo/auto_featurization_demo.mp4"
    print(f"Writing video to {output_path}...")
    final_video.write_videofile(
        output_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        bitrate="5000k",
    )

    print(f"Video created: {output_path}")
    print(f"Total duration: {final_video.duration:.1f} seconds")

    for clip in all_audio_clips:
        clip.close()

    return output_path


async def main():
    print("=" * 60)
    print("Auto Featurization Demo - Final Video Generator")
    print("=" * 60)

    # Step 1: Capture HTML slides
    print("\n[1/4] Capturing HTML slides...")
    total_slides = await capture_html_slides()

    # Step 2: Capture notebook scrolling
    print("\n[2/4] Capturing notebook walkthrough...")
    total_nb_frames, narration_frames = await capture_notebook_scroll()

    # Step 3: Generate all audio
    print("\n[3/4] Generating narration audio...")
    slide_durations, nb_audio_durations, summary_duration = await generate_all_audio(total_slides, narration_frames)

    # Step 4: Create final video
    print("\n[4/4] Assembling final video...")
    video_path = create_final_video(
        total_slides,
        slide_durations,
        total_nb_frames,
        narration_frames,
        nb_audio_durations,
        summary_duration,
    )

    print("\n" + "=" * 60)
    print("✓ Final video generation complete!")
    print(f"  Output: {video_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
