"""
Generate the final demo video combining HTML slides and notebook walkthrough.

The video has three parts:
1. HTML presentation slides (intro, customer ask, gap, capabilities, benchmarks, architecture)
2. Notebook scrolling walkthrough with narration on key cells
3. Summary slide from the HTML presentation

Requires: playwright, edge-tts, moviepy
"""

import asyncio
from pathlib import Path

# Output directories
output_dir = Path("docs/demo/final_video_frames")
output_dir.mkdir(exist_ok=True)

HTML_PATH = Path("docs/demo/fabric_proposal.html")
NOTEBOOK_PATH = Path("examples/featcopilot_demo.ipynb")
FABRIC_NOTEBOOK_URL = (
    "https://dxt.powerbi.com/groups/d268e3c1-3b18-4c1d-a3fa-2808afcc23a0"
    "/synapsenotebooks/a04101d9-5ef9-4be2-be3b-ca1d2e9f4ea9?experience=fabric-developer"
)

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

# Narrations for key notebook sections, keyed by frame index.
# Frame indices are based on 86-frame capture with 300px scroll steps.
NOTEBOOK_NARRATIONS = [
    {
        "frame": 0,
        "text": "Let's now walk through the demo notebook to see FeatCopilot in action.",
    },
    {
        "frame": 5,
        "text": "We import FeatCopilot and configure the LLM backend with the gpt-5.1 model via LiteLLM for semantic feature engineering.",
    },
    {
        "frame": 10,
        "text": "We create a synthetic healthcare dataset for diabetes prediction with 2000 samples and 10 clinical features.",
    },
    {
        "frame": 23,
        "text": "The baseline with no feature engineering gives 0.7075 accuracy and 0.6958 ROC-AUC. This is our starting point.",
    },
    {
        "frame": 26,
        "text": "The tabular engine generates 45 features from 10 originals in under 2 seconds, improving ROC-AUC to 0.7263 — a 4.38% improvement with Logistic Regression.",
    },
    {
        "frame": 31,
        "text": "Now we add the LLM engine, powered by the gpt-5.1 model. It generates 56 features total including domain-aware ones. ROC-AUC reaches 0.7428, a 6.75% improvement over baseline.",
    },
    {
        "frame": 41,
        "text": "The comparison chart shows clear progression: baseline at 0.6958, tabular at 0.7263, and LLM at 0.7428 ROC-AUC.",
    },
    {
        "frame": 55,
        "text": "FLAML AutoML selects CatBoost as the best model, reaching 0.8166 ROC-AUC — a significant jump from the Logistic Regression baseline of 0.6958. FeatCopilot is most impactful when paired with simpler models, delivering a 6.75% ROC-AUC improvement.",
    },
    {
        "frame": 63,
        "text": "Features are saved to Feast feature store for production serving, with automatic materialization for real-time inference.",
    },
    {
        "frame": 80,
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
    """Capture scrolling frames from the Fabric notebook in browser."""
    from playwright.async_api import async_playwright

    frames_dir = output_dir / "notebook"
    frames_dir.mkdir(exist_ok=True)

    async with async_playwright() as p:
        # Use Edge with the user's existing profile to reuse Fabric login session.
        # All existing Edge windows must be closed before running this script.
        user_data = Path.home() / "AppData" / "Local" / "Microsoft" / "Edge" / "User Data"
        context = await p.chromium.launch_persistent_context(
            str(user_data),
            channel="msedge",
            headless=False,
            viewport={"width": 1440, "height": 1080},
        )
        page = context.pages[0] if context.pages else await context.new_page()

        print("  Navigating to Fabric notebook...")
        await page.goto(FABRIC_NOTEBOOK_URL, timeout=120000)

        # Wait for Fabric UI to fully render
        print("  Waiting for notebook to render...")
        await page.wait_for_timeout(10000)

        print("  >>> Press Enter in the terminal when the notebook is fully loaded <<<")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, input)

        # Fabric uses an internal scrollable container; use mouse wheel to scroll.
        # Place the mouse in the center of the notebook content area.
        center_x, center_y = 700, 540
        await page.mouse.move(center_x, center_y)

        # Scroll through notebook, capturing frames via mouse wheel
        scroll_step = 300  # pixels per wheel event
        frame_idx = 0
        narration_frames = {}  # frame_idx -> narration text

        # Build narration map from fixed frame indices
        narration_map = {narr["frame"]: narr["text"] for narr in NOTEBOOK_NARRATIONS}

        # Capture the initial view
        screenshot_path = frames_dir / f"frame_{frame_idx:04d}.png"
        await page.screenshot(path=str(screenshot_path))
        if frame_idx in narration_map:
            narration_frames[frame_idx] = narration_map[frame_idx]
            print(f"  Narration trigger at frame {frame_idx}: {narration_map[frame_idx][:50]}...")
        frame_idx += 1

        # Scroll until we reach the bottom (detected by no further position change)
        max_no_change = 5
        no_change_count = 0
        prev_screenshot_bytes = None

        while no_change_count < max_no_change:
            await page.mouse.wheel(0, scroll_step)
            await page.wait_for_timeout(150)

            screenshot_path = frames_dir / f"frame_{frame_idx:04d}.png"
            await page.screenshot(path=str(screenshot_path))

            # Detect end of scroll by comparing screenshots
            current_bytes = screenshot_path.read_bytes()
            if prev_screenshot_bytes and current_bytes == prev_screenshot_bytes:
                no_change_count += 1
            else:
                no_change_count = 0
            prev_screenshot_bytes = current_bytes

            # Check for narration trigger at this frame index
            if frame_idx in narration_map:
                narration_frames[frame_idx] = narration_map[frame_idx]
                print(f"  Narration trigger at frame {frame_idx}: {narration_map[frame_idx][:50]}...")

            frame_idx += 1

        # Remove trailing duplicate frames (from end-of-scroll detection)
        for i in range(no_change_count):
            dup = frames_dir / f"frame_{frame_idx - 1 - i:04d}.png"
            if dup.exists():
                dup.unlink()
        frame_idx -= no_change_count

        await context.close()

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
