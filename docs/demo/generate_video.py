"""
Generate FeatCopilot demo video from HTML presentation.
Captures slides and creates an MP4 video with TTS narration.
"""

import asyncio
from pathlib import Path

# Create output directory
output_dir = Path("docs/demo/video_frames")
output_dir.mkdir(exist_ok=True)

# Narration script for each slide
NARRATIONS = [
    "FeatCopilot: Next-Generation LLM-Powered Auto Feature Engineering. Transform your ML pipeline with intelligent feature generation.",
    "The Feature Engineering Problem. 80% of data science time is spent on feature engineering. It's manual, requires domain knowledge, and is extremely time consuming.",
    "FeatCopilot is the solution. It takes your raw data, processes it through intelligent engines, and produces better models. Intelligent feature engineering in seconds, not days.",
    "Key Capabilities include: Tabular Engine for polynomials and interactions, LLM Engine for semantic understanding, Human-readable Explanations, and Feature Store integration with Feast.",
    "It's this simple. Initialize the AutoFeatureEngineer with LLM support, then transform your data. That's it! Your features expand from 10 to 48 automatically.",
    "LLM-Powered Intelligence. Provide column descriptions and task context. The LLM understands your domain and creates meaningful features!",
    "Human-Readable Explanations. Get clear explanations for each generated feature, like glucose HbA1c interaction capturing overall glycemic control.",
    "Inspect Generated Code. All feature transformations are transparent, auditable, and reproducible. See exactly how each feature is calculated.",
    "Real Performance Gains. Starting from baseline, the Tabular Engine improves ROC-AUC by 7.5%, and the LLM Engine achieves 8.7% improvement. Nearly 9% gain with zero manual effort!",
    "Production-Ready with Feast. Save features to a feature store and retrieve them for inference. Go from notebook to production in minutes!",
    "100+ LLM Providers supported. Use OpenAI, Anthropic Claude, GitHub Copilot, Azure OpenAI, or local Ollama. Any LLM you prefer.",
    "Why FeatCopilot? It offers LLM-powered features, human-readable explanations, built-in feature store, semantic domain awareness, and transparent code inspection.",
    "Visual Results. See the summary of generated features and their importance.",
    "Perfect for Healthcare patient risk scoring, Finance fraud detection, E-commerce churn prediction, and Manufacturing predictive maintenance.",
    "Get Started in 30 Seconds. Install with pip, set your API key, and start transforming your data!",
    "Ready to Transform Your ML Pipeline? Install FeatCopilot today. Star us on GitHub!",
    "Thank you! FeatCopilot: Next-Generation LLM-Powered Auto Feature Engineering. Questions? Let's discuss!",
]


async def capture_slides():
    """Capture each slide as an image using Playwright."""
    from playwright.async_api import async_playwright

    slides_dir = output_dir

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1440, "height": 1080})

        # Load the presentation
        html_path = Path("docs/demo/index.html").absolute()
        await page.goto(f"file:///{html_path}")

        # Wait for reveal.js to initialize
        await page.wait_for_timeout(2000)

        # Get total number of slides
        total_slides = await page.evaluate("Reveal.getTotalSlides()")
        print(f"Total slides: {total_slides}")

        # Capture each slide
        for i in range(total_slides):
            # Navigate to slide
            await page.evaluate(f"Reveal.slide({i})")
            await page.wait_for_timeout(500)

            # Trigger all fragments on this slide
            fragments = await page.evaluate("Reveal.availableFragments().next")
            while fragments:
                await page.evaluate("Reveal.nextFragment()")
                await page.wait_for_timeout(300)
                fragments = await page.evaluate("Reveal.availableFragments().next")

            await page.wait_for_timeout(300)

            # Capture screenshot
            screenshot_path = slides_dir / f"slide_{i:02d}.png"
            await page.screenshot(path=str(screenshot_path))
            print(f"Captured slide {i + 1}/{total_slides}")

        await browser.close()

    return total_slides


async def generate_narration(total_slides):
    """Generate TTS audio for each slide using edge-tts."""
    import edge_tts

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    voice = "en-US-GuyNeural"  # Professional male voice
    durations = []

    for i in range(min(total_slides, len(NARRATIONS))):
        text = NARRATIONS[i]
        audio_path = audio_dir / f"slide_{i:02d}.mp3"

        print(f"Generating audio for slide {i + 1}...")
        communicate = edge_tts.Communicate(text, voice, rate="-5%")
        await communicate.save(str(audio_path))

        # Get audio duration
        from moviepy.editor import AudioFileClip

        audio_clip = AudioFileClip(str(audio_path))
        durations.append(audio_clip.duration)
        audio_clip.close()

    return durations


def create_video_with_audio(total_slides, audio_durations):
    """Create video from slides with narration audio."""
    from moviepy.editor import AudioFileClip, CompositeAudioClip, ImageClip, concatenate_videoclips

    slides_dir = output_dir
    audio_dir = output_dir / "audio"

    clips = []
    audio_start_time = 0.0
    audio_clips_with_start = []

    # Delay before audio starts on each slide (in seconds)
    AUDIO_DELAY = 0.8

    for i in range(total_slides):
        img_path = slides_dir / f"slide_{i:02d}.png"
        audio_path = audio_dir / f"slide_{i:02d}.mp3"

        if img_path.exists() and audio_path.exists() and i < len(audio_durations):
            # Use audio duration + buffer for slide display
            duration = audio_durations[i] + AUDIO_DELAY + 0.5

            clip = ImageClip(str(img_path)).set_duration(duration)
            clips.append(clip)

            # Load audio and set start time with delay
            audio_clip = AudioFileClip(str(audio_path))
            audio_clips_with_start.append(audio_clip.set_start(audio_start_time + AUDIO_DELAY))

            audio_start_time += duration
            print(f"Added slide {i + 1} with {duration:.1f}s duration")

    # Concatenate all video clips
    print("Concatenating video clips...")
    final_video = concatenate_videoclips(clips, method="compose")

    # Composite all audio clips with their start times
    print("Compositing audio clips...")
    final_audio = CompositeAudioClip(audio_clips_with_start)

    # Set audio to video
    final_video = final_video.set_audio(final_audio)

    # Write video file
    output_path = "docs/demo/featcopilot_demo.mp4"
    print(f"Writing video to {output_path}...")
    final_video.write_videofile(
        output_path,
        fps=24,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        bitrate="5000k",
    )

    print(f"Video created: {output_path}")
    print(f"Duration: {final_video.duration:.1f} seconds")

    # Cleanup
    for clip in audio_clips_with_start:
        clip.close()

    return output_path


def create_video(total_slides, duration_per_slide=5):
    """Create video from captured slides (no audio)."""
    from moviepy.editor import ImageClip, concatenate_videoclips

    slides_dir = output_dir

    clips = []
    for i in range(total_slides):
        img_path = slides_dir / f"slide_{i:02d}.png"
        if img_path.exists():
            if i == 0:
                duration = 4
            elif i == total_slides - 1:
                duration = 3
            elif i in [8, 11]:
                duration = 7
            else:
                duration = duration_per_slide

            clip = ImageClip(str(img_path)).set_duration(duration)
            clips.append(clip)
            print(f"Added slide {i + 1} with {duration}s duration")

    print("Concatenating clips...")
    final_video = concatenate_videoclips(clips, method="compose")

    output_path = "docs/demo/featcopilot_demo.mp4"
    print(f"Writing video to {output_path}...")
    final_video.write_videofile(output_path, fps=24, codec="libx264", audio=False, preset="medium", bitrate="5000k")

    print(f"Video created: {output_path}")
    print(f"Duration: {final_video.duration:.1f} seconds")

    return output_path


async def main(with_audio=True):
    print("=" * 50)
    print("FeatCopilot Demo Video Generator")
    print("=" * 50)

    # Step 1: Capture slides
    print("\n[1/3] Capturing slides...")
    total_slides = await capture_slides()

    if with_audio:
        # Step 2: Generate narration
        print("\n[2/3] Generating narration...")
        audio_durations = await generate_narration(total_slides)

        # Step 3: Create video with audio
        print("\n[3/3] Creating video with audio...")
        video_path = create_video_with_audio(total_slides, audio_durations)
    else:
        # Create video without audio
        print("\n[2/2] Creating video...")
        video_path = create_video(total_slides)

    print("\n" + "=" * 50)
    print("✓ Video generation complete!")
    print(f"  Output: {video_path}")
    print("=" * 50)


if __name__ == "__main__":
    import sys

    with_audio = "--no-audio" not in sys.argv
    asyncio.run(main(with_audio=with_audio))
