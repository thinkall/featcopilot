"""
Generate Fabric Proposal demo video from HTML presentation.
Captures slides and creates an MP4 video with TTS narration.
"""

import asyncio
from pathlib import Path

# Create output directory
output_dir = Path("docs/demo/fabric_video_frames")
output_dir.mkdir(exist_ok=True)

# Narration script for each slide
NARRATIONS = [
    "Welcome to the Auto Featurization demo for Microsoft Fabric. We'll see why auto featurization matters, explore FeatCopilot's capabilities, and walk through a live notebook demo.",
    "Enterprise customers consistently ask for a managed feature store to publish and share engineered features at scale. We're building it. But there's a deeper problem — customers struggle with creating features in the first place. According to the Anaconda State of Data Science report and the dbt Labs 2024 survey, data preparation and feature engineering remain the single largest time sink for data scientists.",
    "We have Lakehouse for raw data, Feature Store in development, and AutoML available. But what about feature creation? Auto Featurization fills this critical gap.",
    "FeatCopilot is an LLM-powered auto feature engineering framework — a fast Tabular Engine, a semantic LLM Engine with GitHub Copilot as the default AI backend, human-readable explanations, and native Feature Store integration.",
    "FeatCopilot delivers over 12% average improvement on text classification with up to 49% best case, and nearly 8% on LLM-powered regression. 12 out of 12 wins on text benchmarks.",
    "The integration flow is straightforward — OneLake, Lakehouse, FeatCopilot, Feature Store, then AutoML. All built on Python, Spark, and scikit-learn compatible APIs.",
    "Auto Featurization fills the gap in Fabric's ML pipeline. GitHub Copilot powers the LLM engine. Up to 49% improvement on text tasks and 14% combined with AutoML. Feature Store integration makes it production ready. Completing the Data Science story in Microsoft Fabric.",
]


async def capture_slides():
    """Capture each slide as an image using Playwright."""
    from playwright.async_api import async_playwright

    slides_dir = output_dir

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1440, "height": 1080})

        # Load the presentation
        html_path = Path("docs/demo/fabric_proposal.html").absolute()
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
    output_path = "docs/demo/fabric_proposal_demo.mp4"
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


async def main():
    print("=" * 60)
    print("Fabric Proposal Demo Video Generator")
    print("=" * 60)

    # Step 1: Capture slides
    print("\n[1/3] Capturing slides...")
    total_slides = await capture_slides()

    # Step 2: Generate narration
    print("\n[2/3] Generating narration...")
    audio_durations = await generate_narration(total_slides)

    # Step 3: Create video with audio
    print("\n[3/3] Creating video with audio...")
    video_path = create_video_with_audio(total_slides, audio_durations)

    print("\n" + "=" * 60)
    print("✓ Video generation complete!")
    print(f"  Output: {video_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
