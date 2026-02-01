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
    "Auto Featurization for Microsoft Fabric. A proposal to complete the ML pipeline in Fabric Data Science by adding intelligent feature engineering alongside Feature Store and AutoML.",
    "The Customer Ask. Enterprise customers consistently rank Feature Store as a top request. We're building it. But there's a gap. Customers also struggle with creating features in the first place. 80% of data science time is spent on feature engineering.",
    "The Gap in Our Platform. We have Lakehouse for raw data, Feature Store in development, and AutoML available. But what about feature creation? Auto Featurization fills this critical gap in our pipeline.",
    "What Competitors Offer. Databricks, AWS SageMaker, and Google Vertex AI all have Feature Store, Auto Featurization, and AutoML. Microsoft Fabric has AutoML, Feature Store in development, but Auto Featurization is missing. We need it to be competitive.",
    "Proposal: Integrate FeatCopilot. An LLM-powered auto feature engineering framework with a fast Tabular Engine, semantic LLM Engine powered by Copilot, human-readable explanations, and native Feature Store integration.",
    "Proven Performance. FeatCopilot delivers 12.44% average improvement on text classification with up to 49% best case, plus 7.79% improvement on LLM-powered regression with up to 19.66% best case. All in under one second for tabular features. 12 out of 12 wins on text classification benchmarks.",
    "Fabric Integration Architecture. The flow is simple: OneLake data source, Lakehouse data prep, FeatCopilot auto features, Feature Store to manage and serve, then AutoML for model training. Seamless integration with existing Fabric components using Python, Spark, and scikit-learn compatible APIs.",
    "Simple API for Fabric Notebooks. Initialize AutoFeatureEngineer with tabular and LLM engines. Provide column descriptions and task context. Transform your data. Features expand from 8 to 42 in seconds. That's it!",
    "Native Feature Store Integration. Save generated features directly to Fabric Feature Store with one API call. Retrieve features for real-time inference. From raw data to production features in minutes!",
    "LLM-Powered Intelligence. Powered by GitHub Copilot or Azure OpenAI. Generate domain-aware features like churn risk score, purchase velocity trend, and engagement decay rate. Each feature comes with human-readable explanations.",
    "Why Now? Feature Store is in development - perfect time to add feature creation. We can leverage our Copilot investment. Competitive pressure from Databricks, AWS, and Google. And customer demand to reduce the 80% feature engineering time burden.",
    "Implementation Plan. Phase 1: Tabular Engine in Fabric Notebooks by Q2 2026. Phase 2: Feature Store Integration by Q3 2026. Phase 3: LLM Engine with Copilot by Q3 2026. Phase 4: Low-Code UI Experience by Q4 2026. Phase 5: AutoML Deep Integration by Q1 2027.",
    "Expected Impact. 80% time saved on feature engineering tasks. 5 to 20% model improvement across various ML tasks. 100% competitive parity - matching Databricks and AWS. Complete the Fabric Data Science story.",
    "Summary. Customer Need: Feature engineering is the number one pain point. Competitive Gap: We're behind Databricks, AWS, and Google. Solution Ready: FeatCopilot framework is proven. Perfect Timing: Aligns with Feature Store development. AI Synergy: Leverages our Copilot investment.",
    "Let's Complete the ML Pipeline. Lakehouse to Auto Featurization to Feature Store to AutoML. Ready to discuss next steps?",
    "Thank you! Auto Featurization for Fabric. Completing the Data Science story with intelligent feature engineering. Questions and Discussion.",
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
