"""
Generate FeatCopilot demo video from HTML presentation.
Captures slides and creates an MP4 video with transitions.
"""

import asyncio
from pathlib import Path

# Create output directory
output_dir = Path("docs/demo/video_frames")
output_dir.mkdir(exist_ok=True)


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


def create_video(total_slides, duration_per_slide=5):
    """Create video from captured slides using moviepy."""
    from moviepy.editor import ImageClip, concatenate_videoclips

    slides_dir = output_dir

    # Create clips for each slide
    clips = []
    for i in range(total_slides):
        img_path = slides_dir / f"slide_{i:02d}.png"
        if img_path.exists():
            # Different durations for different slides
            if i == 0:  # Title
                duration = 4
            elif i == total_slides - 1:  # Thank you
                duration = 3
            elif i in [8, 11]:  # Results tables
                duration = 7
            else:
                duration = duration_per_slide

            clip = ImageClip(str(img_path)).set_duration(duration)
            clips.append(clip)
            print(f"Added slide {i + 1} with {duration}s duration")

    # Concatenate all clips
    print("Concatenating clips...")
    final_video = concatenate_videoclips(clips, method="compose")

    # Write video file
    output_path = "docs/demo/featcopilot_demo.mp4"
    print(f"Writing video to {output_path}...")
    final_video.write_videofile(output_path, fps=24, codec="libx264", audio=False, preset="medium", bitrate="5000k")

    print(f"Video created: {output_path}")
    print(f"Duration: {final_video.duration:.1f} seconds")

    return output_path


async def main():
    print("=" * 50)
    print("FeatCopilot Demo Video Generator")
    print("=" * 50)

    # Step 1: Capture slides
    print("\n[1/2] Capturing slides...")
    total_slides = await capture_slides()

    # Step 2: Create video
    print("\n[2/2] Creating video...")
    video_path = create_video(total_slides)

    # Cleanup frames (optional)
    # shutil.rmtree(output_dir)

    print("\n" + "=" * 50)
    print("✓ Video generation complete!")
    print(f"  Output: {video_path}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
