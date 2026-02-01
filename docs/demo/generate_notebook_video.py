"""
Generate a notebook walkthrough video showing FeatCopilot demo.
Captures key sections of the notebook with explanatory narration.
"""

import asyncio
from pathlib import Path

# Create output directory
output_dir = Path("docs/demo/notebook_video_frames")
output_dir.mkdir(exist_ok=True)

# Key sections to capture from the notebook with their narrations
NOTEBOOK_SECTIONS = [
    {
        "title": "FeatCopilot Demo Notebook",
        "content": """# FeatCopilot Demo: LLM-Powered Auto Feature Engineering

This notebook demonstrates key capabilities:

1. **Tabular Feature Engineering** - Automated feature generation
2. **LLM-Powered Features** - Semantic understanding
3. **Feature Store Integration** - Save and serve with Feast
4. **AutoML Training** - Train models with FLAML

Let's walk through the technical details...""",
        "narration": "Welcome to the FeatCopilot demo notebook. This walkthrough covers tabular feature engineering, LLM-powered features, feature store integration, and AutoML training. Let's dive into the technical details.",
    },
    {
        "title": "Setup & Imports",
        "content": """# Import FeatCopilot
from featcopilot import AutoFeatureEngineer
from featcopilot.engines import TabularEngine
from featcopilot.selection import FeatureSelector

# Import LLM components
from featcopilot.llm import SemanticEngine

# LLM Configuration
LLM_BACKEND = 'copilot'  # or 'litellm'
LLM_MODEL = 'gpt-5.2'

✓ FeatCopilot with LLM support loaded""",
        "narration": "First, we import the core components. AutoFeatureEngineer is the main API. TabularEngine handles mathematical transformations. SemanticEngine provides LLM-powered features. We configure the LLM backend - either GitHub Copilot SDK or LiteLLM for 100+ providers.",
    },
    {
        "title": "Healthcare Dataset",
        "content": """# Dataset: Healthcare Diabetes Prediction
# Target depends on interactions and ratios

Dataset shape: (2000, 13)

Features:
- age, bmi, bp_systolic, bp_diastolic
- cholesterol_total, cholesterol_hdl
- glucose_fasting, hba1c
- smoking_years, exercise_weekly

Target distribution:
  diabetes=0: 1191 (60%)
  diabetes=1:  809 (40%)""",
        "narration": "We use a synthetic healthcare dataset for diabetes prediction. The target depends on interactions and ratios between features that simple models can't easily learn. We have 2000 samples with 12 clinical features.",
    },
    {
        "title": "Column Descriptions for LLM",
        "content": """# Define column descriptions for semantic understanding
column_descriptions = {
    'age': 'Patient age in years',
    'bmi': 'Body Mass Index (kg/m²)',
    'bp_systolic': 'Systolic blood pressure (mmHg)',
    'bp_diastolic': 'Diastolic blood pressure (mmHg)',
    'cholesterol_total': 'Total cholesterol (mg/dL)',
    'cholesterol_hdl': 'HDL good cholesterol (mg/dL)',
    'glucose_fasting': 'Fasting blood glucose (mg/dL)',
    'hba1c': 'HbA1c percentage (3-month average)',
    'smoking_years': 'Years of smoking',
    'exercise_weekly': 'Hours of exercise per week'
}

task_description = "Predict diabetes risk" """,
        "narration": "We define column descriptions for the LLM to understand the domain context. Each feature gets a human-readable description. We also specify the task: predict diabetes risk. This enables semantic feature generation.",
    },
    {
        "title": "Baseline Model Performance",
        "content": """# Baseline: No feature engineering
X_train, X_test, y_train, y_test = train_test_split(...)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

baseline_model = LogisticRegression()
baseline_model.fit(X_train_scaled, y_train)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Baseline Results (No Feature Engineering)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Accuracy:  0.6125
  ROC-AUC:   0.6234
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""",
        "narration": "First, we establish a baseline with no feature engineering. Using logistic regression on the raw features gives us 61.25% accuracy and 0.6234 ROC-AUC. This is our starting point to measure improvement.",
    },
    {
        "title": "Tabular Feature Engineering",
        "content": """# Tabular Engine: Fast feature generation
tabular_engineer = AutoFeatureEngineer(
    engines=['tabular'],
    max_features=50,
    verbose=True
)

X_train_tab = tabular_engineer.fit_transform(X_train, y_train)
X_test_tab = tabular_engineer.transform(X_test)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tabular Engine Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Original features:    10
  Generated features:   50
  Processing time:      0.42 seconds

  Accuracy:  0.6575 (+7.3%)
  ROC-AUC:   0.6682 (+7.2%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""",
        "narration": "The Tabular Engine generates features in under one second. It creates polynomials, interactions, and ratios. From 10 original features, we get 50 engineered features. Accuracy improves by 7.3% and ROC-AUC by 7.2%.",
    },
    {
        "title": "Generated Features Preview",
        "content": """# Sample generated features:

Polynomial Features:
  - age_squared
  - bmi_squared
  - glucose_fasting_squared

Interaction Features:
  - glucose_fasting_x_hba1c       ← Key predictor!
  - bmi_x_glucose_fasting
  - age_x_smoking_years

Ratio Features:
  - cholesterol_total_div_hdl    ← Clinical ratio!
  - bp_systolic_div_bp_diastolic
  - glucose_fasting_div_exercise

Transform Features:
  - log_bmi, sqrt_age, exp_hba1c""",
        "narration": "Let's look at the generated features. Polynomial features capture non-linear relationships. Interaction features like glucose times HbA1c are key diabetes predictors. Ratio features like cholesterol total divided by HDL are clinically meaningful.",
    },
    {
        "title": "LLM-Powered Feature Engineering",
        "content": """# LLM Engine: Semantic feature generation
llm_engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    max_features=40,
    llm_config={
        'model': 'gpt-5.2',
        'backend': 'copilot',
        'domain': 'healthcare',
        'max_suggestions': 15
    }
)

X_train_llm = llm_engineer.fit_transform(
    X_train, y_train,
    column_descriptions=column_descriptions,
    task_description="Predict diabetes risk"
)

Processing with LLM... (30-60 seconds)""",
        "narration": "Now we add the LLM engine. We configure it with GitHub Copilot, specify the healthcare domain, and request up to 15 feature suggestions. We pass column descriptions and task context for semantic understanding.",
    },
    {
        "title": "LLM Feature Explanations",
        "content": """# LLM-generated feature explanations
explanations = llm_engineer.explain_features()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LLM Feature Explanations
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 metabolic_syndrome_score
   Combines BMI, blood pressure, and glucose to create
   a composite metabolic syndrome risk indicator.

📊 glycemic_control_index
   Product of fasting glucose and HbA1c captures
   overall glycemic control. High values = diabetes risk.

📊 cardiovascular_risk_ratio
   Ratio of total cholesterol to HDL - a key
   cardiovascular risk indicator used by clinicians.

📊 lifestyle_risk_factor
   Combines smoking years and exercise inversely -
   high smoking + low exercise increases risk.""",
        "narration": "The LLM provides human-readable explanations for each feature. Metabolic syndrome score combines multiple risk factors. Glycemic control index captures overall glucose management. These are domain-aware, clinically meaningful features.",
    },
    {
        "title": "LLM Engine Results",
        "content": """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LLM Engine Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Original features:    10
  Generated features:   42
  LLM suggestions:      12
  Processing time:      34.2 seconds

  Accuracy:  0.6650 (+8.6%)
  ROC-AUC:   0.6789 (+8.9%)  ← Best!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Improvement over Tabular only:
  Accuracy: +1.1%
  ROC-AUC:  +1.6%""",
        "narration": "The LLM engine achieves the best results: 8.6% accuracy improvement and 8.9% ROC-AUC improvement. The LLM added 12 semantic features on top of tabular features, providing additional 1-2% gain.",
    },
    {
        "title": "Feature Importance Analysis",
        "content": """# Top 10 Most Important Features

 Rank  Feature                        Importance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1    glucose_fasting_x_hba1c        0.187
  2    glycemic_control_index (LLM)   0.156
  3    metabolic_syndrome_score (LLM) 0.098
  4    cholesterol_total_div_hdl      0.087
  5    bmi_x_glucose_fasting          0.076
  6    cardiovascular_risk (LLM)      0.065
  7    age_x_smoking_years            0.054
  8    hba1c_squared                  0.048
  9    lifestyle_risk_factor (LLM)    0.041
 10    glucose_fasting_squared        0.038

LLM features: 4 of top 10!""",
        "narration": "Feature importance analysis shows glucose times HbA1c as the top predictor. Four of the top ten features are LLM-generated, including glycemic control index and metabolic syndrome score. The LLM captures clinically relevant patterns.",
    },
    {
        "title": "Feature Store Integration",
        "content": """# Save to Feast Feature Store
from featcopilot.stores import FeastFeatureStore

store = FeastFeatureStore(
    repo_path='./feature_repo',
    project_name='diabetes_prediction',
    entity_columns=['patient_id'],
    timestamp_column='event_timestamp',
    auto_materialize=True
)

store.initialize()
store.save_features(
    df=X_train_with_features,
    feature_view_name='patient_diabetes_features',
    description='FeatCopilot-generated diabetes features'
)

✓ Saved to feature view: patient_diabetes_features
✓ Materialized to online store""",
        "narration": "Features can be saved to Feast feature store for production use. We configure the store with entity columns and timestamps. Features are automatically materialized to the online store for real-time serving.",
    },
    {
        "title": "Online Feature Retrieval",
        "content": """# Real-time inference with online store
online_features = store.get_online_features(
    entity_dict={'patient_id': [1001, 1002, 1003]},
    feature_names=[
        'glucose_fasting_x_hba1c',
        'glycemic_control_index',
        'metabolic_syndrome_score',
        'cholesterol_total_div_hdl'
    ],
    feature_view_name='patient_diabetes_features'
)

Result: {
  'patient_id': [1001, 1002, 1003],
  'glucose_fasting_x_hba1c': [584.2, 612.8, 498.5],
  'glycemic_control_index': [1.24, 1.38, 0.98],
  'metabolic_syndrome_score': [0.72, 0.81, 0.45],
  ...
}

Low-latency retrieval for production inference!""",
        "narration": "For real-time inference, we retrieve features from the online store. Pass patient IDs and feature names, get values instantly. This enables low-latency production inference with pre-computed features.",
    },
    {
        "title": "AutoML Integration with FLAML",
        "content": """# Train with FLAML AutoML
from flaml import AutoML

automl = AutoML()
automl.fit(
    X_train_features, y_train,
    task='classification',
    time_budget=60,  # 60 seconds
    metric='roc_auc',
    verbose=0
)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AutoML Results (with FeatCopilot features)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Best model:    LGBMClassifier
  ROC-AUC:       0.7123 (+14.3% vs baseline)
  Training time: 58.4 seconds
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""",
        "narration": "Finally, we combine FeatCopilot with FLAML AutoML. With a 60-second time budget, AutoML finds LightGBM as the best model. Combined with engineered features, we achieve 0.7123 ROC-AUC - a 14.3% improvement over baseline.",
    },
    {
        "title": "Summary & Results Comparison",
        "content": """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
         RESULTS SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Method               ROC-AUC   Improvement  Time
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Baseline             0.6234      -           -
Tabular Engine       0.6682    +7.2%       <1s
LLM Engine           0.6789    +8.9%       34s
+ AutoML (FLAML)     0.7123    +14.3%      60s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Key Takeaways:
✓ Tabular engine: Fast (<1s), good improvement
✓ LLM engine: Better accuracy, explainable features
✓ Feature Store: Production-ready serving
✓ AutoML: Best results with combined approach""",
        "narration": "In summary: Tabular engine gives 7.2% improvement in under one second. LLM engine adds another 1.7% with explainable features. Combined with AutoML, we achieve 14.3% total improvement. FeatCopilot provides fast feature engineering with production-ready feature store integration.",
    },
]


async def generate_slides():
    """Generate slide images from notebook sections."""
    from playwright.async_api import async_playwright

    slides_dir = output_dir

    # Create HTML template for each slide
    html_template = """<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: 'Consolas', 'Monaco', monospace;
            background: #1e1e1e;
            color: #d4d4d4;
            margin: 0;
            padding: 40px;
            width: 1360px;
            height: 1000px;
            box-sizing: border-box;
        }}
        .title {{
            color: #569cd6;
            font-size: 32px;
            margin-bottom: 30px;
            border-bottom: 2px solid #569cd6;
            padding-bottom: 10px;
        }}
        .content {{
            font-size: 22px;
            line-height: 1.6;
            white-space: pre-wrap;
        }}
        .highlight {{ color: #4ec9b0; }}
        .string {{ color: #ce9178; }}
        .keyword {{ color: #c586c0; }}
        .number {{ color: #b5cea8; }}
        .comment {{ color: #6a9955; }}
    </style>
</head>
<body>
    <div class="title">{title}</div>
    <div class="content">{content}</div>
</body>
</html>"""

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1440, "height": 1080})

        for i, section in enumerate(NOTEBOOK_SECTIONS):
            # Create HTML for this slide
            html_content = html_template.format(title=section["title"], content=section["content"])

            # Load HTML
            await page.set_content(html_content)
            await page.wait_for_timeout(500)

            # Capture screenshot
            screenshot_path = slides_dir / f"slide_{i:02d}.png"
            await page.screenshot(path=str(screenshot_path))
            print(f"Captured slide {i + 1}/{len(NOTEBOOK_SECTIONS)}: {section['title']}")

        await browser.close()

    return len(NOTEBOOK_SECTIONS)


async def generate_narration(total_slides):
    """Generate TTS audio for each slide."""
    import edge_tts

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    voice = "en-US-GuyNeural"
    durations = []

    for i, section in enumerate(NOTEBOOK_SECTIONS):
        text = section["narration"]
        audio_path = audio_dir / f"slide_{i:02d}.mp3"

        print(f"Generating audio for slide {i + 1}...")
        communicate = edge_tts.Communicate(text, voice, rate="-5%")
        await communicate.save(str(audio_path))

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

    AUDIO_DELAY = 0.8

    for i in range(total_slides):
        img_path = slides_dir / f"slide_{i:02d}.png"
        audio_path = audio_dir / f"slide_{i:02d}.mp3"

        if img_path.exists() and audio_path.exists() and i < len(audio_durations):
            duration = audio_durations[i] + AUDIO_DELAY + 0.5

            clip = ImageClip(str(img_path)).set_duration(duration)
            clips.append(clip)

            audio_clip = AudioFileClip(str(audio_path))
            audio_clips_with_start.append(audio_clip.set_start(audio_start_time + AUDIO_DELAY))

            audio_start_time += duration
            print(f"Added slide {i + 1} with {duration:.1f}s duration")

    print("Concatenating video clips...")
    final_video = concatenate_videoclips(clips, method="compose")

    print("Compositing audio clips...")
    final_audio = CompositeAudioClip(audio_clips_with_start)

    final_video = final_video.set_audio(final_audio)

    output_path = "docs/demo/notebook_walkthrough.mp4"
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

    for clip in audio_clips_with_start:
        clip.close()

    return output_path


async def main():
    print("=" * 60)
    print("Notebook Walkthrough Video Generator")
    print("=" * 60)

    print("\n[1/3] Generating slides...")
    total_slides = await generate_slides()

    print("\n[2/3] Generating narration...")
    audio_durations = await generate_narration(total_slides)

    print("\n[3/3] Creating video...")
    video_path = create_video_with_audio(total_slides, audio_durations)

    print("\n" + "=" * 60)
    print("✓ Video generation complete!")
    print(f"  Output: {video_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
