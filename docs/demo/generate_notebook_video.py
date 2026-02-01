"""
Generate a notebook walkthrough video showing FeatCopilot demo.
Captures key sections of the notebook with explanatory narration.
Uses syntax highlighting and includes output figures.
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
        "code": None,
        "content": """<div class="intro">
<h2>🚀 FeatCopilot Demo: LLM-Powered Auto Feature Engineering</h2>
<p>This notebook demonstrates key capabilities:</p>
<ul>
<li><strong>Tabular Feature Engineering</strong> - Automated feature generation</li>
<li><strong>LLM-Powered Features</strong> - Semantic understanding with Copilot</li>
<li><strong>Feature Store Integration</strong> - Save and serve with Feast</li>
<li><strong>AutoML Training</strong> - Train models with FLAML</li>
</ul>
<p class="highlight">Let's walk through the technical details...</p>
</div>""",
        "narration": "Welcome to the FeatCopilot demo notebook. This walkthrough covers tabular feature engineering, LLM-powered features, feature store integration, and AutoML training. Let's dive into the technical details.",
    },
    {
        "title": "Setup & Imports",
        "code": """# Import FeatCopilot
from featcopilot import AutoFeatureEngineer
from featcopilot.engines import TabularEngine
from featcopilot.selection import FeatureSelector

# Import LLM components
from featcopilot.llm import SemanticEngine

# LLM Configuration
LLM_BACKEND = 'copilot'  # or 'litellm'
LLM_MODEL = 'gpt-5.2'

print("✓ FeatCopilot with LLM support loaded")""",
        "output": "✓ FeatCopilot with LLM support loaded",
        "narration": "First, we import the core components. AutoFeatureEngineer is the main API. TabularEngine handles mathematical transformations. SemanticEngine provides LLM-powered features. We configure the LLM backend - either GitHub Copilot SDK or LiteLLM for 100+ providers.",
    },
    {
        "title": "Healthcare Dataset",
        "code": """# Create synthetic healthcare dataset
def create_healthcare_data(n_samples=2000):
    np.random.seed(42)
    data = pd.DataFrame({
        'patient_id': range(1, n_samples + 1),
        'age': np.random.randint(20, 90, n_samples),
        'bmi': np.random.normal(27, 6, n_samples),
        'bp_systolic': np.random.normal(125, 25, n_samples),
        'cholesterol_total': np.random.normal(210, 50, n_samples),
        'glucose_fasting': np.random.normal(105, 30, n_samples),
        'hba1c': np.random.normal(5.7, 1.4, n_samples),
    })
    return data

data = create_healthcare_data()
print(f"Dataset shape: {data.shape}")""",
        "output": """Dataset shape: (2000, 13)
Target distribution:
  diabetes=0: 1191 (60%)
  diabetes=1:  809 (40%)""",
        "narration": "We use a synthetic healthcare dataset for diabetes prediction. The target depends on interactions and ratios between features that simple models can't easily learn. We have 2000 samples with 12 clinical features.",
    },
    {
        "title": "Data Exploration",
        "code": """# Visualize feature distributions
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for idx, col in enumerate(['age', 'bmi', 'glucose_fasting',
                           'hba1c', 'cholesterol_total', 'bp_systolic']):
    ax = axes[idx // 3, idx % 3]
    data[col].hist(ax=ax, bins=30, alpha=0.7)
    ax.set_title(col)
plt.tight_layout()
plt.show()""",
        "image": "docs/examples/images/dataset_exploration.png",
        "narration": "Let's visualize the feature distributions. We can see age ranges from 20 to 90, BMI is normally distributed around 27, and glucose fasting shows the typical pattern for a mixed healthy and diabetic population.",
    },
    {
        "title": "Column Descriptions for LLM",
        "code": """# Define column descriptions for semantic understanding
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

task_description = "Predict diabetes risk based on patient health metrics"
print("✓ Column descriptions defined for LLM context")""",
        "output": "✓ Column descriptions defined for LLM context",
        "narration": "We define column descriptions for the LLM to understand the domain context. Each feature gets a human-readable description. We also specify the task: predict diabetes risk. This enables semantic feature generation.",
    },
    {
        "title": "Baseline Model Performance",
        "code": """# Baseline: No feature engineering
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

baseline_model = LogisticRegression(max_iter=1000)
baseline_model.fit(X_train_scaled, y_train)
baseline_pred = baseline_model.predict_proba(X_test_scaled)[:, 1]
baseline_auc = roc_auc_score(y_test, baseline_pred)

print(f"Baseline ROC-AUC: {baseline_auc:.4f}")""",
        "output": """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Baseline Results (No Feature Engineering)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Accuracy:  0.6125
  ROC-AUC:   0.6234
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""",
        "narration": "First, we establish a baseline with no feature engineering. Using logistic regression on the raw features gives us 61.25% accuracy and 0.6234 ROC-AUC. This is our starting point to measure improvement.",
    },
    {
        "title": "Tabular Feature Engineering",
        "code": """# Tabular Engine: Fast feature generation
tabular_engineer = AutoFeatureEngineer(
    engines=['tabular'],
    max_features=50,
    verbose=True
)

X_train_tab = tabular_engineer.fit_transform(X_train, y_train)
X_test_tab = tabular_engineer.transform(X_test)

print(f"Original features: {X_train.shape[1]}")
print(f"Generated features: {X_train_tab.shape[1]}")""",
        "output": """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
        "title": "Feature Engineering Comparison",
        "code": """# Compare baseline vs engineered features
results = pd.DataFrame({
    'Method': ['Baseline', 'Tabular Engine', 'LLM Engine'],
    'ROC-AUC': [0.6234, 0.6682, 0.6789],
    'Improvement': ['—', '+7.2%', '+8.9%']
})

# Plot comparison
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(results['Method'], results['ROC-AUC'])
ax.set_ylabel('ROC-AUC Score')
ax.set_title('Feature Engineering Impact')
plt.show()""",
        "image": "docs/examples/images/feature_engineering_comparison.png",
        "narration": "This chart compares the impact of feature engineering. Baseline achieves 0.62 ROC-AUC. Tabular engine improves to 0.67, and adding LLM features pushes it to 0.68. Clear improvement from automated feature engineering.",
    },
    {
        "title": "LLM-Powered Feature Engineering",
        "code": """# LLM Engine: Semantic feature generation
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
print("LLM feature engineering complete!")""",
        "output": """Generating LLM features...
  → Analyzing column semantics
  → Generating domain-aware features
  → Validating feature code
  → Selected 12 LLM features

Processing time: 34.2 seconds
✓ LLM feature engineering complete!""",
        "narration": "Now we add the LLM engine. We configure it with GitHub Copilot, specify the healthcare domain, and request up to 15 feature suggestions. We pass column descriptions and task context for semantic understanding.",
    },
    {
        "title": "LLM Feature Explanations",
        "code": """# Get LLM-generated feature explanations
explanations = llm_engineer.explain_features()

for name, explanation in list(explanations.items())[:4]:
    print(f"📊 {name}")
    print(f"   {explanation}")
    print()""",
        "output": """📊 metabolic_syndrome_score
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
        "title": "Feature Importance Analysis",
        "code": """# Analyze feature importance
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_llm, y_train)

importance_df = pd.DataFrame({
    'feature': X_train_llm.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Plot top features
plt.figure(figsize=(10, 8))
plt.barh(importance_df['feature'][:10], importance_df['importance'][:10])
plt.title('Top 10 Feature Importances')
plt.show()""",
        "image": "docs/examples/images/featcopilot_summary.png",
        "narration": "Feature importance analysis shows glucose times HbA1c as the top predictor. Four of the top ten features are LLM-generated, including glycemic control index and metabolic syndrome score. The LLM captures clinically relevant patterns.",
    },
    {
        "title": "Feature Store Integration",
        "code": """# Save to Feast Feature Store
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
    df=X_train_with_ids,
    feature_view_name='patient_diabetes_features',
    description='FeatCopilot-generated diabetes features'
)
print("✓ Features saved to Feast!")""",
        "output": """Initializing Feast feature store...
  → Created feature repo at ./feature_repo
  → Registered 42 features
  → Materializing to online store...

✓ Saved to feature view: patient_diabetes_features
✓ Materialized to online store (SQLite)""",
        "narration": "Features can be saved to Feast feature store for production use. We configure the store with entity columns and timestamps. Features are automatically materialized to the online store for real-time serving.",
    },
    {
        "title": "Feast Architecture",
        "code": """# Feature Store Architecture
#
#  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
#  │  FeatCopilot │────▶│ Offline     │────▶│  Training   │
#  │  (Generate)  │     │ Store       │     │  Pipeline   │
#  └─────────────┘     └─────────────┘     └─────────────┘
#         │                   │
#         │            ┌──────▼──────┐
#         └───────────▶│   Online    │────▶ Real-time
#                      │   Store     │      Inference
#                      └─────────────┘
""",
        "image": "docs/examples/images/feast_architecture.png",
        "narration": "The architecture shows FeatCopilot generating features that flow to both offline store for training and online store for real-time inference. Feast handles the orchestration and ensures feature consistency.",
    },
    {
        "title": "AutoML with FLAML",
        "code": """# Train with FLAML AutoML
from flaml import AutoML

automl = AutoML()
automl.fit(
    X_train_llm, y_train,
    task='classification',
    time_budget=60,  # 60 seconds
    metric='roc_auc',
    verbose=0
)

pred = automl.predict_proba(X_test_llm)[:, 1]
final_auc = roc_auc_score(y_test, pred)
print(f"Best model: {automl.best_estimator}")
print(f"Final ROC-AUC: {final_auc:.4f}")""",
        "output": """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AutoML Results (with FeatCopilot features)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Best model:    LGBMClassifier
  ROC-AUC:       0.7123 (+14.3% vs baseline)
  Training time: 58.4 seconds
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""",
        "narration": "Finally, we combine FeatCopilot with FLAML AutoML. With a 60-second time budget, AutoML finds LightGBM as the best model. Combined with engineered features, we achieve 0.7123 ROC-AUC - a 14.3% improvement over baseline.",
    },
    {
        "title": "Model Comparison",
        "code": """# Compare all approaches
results = {
    'Baseline (LogReg)': 0.6234,
    'Tabular + LogReg': 0.6682,
    'LLM + LogReg': 0.6789,
    'LLM + FLAML': 0.7123
}

plt.figure(figsize=(10, 6))
colors = ['gray', 'blue', 'purple', 'green']
plt.bar(results.keys(), results.values(), color=colors)
plt.ylabel('ROC-AUC')
plt.title('Model Performance Comparison')
plt.xticks(rotation=15)
plt.show()""",
        "image": "docs/examples/images/model_performance_comparison.png",
        "narration": "The final comparison shows the progression from baseline to fully optimized model. Each step adds value: tabular features, then LLM features, then AutoML model selection. Together they achieve 14.3% improvement.",
    },
    {
        "title": "Summary & Key Takeaways",
        "code": None,
        "content": """<div class="summary">
<h2>📊 Results Summary</h2>
<table>
<tr><th>Method</th><th>ROC-AUC</th><th>Improvement</th><th>Time</th></tr>
<tr><td>Baseline</td><td>0.6234</td><td>—</td><td>—</td></tr>
<tr><td>Tabular Engine</td><td>0.6682</td><td>+7.2%</td><td>&lt;1s</td></tr>
<tr><td>LLM Engine</td><td>0.6789</td><td>+8.9%</td><td>34s</td></tr>
<tr><td>+ AutoML (FLAML)</td><td>0.7123</td><td>+14.3%</td><td>60s</td></tr>
</table>

<h2>✓ Key Takeaways</h2>
<ul>
<li><strong>Tabular engine:</strong> Fast (&lt;1s), good improvement (+7%)</li>
<li><strong>LLM engine:</strong> Better accuracy, explainable features</li>
<li><strong>Feature Store:</strong> Production-ready serving with Feast</li>
<li><strong>AutoML:</strong> Best results with combined approach (+14%)</li>
</ul>
</div>""",
        "narration": "In summary: Tabular engine gives 7.2% improvement in under one second. LLM engine adds another 1.7% with explainable features. Combined with AutoML, we achieve 14.3% total improvement. FeatCopilot provides fast feature engineering with production-ready feature store integration.",
    },
]


# HTML template with syntax highlighting
HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github-dark.min.css">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e4e4e4;
            width: 1440px;
            height: 1080px;
            padding: 40px;
            overflow: hidden;
        }}
        .header {{
            display: flex;
            align-items: center;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 2px solid #4a90d9;
        }}
        .cell-number {{
            background: #4a90d9;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin-right: 20px;
            font-size: 18px;
        }}
        .title {{
            color: #4a90d9;
            font-size: 32px;
            font-weight: 600;
        }}
        .code-cell {{
            background: #0d1117;
            border-radius: 10px;
            margin-bottom: 20px;
            overflow: hidden;
            border: 1px solid #30363d;
        }}
        .code-header {{
            background: #161b22;
            padding: 10px 15px;
            font-size: 14px;
            color: #8b949e;
            border-bottom: 1px solid #30363d;
        }}
        .code-content {{
            padding: 20px;
            font-family: 'Fira Code', 'Consolas', monospace;
            font-size: 18px;
            line-height: 1.5;
            overflow: hidden;
        }}
        .code-content pre {{
            margin: 0;
            white-space: pre-wrap;
        }}
        .output-cell {{
            background: #161b22;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 4px solid #3fb950;
            font-family: 'Consolas', monospace;
            font-size: 17px;
            line-height: 1.5;
            color: #c9d1d9;
        }}
        .output-header {{
            color: #3fb950;
            font-size: 14px;
            margin-bottom: 10px;
            font-weight: 600;
        }}
        .image-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .image-container img {{
            max-width: 100%;
            max-height: 500px;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        .intro, .summary {{
            padding: 30px;
            font-size: 22px;
            line-height: 1.8;
        }}
        .intro h2, .summary h2 {{
            color: #4a90d9;
            margin-bottom: 20px;
            font-size: 36px;
        }}
        .intro ul, .summary ul {{
            margin-left: 30px;
            margin-top: 15px;
        }}
        .intro li, .summary li {{
            margin-bottom: 12px;
        }}
        .intro .highlight {{
            color: #f0883e;
            font-size: 24px;
            margin-top: 30px;
        }}
        .summary table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 20px;
        }}
        .summary th, .summary td {{
            padding: 15px 20px;
            text-align: left;
            border: 1px solid #30363d;
        }}
        .summary th {{
            background: #238636;
            color: white;
        }}
        .summary tr:nth-child(even) {{
            background: rgba(255,255,255,0.05);
        }}
        /* Syntax highlighting overrides */
        .hljs {{
            background: transparent !important;
            padding: 0 !important;
        }}
    </style>
</head>
<body>
    <div class="header">
        <span class="cell-number">{cell_num}</span>
        <span class="title">{title}</span>
    </div>
    {content}
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/highlight.min.js"></script>
    <script>hljs.highlightAll();</script>
</body>
</html>"""


def generate_slide_content(section, idx):
    """Generate HTML content for a slide."""
    content_parts = []

    if section.get("code"):
        # Code cell with syntax highlighting
        code_html = f"""
        <div class="code-cell">
            <div class="code-header">In [{idx + 1}]:</div>
            <div class="code-content">
                <pre><code class="language-python">{section['code']}</code></pre>
            </div>
        </div>"""
        content_parts.append(code_html)

    if section.get("content"):
        # Custom HTML content (intro/summary slides)
        content_parts.append(section["content"])

    if section.get("output"):
        # Output cell
        output_html = f"""
        <div class="output-cell">
            <div class="output-header">Output:</div>
            <pre>{section['output']}</pre>
        </div>"""
        content_parts.append(output_html)

    if section.get("image"):
        # Image - will be handled separately
        content_parts.append('<div class="image-container"><img id="chart-image" /></div>')

    return "\n".join(content_parts)


async def generate_slides():
    """Generate slide images from notebook sections."""
    import base64

    from playwright.async_api import async_playwright

    slides_dir = output_dir

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1440, "height": 1080})

        for i, section in enumerate(NOTEBOOK_SECTIONS):
            content = generate_slide_content(section, i)
            html = HTML_TEMPLATE.format(
                cell_num=f"[{i + 1}]" if section.get("code") else "📓",
                title=section["title"],
                content=content,
            )

            # Load HTML
            await page.set_content(html)
            await page.wait_for_timeout(1000)  # Wait for highlight.js

            # If there's an image, load it
            if section.get("image"):
                image_path = Path(section["image"]).absolute()
                if image_path.exists():
                    with open(image_path, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode()
                    ext = image_path.suffix[1:]
                    await page.evaluate(
                        f"""
                        document.getElementById("chart-image").src = "data:image/{ext};base64,{img_data}";
                    """
                    )
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
    print("Notebook Walkthrough Video Generator (Enhanced)")
    print("=" * 60)

    print("\n[1/3] Generating slides with syntax highlighting...")
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
