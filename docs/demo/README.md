# FeatCopilot Demo Presentation

A professional HTML presentation for demonstrating FeatCopilot to customers.

## View the Presentation

### Option 1: Open Directly
Simply open `index.html` in your web browser.

### Option 2: Local Server (Recommended)
```bash
cd docs/demo
python -m http.server 8000
```
Then open http://localhost:8000 in your browser.

## Navigation

- **Arrow keys** or **Space**: Navigate slides
- **F**: Fullscreen mode
- **S**: Speaker notes (if added)
- **ESC**: Slide overview
- **?**: Help

## Recording a Video

### Method 1: Browser Recording (Easiest)
1. Open presentation in fullscreen (F)
2. Use OBS Studio, Loom, or browser extension to record screen
3. Advance slides manually while narrating

### Method 2: Auto-Advance Mode
Uncomment these lines in the script section of `index.html`:
```javascript
autoSlide: 5000,
autoSlideStoppable: true,
```
This will auto-advance every 5 seconds.

### Recommended Recording Settings
- Resolution: 1920x1080 (1080p)
- Frame rate: 30fps
- Format: MP4 or WebM

## Presentation Structure

| Slide | Topic | Talking Points |
|-------|-------|----------------|
| 1 | Title | Introduce FeatCopilot tagline |
| 2 | Problem | 80% time on feature engineering |
| 3 | Solution | Simple workflow visualization |
| 4 | Key Features | 4 main capabilities |
| 5 | Code Demo | Show simplicity (3 lines) |
| 6 | LLM Intelligence | Column descriptions |
| 7 | Explanations | Human-readable outputs |
| 8 | Generated Code | Transparency |
| 9 | Performance | ~9% improvement |
| 10 | Feature Store | Feast integration |
| 11 | LLM Providers | Flexibility |
| 12 | Comparison | vs competitors |
| 13 | Visual Results | Summary figure |
| 14 | Use Cases | Industry applications |
| 15 | Getting Started | Quick install |
| 16 | Call to Action | pip install |
| 17 | Thank You | Close |

## Suggested Narration Script

### Slide 1 - Title (5 sec)
"Welcome to FeatCopilot - the next-generation LLM-powered auto feature engineering framework."

### Slide 2 - Problem (15 sec)
"Data scientists spend up to 80% of their time on feature engineering. It's manual, requires deep domain knowledge, and can take days or weeks. What if AI could do this for you?"

### Slide 3 - Solution (10 sec)
"FeatCopilot transforms your raw data into better models through intelligent feature engineering - in seconds, not days."

### Slide 4 - Key Features (15 sec)
"FeatCopilot offers four key capabilities: a fast tabular engine for polynomials and interactions, an LLM engine for semantic understanding, human-readable explanations, and built-in Feast integration for production deployment."

### Slide 5 - Code Demo (15 sec)
"Using FeatCopilot is incredibly simple. Import, initialize with your preferred engines, and transform. That's it! Your 10 features become 48 engineered features automatically."

### Slide 6 - LLM Intelligence (15 sec)
"What makes FeatCopilot unique is its LLM-powered intelligence. Describe your columns and task, and the LLM understands your domain to create meaningful, relevant features."

### Slide 7 - Explanations (15 sec)
"Every generated feature comes with a human-readable explanation. Your business stakeholders can understand why glucose times HbA1c captures glycemic control, or why cholesterol ratio is a key cardiovascular indicator."

### Slide 8 - Generated Code (10 sec)
"And it's completely transparent. You can inspect the exact code that generates each feature. No black boxes - fully auditable and reproducible."

### Slide 9 - Performance (15 sec)
"In our healthcare demo, FeatCopilot improved model performance by nearly 9% - from 0.612 to 0.665 ROC-AUC - with zero manual effort. The tabular engine alone gives 7.5% improvement."

### Slide 10 - Feature Store (15 sec)
"For production, FeatCopilot integrates directly with Feast. Save your features with one line, retrieve them for inference with another. Go from notebook to production in minutes."

### Slide 11 - LLM Providers (10 sec)
"FeatCopilot supports over 100 LLM providers through LiteLLM - OpenAI, Anthropic, Azure, GitHub Copilot, even local models with Ollama. Use whatever works for your organization."

### Slide 12 - Comparison (15 sec)
"Compared to other feature engineering tools, FeatCopilot offers LLM-powered features, human-readable explanations, built-in Feast integration, domain-aware context, and full transparency."

### Slide 13 - Visual Results (10 sec)
"Here's a visual summary showing the feature generation and performance improvements across different methods."

### Slide 14 - Use Cases (10 sec)
"FeatCopilot is perfect for healthcare risk scoring, financial fraud detection, e-commerce churn prediction, and manufacturing predictive maintenance."

### Slide 15 - Getting Started (10 sec)
"Getting started takes just 30 seconds. Install with pip, set your API key, and you're ready to transform your data."

### Slide 16 - Call to Action (10 sec)
"Ready to transform your ML pipeline? Install FeatCopilot today. Star us on GitHub!"

### Slide 17 - Thank You (5 sec)
"Thank you for watching. Questions?"

## Customization

### Change Colors
Edit the CSS variables in the `<style>` section:
```css
--r-heading-color: #58a6ff;  /* Headings */
--r-link-color: #58a6ff;      /* Links */
```

### Add Your Logo
Add an image to the title slide:
```html
<img src="your-logo.png" style="height: 100px;">
```

### Modify Transitions
Change the `transition` option in the script:
```javascript
transition: 'fade',  // none, fade, slide, convex, concave, zoom
```

## Dependencies

The presentation uses CDN-hosted libraries:
- Reveal.js 4.5.0 (presentation framework)
- Highlight.js 11.8.0 (code syntax highlighting)

No local installation required!
