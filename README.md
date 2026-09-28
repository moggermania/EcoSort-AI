# ♻️ Paras — Smart Waste Segregation & Recycling Assistant

**1M1B AI for Sustainability Virtual Internship — July–September 2026**

## Project Overview
Paras is a beginner-friendly AI/ML sustainability prototype aligned with **UN SDG 12: Responsible Consumption and Production**. It classifies everyday waste descriptions into useful waste categories and provides disposal guidance and an eco tip.

The project uses:
- Python
- Streamlit
- Pandas
- Scikit-learn
- TF-IDF text features
- Logistic Regression
- Optional Gemini generative AI layer

## Features
- Waste description input
- AI/ML category prediction
- Material identification from prototype knowledge
- Disposal guidance
- Sustainability tip
- Dataset explorer and category chart
- Responsible AI section
- Optional Gemini AI response layer
- Offline core functionality (no API key required)

## Project Structure
```text
Paras/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── waste_items.csv
├── docs/
│   ├── project_report.md
│   └── presentation_outline.md
└── .gitignore
```

## Run Locally
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Start the app:
```bash
streamlit run app.py
```

## Optional Generative AI
The project works without an API key. To enable the optional Gemini layer, set:
```text
GEMINI_API_KEY=your_key_here
```
Do not commit API keys to GitHub.

## SDG Alignment
Primary SDG: **SDG 12 — Responsible Consumption and Production**

The project supports awareness and decision support around waste segregation, reuse, recycling, and appropriate handling.

## Important Prototype Limitation
Waste-collection rules differ by location. Paras therefore presents guidance as informational decision support and asks users to verify local authority instructions, especially for hazardous waste and e-waste.

## Responsible AI
The project addresses:
- Fairness
- Transparency
- Ethics
- Privacy
- Safety and uncertainty

## Author
**Parul**  
B.Tech CSE  
Sardar Beant Singh State University, Gurdaspur
