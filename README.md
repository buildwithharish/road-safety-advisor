# 🚦 AI Road Safety Advisor

An AI-powered web application that predicts road accident severity and provides intelligent safety recommendations using Machine Learning and Google Gemini AI.

🔗 **Live App:** [Click here to open](https://road-safety-advisor-92juu9apppyrbaksxafk3qm.streamlit.app/)

---

## What it does

- Predicts accident severity (Low / Medium / High risk) based on road conditions
- Shows a color-coded risk meter with confidence percentage
- AI chatbot answers any road safety question in real time
- Built on 1.8 million real UK road accident records

---

## Screenshots

### Home Page
![Home](screenshots/home.png)

### Prediction Result
![Prediction](screenshots/prediction.png)

### AI Chatbot
![Chatbot](screenshots/chatbot.png)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| ML Model | Random Forest (Scikit-learn) |
| AI Chatbot | Google Gemini API |
| Dataset | UK Road Safety — data.gov.uk |
| Language | Python 3.10+ |
| Deployment | Streamlit Cloud |

---

## How to run locally

**1. Clone the repository**
```bash
git clone https://github.com/buildwithharish/road-safety-advisor.git
cd road-safety-advisor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your Gemini API key**

Create a file `.streamlit/secrets.toml` and add:
GEMINI_API_KEY = "your-key-here"
**4. Run the app**
```bash
streamlit run app.py
```

---

## Model Details

| | |
|---|---|
| Algorithm | Random Forest Classifier |
| Dataset | UK Road Safety (1.8M records) |
| Input features | Weather, Road Type, Light, Surface, Area |
| Output | Accident Severity (1=Low, 2=Medium, 3=High) |
| Accuracy | 90.58% |

---

## Project Structure
road-safety-advisor/
├── app.py              ← Main Streamlit application
├── model.pkl           ← Trained Random Forest model
├── requirements.txt    ← Python dependencies
├── README.md           ← Project documentation
└── screenshots/        ← App screenshots
---

## Future Scope

- Real-time weather API integration
- GPS-based location risk detection
- Mobile application
- Multi-language support (Hindi + English)
- Integration with smart city traffic systems

---

## Author

**Harish** — 1st Year AIML Student

Built as an internship project demonstrating practical application of
Machine Learning and AI in road safety.
