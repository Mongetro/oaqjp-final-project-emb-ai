# Emotion Detection Web Application – NLP Project

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![Unit Tests](https://img.shields.io/badge/Tests-passing-brightgreen.svg)](https://github.com/Mongetro/oaqjp-final-project-emb-ai/actions)
[![PyLint](https://img.shields.io/badge/PyLint-10%2F10-green.svg)](https://github.com/Mongetro/oaqjp-final-project-emb-ai)

A web application that analyzes emotions in text (anger, disgust, fear, joy, sadness) and identifies the dominant emotion.

This is the **final project** for the "Python and AI" course on Coursera / IBM Skills Network.

**Example:**  
Input: "I think I am having fun" → **joy** (high score)  
Input: "I hate this situation" → **anger** (high score)

## Features

- Clean web interface with Bootstrap 4 and AJAX
- Emotion analysis with 5 categories + dominant emotion detection
- Robust error handling (empty input, API issues)
- Modular Python package (`EmotionDetection`)
- Unit tests covering all main emotions
- Code quality: PyLint score 10/10
- Local demo mode using keyword-based mock (no external API dependency)

## Technologies

- **Backend**: Python 3.10+, Flask, Requests
- **Frontend**: HTML, Bootstrap 4, vanilla JavaScript (XMLHttpRequest)
- **Testing**: unittest
- **Code quality**: PyLint
- **Local simulation**: Keyword-based mock (real IBM Watson API only works in lab)

## Project Structure

```sh

├── EmotionDetection/               # Python package with core logic
│   ├── init.py
│   └── emotion_detection.py
├── static/                         # JavaScript for frontend
│   └── mywebscript.js
├── templates/                      # HTML template
│   └── index.html
├── tests/                          # Unit tests
│   └── test_emotion_detection.py
├── server.py                       # Main Flask application
├── requirements.txt
├── .gitignore
└── README.md

```

## Local Installation & Demo

1. Clone the repository

   ```bash
   git clone https://github.com/Mongetro/oaqjp-final-project-emb-ai.git
   cd oaqjp-final-project-emb-ai

   ```

2. Create and activate a virtual environment (recommended)Bash

```sh
python3 -m venv venv
source venv/bin/activate # Linux / macOS
# or on Windows: venv\Scripts\activate
```

3. Install dependencies

```sh
pip install -r requirements.txt
```

4. Run the application

```sh
python3 server.py
```

5. Open in your browser:
   [http://127.0.0.1:5000](http://127.0.0.1:5000)

The application uses a local mock engine by default (keyword-based simulation) so it works instantly without any external API.

## Usage

1. Enter any English text in the input field
2. Click "Run Sentiment Analysis"
3. View the emotion scores and dominant emotion

Sample outputs (mock mode):

- "I think I am having fun" → joy ~0.85
- "I am so angry right now" → anger ~0.80
- "This is terrifying" → fear ~0.70
- Neutral text → neutral (all scores 0.0)

## Running Unit Tests

```sh
python3 -m unittest discover -s tests
```

All tests verify correct dominant emotion detection for classic cases.

## Important Notes

- Real IBM Watson NLP API (used in the original lab) is only accessible inside the IBM Skills Network cloud lab environment.
- For local development, portfolio showcase, or public GitHub demo, this project uses a keyword-based mock in emotion_detection.py.
- To switch to real API mode (lab only): set export USE_MOCK=false and uncomment the real API call in the code.


## Screenshots

### Joy-dominant example
<image-card alt="Joy example" src="screenshots/joy-example.png" ></image-card>

### Anger-dominant example
<image-card alt="Anger example" src="screenshots/anger-example.png" ></image-card>

### Sadness / neutral example
<image-card alt="Sadness example" src="screenshots/sadness-example.png" ></image-card>