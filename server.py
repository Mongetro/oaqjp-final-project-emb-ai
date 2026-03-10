# server.py
# Flask web application for the Emotion Detection project
# This file serves the frontend (index.html) and handles emotion analysis requests

from flask import Flask, render_template, request
# Import the emotion_detector function from our package
# Use the version that matches your __init__.py configuration
from EmotionDetection.emotion_detection import emotion_detector
# Alternative (if you used "from .emotion_detection import emotion_detector" in __init__.py):
# from EmotionDetection import emotion_detector

app = Flask("Emotion Detector")

@app.route("/")
def renderIndexPage():
    """
    Renders the main page (index.html) located in the 'templates' folder.
    This is the entry point of the web application.
    """
    return render_template('index.html')

@app.route("/emotionDetector", methods=["GET", "POST"])
def emotionDetector():
    """
    Endpoint for emotion analysis.
    Supports GET (query string from JS) and POST.
    Handles API errors gracefully to avoid 500 crashes.
    """
    # Get text from GET first (matches JS behavior)
    text_to_analyze = request.args.get("textToAnalyze")

    # Fallback to POST
    if text_to_analyze is None:
        text_to_analyze = request.form.get("textToAnalyze")

    if not text_to_analyze:
        return "Error: No text provided for analysis."

    # Call the detector
    response = emotion_detector(text_to_analyze)

    # Check if API returned an error dict
    if "error" in response:
        return f"Error from analysis API: {response['error']}. Please try again."

    # Safe access to keys (with defaults if missing)
    anger   = response.get("anger", 0.0)
    disgust = response.get("disgust", 0.0)
    fear    = response.get("fear", 0.0)
    joy     = response.get("joy", 0.0)
    sadness = response.get("sadness", 0.0)
    dominant = response.get("dominant_emotion", "unknown")

    # Format response
    formatted_response = (
        f"For the given statement, the system response is "
        f"'anger': {anger}, 'disgust': {disgust}, "
        f"'fear': {fear}, 'joy': {joy} and 'sadness': {sadness}. "
        f"The dominant emotion is {dominant}."
    )

    return formatted_response


if __name__ == "__main__":
    """
    Starts the Flask development server.
    - host="0.0.0.0" makes it accessible from outside the container (important in labs)
    - port=5000 is the required port for this project
    """
    app.run(host="0.0.0.0", port=5000)
    # You can add debug=True during development if needed:
    # app.run(host="0.0.0.0", port=5000, debug=True)