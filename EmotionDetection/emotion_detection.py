"""Emotion detection module with keyword-based mock for local/portfolio use.

This module simulates emotion analysis using simple keyword matching.
It returns scores for anger, disgust, fear, joy, sadness and determines
the dominant emotion. No external API is called in this version.

This is used for local development and portfolio demonstration since
the original IBM Watson NLP API is only available inside the IBM Skills
Network lab environment, and public alternatives have proven unstable.

For real API mode (lab only), uncomment the _real_api_analysis function.
"""

import os


def emotion_detector(text_to_analyze):
    """
    Analyzes emotions in the given text using a keyword-based mock simulation.

    Args:
        text_to_analyze (str): The text to analyze for emotions.

    Returns:
        dict: A dictionary containing:
            - 'anger': float (0.0 to 1.0)
            - 'disgust': float
            - 'fear': float
            - 'joy': float
            - 'sadness': float
            - 'dominant_emotion': str
    """
    # Check if real API mode is forced (for IBM lab environment)
    use_mock = os.getenv("USE_MOCK", "true").lower() in ["true", "1", "yes"]

    if not use_mock:
        # Optional: real API call (lab only) - uncomment if needed
        # return _real_api_analysis(text_to_analyze)
        pass

    # Use mock for local/portfolio/demo
    return _mock_emotion_analysis(text_to_analyze)


def _mock_emotion_analysis(text):
    """
    Keyword-based simulation of emotion detection.

    Returns realistic scores based on common English keywords.
    If no keywords match, returns neutral result.
    """
    text_lower = text.lower()

    # Initialize scores
    anger = disgust = fear = joy = sadness = 0.0

    # Joy keywords
    if any(word in text_lower for word in [
        'happy', 'joy', 'fun', 'love', 'great', 'awesome', 'excited',
        'good', 'nice', 'playing', 'enjoy', 'smile', 'laugh'
    ]):
        joy = 0.85

    # Anger keywords
    if any(word in text_lower for word in [
        'angry', 'mad', 'hate', 'furious', 'annoyed', 'rage', 'bad',
        'irritated', 'pissed'
    ]):
        anger = 0.80

    # Sadness keywords
    if any(word in text_lower for word in [
        'sad', 'depressed', 'lonely', 'unhappy', 'sorry', 'cry',
        'heartbroken', 'down'
    ]):
        sadness = 0.75

    # Fear keywords
    if any(word in text_lower for word in [
        'afraid', 'scared', 'fear', 'terrified', 'worried', 'anxious',
        'nervous', 'panic'
    ]):
        fear = 0.70

    # Disgust keywords
    if any(word in text_lower for word in [
        'disgust', 'gross', 'repulsive', 'sick', 'yuck', 'ew',
        'nauseous', 'hate'
    ]):
        disgust = 0.65

    # Normalize if multiple emotions detected
    total = anger + disgust + fear + joy + sadness
    if total > 1.0:
        anger /= total
        disgust /= total
        fear /= total
        joy /= total
        sadness /= total

    # Build result dictionary
    scores = {
        'anger': round(anger, 4),
        'disgust': round(disgust, 4),
        'fear': round(fear, 4),
        'joy': round(joy, 4),
        'sadness': round(sadness, 4)
    }

    # Determine dominant emotion
    if all(score == 0.0 for score in scores.values()):
        dominant = 'neutral'
    else:
        dominant = max(scores, key=scores.get)

    result = {**scores, 'dominant_emotion': dominant}

    # Debug output
    print("[DEBUG] Mock emotion analysis result:", result)

    return result


# Optional: Real API call (only works in IBM Skills Network lab)
def _real_api_analysis(text):
    """
    Placeholder for the original IBM Watson NLP API call.
    This only works inside the IBM lab environment.
    """
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    payload = {"raw_document": {"text": text}}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        emotions = data['emotionPredictions'][0]['emotion']
        result = {
            'anger': emotions['anger'],
            'disgust': emotions['disgust'],
            'fear': emotions['fear'],
            'joy': emotions['joy'],
            'sadness': emotions['sadness'],
            'dominant_emotion': max(emotions, key=emotions.get)
        }
        return result
    except Exception as e:
        return {"error": f"Real API error: {str(e)}"}