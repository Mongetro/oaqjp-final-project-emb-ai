import requests
import json

def emotion_detector(text_to_analyze):
    """
    Analyzes the emotions in the given text using Watson NLP Emotion Predict service.
    Returns a formatted dictionary with emotion scores and the dominant emotion.
    
    Args:
        text_to_analyze (str): The text to analyze for emotions
        
    Returns:
        dict: A dictionary containing scores for anger, disgust, fear, joy, sadness
              and the name of the dominant emotion (the one with the highest score)
    """
    
    # API endpoint provided by the lab
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    
    # Required header to specify the model
    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }
    
    # JSON payload format expected by the API
    payload = {
        "raw_document": {
            "text": text_to_analyze
        }
    }
    
    # Send POST request to the Watson NLP service
    response = requests.post(url, json=payload, headers=headers)
    
    # Optional: basic error handling if the request fails
    if response.status_code != 200:
        return {"error": f"API returned status {response.status_code}"}
    
    # Convert the raw JSON string response into a Python dictionary
    try:
        data = json.loads(response.text)
    except json.JSONDecodeError:
        return {"error": "Failed to parse response as JSON"}
    
    # Extract the emotion scores from the nested structure
    # The path is: emotionPredictions -> first item -> emotion
    try:
        emotions = data['emotionPredictions'][0]['emotion']
        
        # Get each emotion score (they are floats between 0 and 1)
        anger   = emotions['anger']
        disgust = emotions['disgust']
        fear    = emotions['fear']
        joy     = emotions['joy']
        sadness = emotions['sadness']
        
        # Create the exact output format requested by the lab
        result = {
            'anger': anger,
            'disgust': disgust,
            'fear': fear,
            'joy': joy,
            'sadness': sadness,
            'dominant_emotion': max(
                ['anger', 'disgust', 'fear', 'joy', 'sadness'],
                key=lambda emo: emotions[emo]
            )
        }
        
        return result
    
    # In case the response structure is unexpected (rare, but good practice)
    except (KeyError, IndexError):
        return {"error": "Unexpected response format from API"}