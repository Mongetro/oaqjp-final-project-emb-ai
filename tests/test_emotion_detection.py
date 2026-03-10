import unittest
from EmotionDetection.emotion_detection import emotion_detector

class TestEmotionDetector(unittest.TestCase):
    """
    Unit tests for the emotion_detector function.
    Verifies that the dominant emotion matches the expected one for each test sentence.
    """

    def test_joy(self):
        """Test a joyful statement → should detect 'joy' as dominant"""
        text = "I am glad this happened"
        result = emotion_detector(text)
        self.assertEqual(result['dominant_emotion'], 'joy')

    def test_anger(self):
        """Test an angry statement → should detect 'anger' as dominant"""
        text = "I am really mad about this"
        result = emotion_detector(text)
        self.assertEqual(result['dominant_emotion'], 'anger')

    def test_disgust(self):
        """Test a disgusted statement → should detect 'disgust' as dominant"""
        text = "I feel disgusted just hearing about this"
        result = emotion_detector(text)
        self.assertEqual(result['dominant_emotion'], 'disgust')

    def test_sadness(self):
        """Test a sad statement → should detect 'sadness' as dominant"""
        text = "I am so sad about this"
        result = emotion_detector(text)
        self.assertEqual(result['dominant_emotion'], 'sadness')

    def test_fear(self):
        """Test a fearful statement → should detect 'fear' as dominant"""
        text = "I am really afraid that this will happen"
        result = emotion_detector(text)
        self.assertEqual(result['dominant_emotion'], 'fear')


if __name__ == '__main__':
    unittest.main()