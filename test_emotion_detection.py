from EmotionDetection.emotion_detection import emotion_detector
import json

# Test anger
result_anger = emotion_detector("I am really mad about this")
assert result_anger['dominant_emotion'] == 'anger'

# Test disgust
result_disgust = emotion_detector("To hear this disgusts me")
assert result_disgust['dominant_emotion'] == 'disgust'

# Test fear
result_fear = emotion_detector("I am really afraid that this will happen")
assert result_fear['dominant_emotion'] == 'fear'

# Test joy
result_joy = emotion_detector("I am glad this happened")
assert result_joy['dominant_emotion'] == 'joy'

# Test sadness
result_sadness = emotion_detector("I am so sad about this")
assert result_sadness['dominant_emotion'] == 'sadness'

print("All tests passed successfully!")