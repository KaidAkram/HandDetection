import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time
import Levenshtein

# List of words for correction
words_list = ["he", "ho", "lo", "oh", "eh", "hoe", "hel", "ole", "leo", "hole", "heel", "hell", "hello"]

# Levenshtein-based closest word suggestion
def closest_word_jw(word, words_list):
    closest_match = None
    highest_similarity = 0
    for w in words_list:
        similarity = Levenshtein.jaro_winkler(word, w)
        if similarity > highest_similarity:
            highest_similarity = similarity
            closest_match = w
    return closest_match

# Text-to-speech
def speak_word(word):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1)
    voices = engine.getProperty('voices')
    for voice in voices:
        if "English" in voice.name:
            engine.setProperty('voice', voice.id)
            break
    engine.say(word)
    engine.runAndWait()

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
word = []  # Stores the recognized characters

# Video capture
cap = cv2.VideoCapture(0)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)



# Prediction labels and variables
labels_dict = {0: 'H', 1: 'E', 2: 'L', 3: 'O', 4: 'end'}
cooldown_duration = 1.5  # Min time between same character inputs
last_predicted_character = None
last_added_time = 0  # Track time of last added character

while cv2.waitKey(1) & 0xFF != ord('q'):
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    # Process landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            # Draw landmarks with rainbow colors
            
            # Draw connections in a fixed color
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                 
            )

            # Extract and normalize coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
               
            data_aux = np.asarray(data_aux)
            # Prediction with probability check
            if data_aux.shape[0] == 42:
                prediction_probs = model.predict_proba([data_aux])
                predicted_index = np.argmax(prediction_probs, axis=1)[0]
                probability = prediction_probs[0][predicted_index]

                predicted_character = None
                # Check if probability meets threshold
                if probability >= 0.9:
                    predicted_character = labels_dict[predicted_index]
                    current_time = time.time()

                    # If 'end' detected, process the word
                    if predicted_character == 'end':
                        if word:
                            recognized_word = ''.join(word)
                            print('Final word before correction:', recognized_word)
                            corrected_word = closest_word_jw(recognized_word.lower(), words_list)
                            print('Corrected word:', corrected_word)
                            speak_word(corrected_word)
                        word = []  # Reset word
                        last_predicted_character = None  # Reset character to allow new input
                    else:
                        # Add character with cooldown check
                        if predicted_character == last_predicted_character:
                            if current_time - last_added_time > cooldown_duration:
                                word.append(predicted_character)
                                last_added_time = current_time
                        else:
                            word.append(predicted_character)
                            last_predicted_character = predicted_character
                            last_added_time = current_time

                # Display bounding box and character
                x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                x2, y2 = int(max(x_) * W) - 10, int(max(y_) * H) - 10
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)

    # Display frame
    cv2.imshow('frame', frame)

    
    

cap.release()
cv2.destroyAllWindows()
