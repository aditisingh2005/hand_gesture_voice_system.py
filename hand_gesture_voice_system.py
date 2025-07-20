import cv2
import mediapipe as mp
import pyttsx3

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

engine = pyttsx3.init()
engine.setProperty('rate', 140)
engine.setProperty('volume', 1.0)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id if len(voices) > 1 else voices[0].id)

emoji_map = {
    "Hello": "👋",
    "Peace": "✌️",
    "Stop": "✊",
    "Pointing": "👉",
    "Good Job": "👍",
    "No": "👎",
    "I love you": "🤟",
    "Thank you": "🙏",
    "Help": "🙌",
    "Miss you": "👋💔"
}

def detect_gesture(hand_landmarks):
    if hand_landmarks:
        lm = hand_landmarks.landmark
        fingers = []
        fingers.append(lm[4].x < lm[3].x)
        for tip in [8, 12, 16, 20]:
            fingers.append(lm[tip].y < lm[tip - 2].y)

        if fingers == [True, True, True, True, True]:
            return "Hello"
        elif fingers == [False, True, True, False, False]:
            return "Peace"
        elif fingers == [False, False, False, False, False]:
            return "Stop"
        elif fingers == [False, True, False, False, False]:
            return "Pointing"
        elif fingers == [True, False, False, False, False]:
            return "Good Job"
        elif fingers == [False, False, False, False, True]:
            return "No"
        elif fingers == [True, True, False, False, True]:
            return "I love you"
    return None

cap = cv2.VideoCapture(0)
prev_gesture = ""

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    gesture_text = None

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)
            gesture_text = detect_gesture(hand_landmark)

    if gesture_text and gesture_text != prev_gesture:
        prev_gesture = gesture_text
        if gesture_text == "I love you":
            engine.say("I love you from my heart")
        elif gesture_text == "Thank you":
            engine.say("Thank you so much, I really appreciate it")
        elif gesture_text == "Help":
            engine.say("Please help me, I need assistance")
        elif gesture_text == "Miss you":
            engine.say("I miss you a lot. Wish you were here")
        elif gesture_text == "Hello":
            engine.say("Hello! Nice to see you")
        elif gesture_text == "Peace":
            engine.say("Peace and calm to you")
        elif gesture_text == "Stop":
            engine.say("Please stop")
        elif gesture_text == "Pointing":
            engine.say("I am pointing to something")
        elif gesture_text == "Good Job":
            engine.say("You're doing a great job")
        elif gesture_text == "No":
            engine.say("No, I don’t agree")
        else:
            engine.say(gesture_text)
        engine.runAndWait()

    if gesture_text:
        emoji = emoji_map.get(gesture_text, "")
        cv2.putText(img, f"{gesture_text} {emoji}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)

    cv2.imshow("Hand Gesture to Voice", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
