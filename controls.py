import cv2
import mediapipe as mp
import pyautogui
import math
import time

# --- Configuration ---
# You can tweak these values to change sensitivity

# Fist detection sensitivity
# A fist is detected if the finger tips are below (higher y-value) their middle joints.
# You can add a small offset to make it more or less sensitive.
FIST_DETECTION_OFFSET = 0.05 

# Roll control (tilting hands)
# This is the angle in radians for tilting. Larger value = more tilt needed.
TILT_THRESHOLD = 0.3 # About 17 degrees

# Speed control (distance between hands)
# Distances are normalized (0.0 to ~1.4). Experiment to find what feels right.
DISTANCE_SPEED_UP = 0.2  # Speed up when hands are closer than this
DISTANCE_SLOW_DOWN = 0.6 # Slow down when hands are farther than this
SPEED_COMMAND_COOLDOWN = 1.0 # Seconds between speed commands

# --- Initialization ---
pyautogui.FAILSAFE = False
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

keys_pressed = set()
last_speed_command_time = 0

print("New Control System Initialized. Switch to your game window!")
print("Controls: Both Fists=Neutral | Tilt Hands=Roll | Hand Distance=Speed")

# --- Helper Function to Detect a Fist ---
def is_fist(hand_landmarks):
    """Checks if a hand is making a fist."""
    # We check if the tips of the 4 fingers are below their PIP joints (middle knuckle)
    # A lower Y coordinate means the point is higher up on the screen.
    try:
        index_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        index_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y

        middle_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
        middle_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y

        ring_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
        ring_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y

        pinky_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
        pinky_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y

        # If all finger tips are lower than their middle joints, it's a fist.
        if (index_tip_y > index_pip_y + FIST_DETECTION_OFFSET and
            middle_tip_y > middle_pip_y + FIST_DETECTION_OFFSET and
            ring_tip_y > ring_pip_y + FIST_DETECTION_OFFSET and
            pinky_tip_y > pinky_pip_y + FIST_DETECTION_OFFSET):
            return True
    except:
        return False
    return False


# --- Main Loop ---
try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)
        
        current_keys = set() # Keys that should be pressed in this frame

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            # We have two hands, proceed with logic
            hand1, hand2 = results.multi_hand_landmarks

            # Draw landmarks for visualization
            mp_drawing.draw_landmarks(image, hand1, mp_hands.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, hand2, mp_hands.HAND_CONNECTIONS)

            # --- Control Logic ---
            is_hand1_fist = is_fist(hand1)
            is_hand2_fist = is_fist(hand2)

            if is_hand1_fist and is_hand2_fist:
                # NEUTRAL STATE: Both hands are fists
                # No keys should be pressed.
                pass # current_keys remains empty
            else:
                # ACTIVE STATE: At least one hand is open
                wrist1 = hand1.landmark[mp_hands.HandLandmark.WRIST]
                wrist2 = hand2.landmark[mp_hands.HandLandmark.WRIST]

                # Determine left and right hand for consistent control
                if wrist1.x < wrist2.x:
                    left_wrist, right_wrist = wrist1, wrist2
                else:
                    left_wrist, right_wrist = wrist2, wrist1

                # 1. Roll Control (Tilting)
                angle = math.atan2(right_wrist.y - left_wrist.y, right_wrist.x - left_wrist.x)
                if angle > TILT_THRESHOLD:
                    current_keys.add('right') # Clockwise tilt
                elif angle < -TILT_THRESHOLD:
                    current_keys.add('left') # Counter-clockwise tilt

                # 2. Speed Control (Distance)
                distance = math.hypot(right_wrist.x - left_wrist.x, right_wrist.y - left_wrist.y)
                current_time = time.time()
                
                # Check for cooldown before issuing another speed command
                if current_time - last_speed_command_time > SPEED_COMMAND_COOLDOWN:
                    if distance < DISTANCE_SPEED_UP:
                        pyautogui.press('pageup') # Speed Up
                        print("SPEED UP!")
                        last_speed_command_time = current_time
                    elif distance > DISTANCE_SLOW_DOWN:
                        pyautogui.press('pagedown') # Slow Down
                        print("SLOW DOWN!")
                        last_speed_command_time = current_time
        else:
            # If we don't see two hands, we are in a neutral state.
            pass # current_keys remains empty

        # --- Manage Key Presses ---
        # Press new keys that are not already pressed
        for key in current_keys:
            if key not in keys_pressed:
                pyautogui.keyDown(key)
                keys_pressed.add(key)
                print(f"Key DOWN: {key}")
        
        # Release old keys that are no longer needed
        keys_to_release = keys_pressed - current_keys
        for key in keys_to_release:
            pyautogui.keyUp(key)
            keys_pressed.remove(key)
            print(f"Key UP: {key}")

        cv2.imshow('New Hand Controls', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
finally:
    # --- Cleanup ---
    print("Releasing all keys and shutting down.")
    for key in keys_pressed:
        pyautogui.keyUp(key)
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()