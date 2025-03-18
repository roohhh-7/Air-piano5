import cv2
import mediapipe as mp
import pygame

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize Pygame for sound
pygame.mixer.init()

# Load only available piano sounds
notes_sounds = {
    "C": pygame.mixer.Sound("./sounds/C.wav"),
    "D": pygame.mixer.Sound("./sounds/D.wav"),
    "E": pygame.mixer.Sound("./sounds/E.wav"),
    "F": pygame.mixer.Sound("./sounds/F.wav"),
    "G": pygame.mixer.Sound("./sounds/G.wav"),
}

# Define key positions (Only 5 keys)
keys = [
    (100, 400, "C"),
    (220, 400, "D"),
    (340, 400, "E"),
    (460, 400, "F"),
    (580, 400, "G"),
]

# Function to draw piano keys on screen
def draw_piano(frame, active_note=None):
    for x, y, note in keys:
        color = (255, 255, 255)  # Default white keys
        if note == active_note:
            color = (0, 255, 0)  # Highlight active key in green
        
        cv2.rectangle(frame, (x, y), (x + 100, y + 200), color, -1)  # White key
        cv2.rectangle(frame, (x, y), (x + 100, y + 200), (0, 0, 0), 3)  # Black outline
        cv2.putText(frame, note, (x + 30, y + 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Function to get fingertip position
def get_fingertip_position(hand_landmarks, frame_width, frame_height):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    x, y = int(index_tip.x * frame_width), int(index_tip.y * frame_height)
    return x, y

# Function to detect which key is pressed
def detect_pressed_key(x, y):
    for key_x, key_y, note in keys:
        if key_x < x < key_x + 100 and key_y < y < key_y + 200:
            return note
    return None

# Function to play piano sound
def play_sound(note):
    if note in notes_sounds:
        notes_sounds[note].play()

# Start Video Capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    frame_height, frame_width, _ = frame.shape
    active_note = None
    
    draw_piano(frame)  # Draw virtual piano keys

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get fingertip position
            x, y = get_fingertip_position(hand_landmarks, frame_width, frame_height)
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            # Check if a key is pressed
            note = detect_pressed_key(x, y)
            if note:
                play_sound(note)
                active_note = note
                cv2.putText(frame, f"Playing: {note}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    draw_piano(frame, active_note)  # Update UI with active key
    cv2.imshow("Virtual Piano", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
