import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os

def calculate_angle(point1, point2, point3):
    """Calculate angle between three points"""
    x1, y1 = point1.x, point1.y
    x2, y2 = point2.x, point2.y
    x3, y3 = point3.x, point3.y
    
    angle = np.degrees(np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def detect_gesture(hand_landmarks):
    """Detect ASL gesture using hand landmarks"""
    # Get finger states
    fingers = []
    # Thumb
    thumb_angle = calculate_angle(hand_landmarks.landmark[4], hand_landmarks.landmark[3], hand_landmarks.landmark[2])
    fingers.append(thumb_angle > 150)
    
    # Other fingers
    for tip, pip, dip in zip([8, 12, 16, 20], [6, 10, 14, 18], [5, 9, 13, 17]):
        finger_angle = calculate_angle(hand_landmarks.landmark[tip], hand_landmarks.landmark[pip], hand_landmarks.landmark[dip])
        fingers.append(finger_angle > 150)
    
    # Calculate distances between fingertips
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    
    thumb_index_dist = calculate_distance(thumb_tip, index_tip)
    thumb_middle_dist = calculate_distance(thumb_tip, middle_tip)
    thumb_ring_dist = calculate_distance(thumb_tip, ring_tip)
    thumb_pinky_dist = calculate_distance(thumb_tip, pinky_tip)
    index_middle_dist = calculate_distance(index_tip, middle_tip)
    
    # Adjust threshold based on hand size
    hand_size = calculate_distance(hand_landmarks.landmark[0], hand_landmarks.landmark[5])
    threshold = hand_size * 0.2  # 20% of hand size
    
    # Detect numbers
    if all(not f for f in fingers):  # All fingers closed
        return '0'
    elif fingers[1] and not any(fingers[2:]):  # Only index finger
        return '1'
    elif fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:  # Index and middle
        return '2'
    elif fingers[1] and fingers[2] and fingers[3] and not fingers[4]:  # First three fingers
        return '3'
    elif all(fingers[1:]) and not fingers[0]:  # All fingers except thumb
        return '4'
    elif all(fingers):  # All fingers extended
        return '5'
    elif fingers[0] and fingers[4] and not any(fingers[1:4]):  # Thumb and pinky
        return '6'
    elif fingers[0] and fingers[1] and not any(fingers[2:]):  # Thumb and index
        return '7'
    elif fingers[0] and fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:  # Thumb, index, middle
        return '8'
    elif fingers[0] and fingers[1] and fingers[2] and fingers[3] and not fingers[4]:  # Thumb and first three fingers
        return '9'
    
    # Detect letters
    elif fingers[0] and not any(fingers[1:]):  # Only thumb
        return 'A'
    elif all(fingers[1:]) and not fingers[0]:  # All fingers except thumb
        return 'B'
    elif fingers[0] and fingers[1] and not any(fingers[2:]):  # Thumb and index
        if thumb_index_dist < threshold:  # Thumb touches index
            return 'F'
        return 'L'
    elif fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:  # Index and middle
        if thumb_index_dist < threshold:  # Thumb touches index
            return 'T'
        if index_middle_dist < threshold:  # Fingers together
            return 'U'
        return 'V'
    elif fingers[1] and fingers[2] and fingers[3] and not fingers[4]:  # First three fingers
        return 'W'
    elif fingers[0] and fingers[4] and not any(fingers[1:4]):  # Thumb and pinky
        return 'Y'
    elif fingers[0] and fingers[1] and fingers[2]:  # Thumb, index, and middle
        if thumb_index_dist < threshold:  # Thumb touches index
            return 'K'
        return 'R'
    elif fingers[0] and fingers[1] and fingers[2] and fingers[3]:  # All fingers except pinky
        if thumb_index_dist < threshold:  # Thumb touches index
            return 'O'
        return 'C'
    elif fingers[4] and not any(fingers[:4]):  # Only pinky
        return 'I'
    elif fingers[1] and fingers[2] and fingers[4] and not fingers[3]:  # Index, middle, and pinky
        return 'H'
    elif fingers[0] and fingers[1] and fingers[4] and not fingers[2] and not fingers[3]:  # Thumb, index, and pinky
        return 'G'
    elif fingers[0] and fingers[2] and fingers[3] and not fingers[1] and not fingers[4]:  # Thumb, middle, and ring
        return 'M'
    elif fingers[0] and fingers[2] and not fingers[1] and not fingers[3] and not fingers[4]:  # Thumb and middle
        return 'N'
    elif fingers[0] and fingers[3] and not fingers[1] and not fingers[2] and not fingers[4]:  # Thumb and ring
        return 'P'
    elif fingers[0] and fingers[1] and fingers[4] and not fingers[2] and not fingers[3]:  # Thumb, index, and pinky
        return 'Q'
    elif fingers[0] and fingers[1] and fingers[2] and fingers[4] and not fingers[3]:  # All fingers except ring
        return 'U'
    elif fingers[0] and fingers[1] and fingers[3] and fingers[4] and not fingers[2]:  # All fingers except middle
        return 'X'
    elif fingers[0] and fingers[2] and fingers[3] and fingers[4] and not fingers[1]:  # All fingers except index
        return 'Z'
    
    return '?'

def main():
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    # Dictionary mapping letters and numbers to their gestures
    gesture_descriptions = {
        'A': 'Make a fist with thumb alongside fingers',
        'B': 'Extend all fingers straight up',
        'C': 'Form a C shape with thumb and fingers',
        'D': 'Point index finger up, others closed',
        'E': 'All fingers closed, thumb across fingers',
        'F': 'Index and thumb touching, others extended',
        'G': 'Point index finger to the side',
        'H': 'Index and middle fingers extended side by side',
        'I': 'Pinky finger extended, others closed',
        'J': 'Make a J shape with pinky finger',
        'K': 'Index and middle fingers extended, thumb up',
        'L': 'Index finger and thumb extended, others closed',
        'M': 'All fingers closed, thumb under fingers',
        'N': 'Index and middle fingers closed, others extended',
        'O': 'Form an O shape with thumb and fingers',
        'P': 'Index finger pointing down, thumb extended',
        'Q': 'Index finger pointing down, thumb to side',
        'R': 'Cross index and middle fingers',
        'S': 'Make a fist with thumb across fingers',
        'T': 'Thumb between index and middle fingers',
        'U': 'Index and middle fingers extended up',
        'V': 'Index and middle fingers extended apart',
        'W': 'Index, middle, and ring fingers extended',
        'X': 'Bend index finger',
        'Y': 'Thumb and pinky extended, others closed',
        'Z': 'Make a Z shape with index finger',
        '0': 'Make a fist',
        '1': 'Index finger extended',
        '2': 'Index and middle fingers extended',
        '3': 'Index, middle, and ring fingers extended',
        '4': 'All fingers extended except thumb',
        '5': 'All fingers extended',
        '6': 'Thumb and pinky touching, others extended',
        '7': 'Thumb and index touching, others extended',
        '8': 'Thumb and middle touching, others extended',
        '9': 'Thumb and ring touching, others extended'
    }

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Create window with specific size
    cv2.namedWindow('Sign Language Recognition', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Sign Language Recognition', 1280, 720)

    print("\nSign Language Recognition System")
    print("-------------------------------")
    print("Instructions:")
    print("1. Ensure good lighting")
    print("2. Keep your hand in frame")
    print("3. Make clear gestures")
    print("4. Press 'q' to quit")
    print("\nSupported gestures:")
    print("- Letters: A-Z")
    print("- Numbers: 0-9")

    # Previous predictions for smoothing
    prev_predictions = []
    max_predictions = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
                
                # Get hand position for visual feedback
                h, w = frame.shape[:2]
                center_x = int(np.mean([lm.x for lm in hand_landmarks.landmark]) * w)
                center_y = int(np.mean([lm.y for lm in hand_landmarks.landmark]) * h)
                
                # Draw target zone
                zone_color = (0, 255, 0) if 0.3 < center_x/w < 0.7 and 0.3 < center_y/h < 0.7 else (0, 0, 255)
                cv2.rectangle(frame, 
                            (int(w*0.3), int(h*0.3)), 
                            (int(w*0.7), int(h*0.7)), 
                            zone_color, 2)
                
                # Detect gesture
                gesture = detect_gesture(hand_landmarks)
                
                # Add to previous predictions
                prev_predictions.append(gesture)
                if len(prev_predictions) > max_predictions:
                    prev_predictions.pop(0)
                
                # Get most common prediction
                if prev_predictions:
                    gesture = max(set(prev_predictions), key=prev_predictions.count)
                    gesture_desc = gesture_descriptions.get(gesture, 'Unknown gesture')
                    
                    # Display the prediction
                    cv2.putText(frame, f"Sign: {gesture}", (10, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                    cv2.putText(frame, f"Gesture: {gesture_desc}", (10, 100),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Show hand in frame", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            prev_predictions = []

        # Display the frame
        cv2.imshow('Sign Language Recognition', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 