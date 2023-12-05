import mediapipe as mp
import cv2
import os
import uuid

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Create the 'Output Images' directory if it doesn't exist
output_dir = 'Output Images'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL)  # Create the OpenCV window
    while cap.isOpened():
        ret, frame = cap.read()

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip on horizontal
        image = cv2.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        results = hands.process(image)

        # Set flag to true
        image.flags.writeable = True

        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Detections
        print(results)

        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                          )

        # Save the image with a unique name
        image_path = os.path.join(output_dir, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(image_path, image)

        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Get the current working directory
current_directory = os.getcwd()

# Print the current directory
print("Current working directory:", current_directory)

