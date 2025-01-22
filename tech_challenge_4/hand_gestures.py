import cv2
import mediapipe as mp

class HandGesture:

    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils

    def start_camera(self):
        while self.capture.isOpened():
            success, frame = self.capture.read()

            if success:
                frame_processed = self.analyze_gesture(frame)
                cv2.imshow('Camera', frame_processed)

                # Press Q to exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break

    def analyze_gesture(self, frame):
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_RGB)
        print(results)

        # TODO implementation

        return frame

    def __del__(self):
        self.capture.release()
        cv2.destroyAllWindows()