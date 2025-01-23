import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

from tech_challenge_4.body_mediapipe import VisionRunningMode


class HandGesture2:

    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    TEXT_COLOR = (88, 205, 54)

    def __init__(self):
        self.capture = cv2.VideoCapture(0)

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
        self.options = vision.GestureRecognizerOptions(base_options=self.base_options,
                                                       running_mode=VisionRunningMode.LIVE_STREAM,
                                                       num_hands=2,
                                                       result_callback=self.print_result)
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)
        self.results = None
        self.timestamp = 0

    def print_result(self, result, output_image: mp.Image, timestamp_ms: int):
        self.results = result

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
        self.timestamp += 1
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        self.recognizer.recognize_async(mp_image, self.timestamp)

        if self.results is not None:
            gestures = self.results.gestures
            hand_landmarks = self.results.hand_landmarks
            handedness_list = self.results.handedness

            for hand, gesture, handedness in zip(hand_landmarks, gestures, handedness_list):
                print(hand)
                print(gesture)
                print(handedness)

                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand
                ])

                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks_proto,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())

                height, width, _ = frame.shape
                x_coordinates = [landmark.x for landmark in hand]
                y_coordinates = [landmark.y for landmark in hand]
                text_x = int(min(x_coordinates) * width)
                text_y = int(min(y_coordinates) * height) - self.MARGIN

                cv2.putText(frame, f"{handedness[0].category_name} Hand - {gesture[0].category_name}",
                            (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                            self.FONT_SIZE, self.TEXT_COLOR, self.FONT_THICKNESS, cv2.LINE_AA)

        return frame

    def __del__(self):
        self.capture.release()
        cv2.destroyAllWindows()
