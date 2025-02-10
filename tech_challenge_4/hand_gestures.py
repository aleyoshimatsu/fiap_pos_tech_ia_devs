import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2




class HandGesture:

    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    TEXT_COLOR = (88, 205, 54)

    def __init__(self):
        self.capture = cv2.VideoCapture(0)

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_running_mode = mp.tasks.vision.RunningMode

        self.base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
        self.options = vision.GestureRecognizerOptions(base_options=self.base_options,
                                                       running_mode=self.mp_running_mode.LIVE_STREAM,
                                                       num_hands=2,
                                                       result_callback=self.print_result)
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)
        self.results = None
        self.timestamp = 0
        self.alpha = 0.7
        self.overlay_video = None
        self.overlay_cap = None
        self.overlay_active = False

        self.gestures_videos = {
            "Victory": "videos/fireworks.mov",
            "Thumb_Up": "videos/thumbs_up.mp4",
            "Thumb_Down": "videos/thumbs_down.mov",
            "Closed_Fist": "videos/light-rain.mp4",
            "Pointing_Up": "videos/balloons.mov",
            "ILoveYou": "videos/red-hearts.mov",
        }

    def print_result(self, result, output_image: mp.Image, timestamp_ms: int):
        self.results = result

    def load_overlay(self, overlay_path):
        if self.overlay_active:
            print("Stopping current overlay.")
            self.overlay_cap.release()

        self.overlay_cap = cv2.VideoCapture(overlay_path)
        if not self.overlay_cap.isOpened():
            raise ValueError("Unable to open the overlay video.")

        self.overlay_active = True
        print("Overlay video added successfully.")

    def remove_overlay(self):
        if self.overlay_active:
            self.overlay_cap.release()
            self.overlay_active = False
            print("Overlay removed.")

    def start_camera(self):
        while self.capture.isOpened():
            success, frame = self.capture.read()

            if success:

                if not self.overlay_active:
                    frame_processed, main_gesture = self.analyze_gesture(frame)

                    if main_gesture and main_gesture in self.gestures_videos.keys():
                        self.load_overlay(self.gestures_videos[main_gesture])
                else:
                    frame_processed = frame

                frame_processed = self.add_overlay(frame, frame_processed)

                cv2.imshow('Camera', frame_processed)

                # Press Q to exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break

    def add_overlay(self, frame, frame_processed):
        if self.overlay_active and self.overlay_cap.isOpened():
            ret_overlay, overlay_frame = self.overlay_cap.read()
            if ret_overlay:
                overlay_frame = cv2.resize(overlay_frame, (frame.shape[1], frame.shape[0]))
                frame_processed = cv2.addWeighted(frame_processed, 1 - self.alpha, overlay_frame, self.alpha, 0)
            else:
                print("Overlay video ended.")
                self.remove_overlay()
        return frame_processed

    def analyze_gesture(self, frame, draw_landmarks=True):
        self.timestamp += 1
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        self.recognizer.recognize_async(mp_image, self.timestamp)

        main_gesture = None

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

                if gesture and len(gesture) > 0:
                    main_gesture = gesture[0].category_name

                if draw_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks_proto,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())

                    # height, width, _ = frame.shape
                    # x_coordinates = [landmark.x for landmark in hand]
                    # y_coordinates = [landmark.y for landmark in hand]
                    # text_x = int(min(x_coordinates) * width)
                    # text_y = int(min(y_coordinates) * height) - self.MARGIN
                    #
                    # cv2.putText(frame, f"{handedness[0].category_name} Hand - {gesture[0].category_name}",
                    #             (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    #             self.FONT_SIZE, self.TEXT_COLOR, self.FONT_THICKNESS, cv2.LINE_AA)

        return frame, main_gesture

    def __del__(self):
        self.capture.release()
        if self.overlay_cap:
            self.overlay_cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    hand_gesture = HandGesture()
    hand_gesture.start_camera()