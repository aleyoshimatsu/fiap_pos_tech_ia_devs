import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2




class FaceEmotionDetection:

    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    TEXT_COLOR = (88, 205, 54)

    def __init__(self):
        self.capture = cv2.VideoCapture(0)

        self.mp_face = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_running_mode = mp.tasks.vision.RunningMode

        self.base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
        self.options = vision.FaceLandmarkerOptions(base_options=self.base_options,
                                                    running_mode=self.mp_running_mode.LIVE_STREAM,
                                                    output_face_blendshapes=True,
                                                    output_facial_transformation_matrixes=True,
                                                    num_faces=1,
                                                    result_callback=self.print_result)
        self.recognizer = vision.FaceLandmarker.create_from_options(self.options)
        self.results = None
        self.timestamp = 0
        self.alpha = 0.7
        self.overlay_video = None
        self.overlay_cap = None
        self.overlay_active = False

        self.emotions_videos = {
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

                frame_processed, main_emotion = self.analyze_face(frame)

                if not self.overlay_active:
                    if main_emotion and main_emotion in self.emotions_videos.keys():
                        self.load_overlay(self.emotions_videos[main_emotion])
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

    def analyze_face(self, frame, draw_landmarks=True):
        self.timestamp += 1
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        self.recognizer.detect_async(mp_image, self.timestamp)

        main_emotion = None

        if self.results is not None:
            face_landmarks_list = self.results.face_landmarks

            for idx in range(len(face_landmarks_list)):
                face_landmarks = face_landmarks_list[idx]
                print(face_landmarks)

                face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                face_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in
                    face_landmarks
                ])

                if draw_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks_proto,
                        connections=self.mp_face.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks_proto,
                        connections=self.mp_face.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks_proto,
                        connections=self.mp_face.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style())

        return frame, main_emotion

    def __del__(self):
        self.capture.release()
        if self.overlay_cap:
            self.overlay_cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    face_emotion_detection = FaceEmotionDetection()
    face_emotion_detection.start_camera()