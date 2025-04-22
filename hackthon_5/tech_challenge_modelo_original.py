import cv2
import torch
import yagmail
import tempfile
import os
import time
import threading

EMAIL_USER     = "gdtavares1@gmail.com"
EMAIL_PASS     = "dmsu ucke axcy pxtn"
EMAIL_TO       = ["gdtavares1@gmail.com"]
ALERT_COOLDOWN = 10       # segundos entre alertas

MODEL_NAME     = 'yolov5l'   # modelo maior, mais preciso
CONF_THRESHOLD = 0.1          # confiança mínima (10%)
IOU_THRESHOLD  = 0.2          # NMS IoU threshold
IMG_SCALE      = 1280         # largura em px para redimensionar antes da inferência
DETECT_INTERVAL = 1           # inferir em todos os frames
TARGET_NAMES   = ['knife', 'scissors']

class Detector:

    def __init__(self, fonte):
        self.capture = cv2.VideoCapture(fonte)

    def _send_alert_worker(self, frame, label):
        # salva o frame em arquivo temporário
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        cv2.imwrite(path, frame)
        try:
            yag = yagmail.SMTP(user=EMAIL_USER, password=EMAIL_PASS)
            yag.send(
                to=EMAIL_TO,
                subject="🚨 Objeto cortante detectado",
                contents=[label, path]
            )
            print(f"[ALERTA] {label} enviado para {EMAIL_TO}")
        except Exception as e:
            print(f"[ERRO] falha ao enviar e‑mail: {e}")
        finally:
            os.remove(path)

    def send_alert(self, frame, label):
        # captura cópia do frame e dispara thread
        worker_thread = threading.Thread(
            target=self._send_alert_worker,
            args=(frame, label),
            daemon=True
        )
        worker_thread.start()

    def run_detection(self):
        model = torch.hub.load('ultralytics/yolov5', MODEL_NAME, pretrained=True)
        model.conf = CONF_THRESHOLD
        model.iou  = IOU_THRESHOLD

        if not self.capture.isOpened():
            print("Erro: não foi possível abrir a fonte de vídeo.")
            return

        print("Iniciando detecção. Pressione 'q' para sair.")

        fps = self.capture.get(cv2.CAP_PROP_FPS) or 30
        delay = int(1000 / fps)
        last_alert_time = 0
        frame_idx = 0

        while self.capture.isOpened():
            success, frame = self.capture.read()
            frame_idx += 1
            if success:
                self.detect_frame(frame, frame_idx, last_alert_time, model)

                cv2.imshow('Deteccao de Objetos Cortantes', frame)

                # Press Q to exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break

    def detect_frame(self, frame, frame_idx, last_alert_time, model):
        if frame_idx % DETECT_INTERVAL == 0:
            h, w = frame.shape[:2]
            new_h = int(h * IMG_SCALE / w)
            small = cv2.resize(frame, (IMG_SCALE, new_h))

            lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
            small_enhanced = cv2.cvtColor(cv2.merge([cl, a, b]), cv2.COLOR_LAB2BGR)

            results = model(small_enhanced)
            df = results.pandas().xyxy[0]

            print(f"[DEBUG] Frame {frame_idx}: {len(df)} detecções")

            fx = w / small.shape[1]
            fy = h / small.shape[0]

            detections = []
            now = time.time()
            for _, r in df.iterrows():
                if r['name'] in TARGET_NAMES:
                    x1 = int(r.xmin * fx)
                    y1 = int(r.ymin * fy)
                    x2 = int(r.xmax * fx)
                    y2 = int(r.ymax * fy)
                    conf = r['confidence']
                    detections.append((x1, y1, x2, y2, r['name'], conf))
                    # envia alerta para a primeira detectada no frame
                    if now - last_alert_time >= ALERT_COOLDOWN:
                        label = f"{r['name']} {conf:.2f}"
                        self.send_alert(frame.copy(), label)
                        last_alert_time = now
                        break

        for x1, y1, x2, y2, name, conf in locals().get('detections', []):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, f"{name} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )

    def __del__(self):
        self.capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # escolha = input("1 = Webcam, 2 = Vídeo: ").strip()
    # fonte = 0 if escolha == "1" else input("Caminho do vídeo: ").strip()
    fonte = 0
    # fonte = "/home/alexandre_pantalena/desenvolvimento/repos/fiap_pos_tech_ia_devs/tech_challenge_5/video.mp4"
    detector = Detector(fonte=fonte)
    detector.run_detection()
