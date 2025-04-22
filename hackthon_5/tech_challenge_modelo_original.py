import cv2
import torch
import yagmail
import tempfile
import os
import time
import threading

EMAIL_USER     = "gdtavares1@gmail.com"
EMAIL_PASS     = "dmsu ucke axcy pxtn"
EMAIL_TO       = ["gdtavares1@gmail.com", "alexandre.yoshimatsu@virgo.inc"]
ALERT_COOLDOWN = 10
MODEL_NAME     = 'yolov5m'
CONF_THRESHOLD = 0.5
IOU_THRESHOLD  = 0.3
IMG_SIZE       = 960
TARGET_NAMES   = ['knife', 'scissors']

def _send_alert_worker(frame, label):
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

def send_alert(frame, label):
    # captura cópia do frame e dispara thread
    worker_thread = threading.Thread(
        target=_send_alert_worker,
        args=(frame, label),
        daemon=True
    )
    worker_thread.start()

def run_detection(source):
    model = torch.hub.load('ultralytics/yolov5', MODEL_NAME, pretrained=True)
    model.conf = CONF_THRESHOLD
    model.iou  = IOU_THRESHOLD

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Erro: não foi possível abrir a fonte de vídeo.")
        return

    last_alert_time = 0
    print("Iniciando detecção. Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fim da fonte de vídeo.")
            break

        results = model(frame, size=IMG_SIZE)
        df = results.pandas().xyxy[0]

        now = time.time()
        for _, r in df.iterrows():
            name = r['name']
            if name in TARGET_NAMES and now - last_alert_time >= ALERT_COOLDOWN:
                x1, y1 = int(r.xmin), int(r.ymin)
                x2, y2 = int(r.xmax), int(r.ymax)
                label  = f"{name} {r['confidence']:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                # dispara envio async
                send_alert(frame.copy(), label)
                last_alert_time = now
                break

        cv2.imshow("Detecção de Objetos", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    escolha = input("1 = Webcam, 2 = Vídeo: ").strip()
    fonte = 0 if escolha == "1" else input("Caminho do vídeo: ").strip()
    run_detection(fonte)
