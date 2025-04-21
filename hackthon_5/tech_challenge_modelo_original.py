import cv2
import torch
import yagmail
import tempfile
import os
import time

EMAIL_USER = "gdtavares1@gmail.com"
EMAIL_PASS = ""
EMAIL_TO   = ["gdtavares1@gmail.com", "Matheusa761@gmail.com"]

ALERT_COOLDOWN = 2

def send_alert(frame, label):
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    cv2.imwrite(path, frame)
    yag = yagmail.SMTP(user=EMAIL_USER, password=EMAIL_PASS)
    yag.send(
        to=EMAIL_TO,
        subject="Objeto cortante detectado",
        contents=[label, path]
    )
    print(f"Alerta enviado para {EMAIL_TO}: {label}")
    os.remove(path)

def run_detection(source):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    sharp = ['knife', 'scissors']
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

        df = model(frame).pandas().xyxy[0]
        now = time.time()
        for _, r in df.iterrows():
            if r['name'] in sharp:
                if now - last_alert_time >= ALERT_COOLDOWN:
                    x1, y1 = int(r.xmin), int(r.ymin)
                    x2, y2 = int(r.xmax), int(r.ymax)
                    label = f"{r['name']} {r['confidence']:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
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
