#!/usr/bin/env python3
import cv2
import os
import glob
from ultralytics import YOLO

def find_best_weights(runs_dir="runs"):
    """
    Procura recursivamente pelo arquivo best.pt na pasta runs/.
    Retorna o caminho do best.pt mais recente.
    """
    pattern = os.path.join(runs_dir, "**", "weights", "best.pt")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        raise FileNotFoundError(f"Nenhum best.pt encontrado em {runs_dir}")
    # escolhe o mais recentemente modificado
    return max(candidates, key=os.path.getmtime)

def run_detection(video_source, weights_path):
    # Carrega seu modelo customizado
    model = YOLO(weights_path)

    # Somente essas classes
    sharp_names = ['Faca', 'Tesoura']
    sharp_ids   = [i for i,n in model.names.items() if n in sharp_names]

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a fonte de vídeo.")
        return

    print("Iniciando a detecção. Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fim da fonte de vídeo ou falha na captura do frame.")
            break

        # Inferência
        results = model.predict(source=frame, conf=0.25, verbose=False)
        r = results[0]

        # Desenha apenas classes de interesse
        for box in r.boxes:
            cls_id = int(box.cls.cpu().numpy()[0])
            if cls_id not in sharp_ids:
                continue

            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)
            conf = float(box.conf.cpu().numpy()[0])
            name = model.names[cls_id]
            label = f"{name} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(
                frame, label, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2
            )

        cv2.imshow("Detecção de Objetos", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 1) Encontra o best.pt mais recente
    try:
        weights = find_best_weights(runs_dir="runs")
        print("✔️ Usando pesos:", weights)
    except Exception as e:
        print("❌", e)
        exit()

    # 2) Escolha da fonte
    print("1 - Webcam")
    print("2 - Vídeo")
    escolha = input("Digite 1 ou 2: ").strip()
    if escolha == "1":
        fonte = 0
    else:
        fonte = input("Caminho do vídeo: ").strip()

    # 3) Roda detecção
    run_detection(fonte, weights)
