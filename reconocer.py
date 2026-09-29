import os
import cv2

os.environ["DEEPFACE_BACKEND"] = "pytorch"

from deepface import DeepFace

camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("Error: No se pudo acceder a la cámara web.")
    exit()

print("Programa iniciado. Presiona 'q' para salir.")

contador_frames = 0
nombre_persona = "Buscando..."

while True:
    ret, frame = camara.read()
    if not ret:
        break

    contador_frames += 1

    # Procesar cada 8 fotogramas
    if contador_frames % 8 == 0:
        try:
            # Primero verificamos si hay algún rostro presente en la toma
            rostros = DeepFace.extract_faces(
                img_path=frame, 
                detector_backend="ssd", 
                enforce_detection=False
            )
            
            # Si el rostro detectado tiene una confianza muy baja (fondo/vacio), marcamos "Nadie"
            if len(rostros) == 0 or rostros[0]['confidence'] < 0.60:
                nombre_persona = "Sin rostro en cámara"
            else:
                # Si hay un rostro válido, buscamos en la base de datos
                resultados = DeepFace.find(
                    img_path=frame,
                    db_path="imagenes",
                    model_name="Facenet",
                    detector_backend="ssd",
                    enforce_detection=False
                )

                if len(resultados) > 0 and not resultados[0].empty:
                    # Validar si la distancia al mejor resultado es aceptable
                    distancia = resultados[0]['distance'][0]
                    if distancia < 0.40:  # Umbral para evitar falsos positivos
                        ruta_img = resultados[0]['identity'][0]
                        nombre_raw = os.path.basename(ruta_img)
                        nombre_persona = os.path.splitext(nombre_raw)[0]
                    else:
                        nombre_persona = "Desconocido"
                else:
                    nombre_persona = "Desconocido"

        except Exception:
            nombre_persona = "Buscando..."

    # Dibujar la etiqueta en pantalla
    cv2.putText(frame, f"Persona: {nombre_persona}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Reconocimiento Facial", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camara.release()
cv2.destroyAllWindows()