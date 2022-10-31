from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import math
import time

# Creando la función dibujo

mpDibujo = mp.solutions.drawing_utils
ConfDibu = mpDibujo.DrawingSpec(thickness = 1, circle_radius = 1)

#Creamos el objeto que almacena la malla facial

mpMallaFacial = mp.solutions.face_mesh
MallaFacial = mpMallaFacial.FaceMesh(max_num_faces = 1)

# VideoCaptura
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
# Creamos la app
app = Flask(__name__)

# Mostramos el video en RT
def gen_frame():

    # Variables de conteo

    parpadeo = False
    conteo = 0
    tiempo = 0
    inicio = 0
    final = 0
    conteo_sue = 0
    muestra = 0
    
    while True:
        # Lectura video
        ret, frame = cap.read()

        #Creamos una lista donde almacenamos los resultados
        px = []
        py = []
        lista = []

        if not ret:
            break

        else:

            # Correción del color
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultados = MallaFacial.process(frameRGB)

            if resultados.multi_face_landmarks: #Si se detecta un rostro
                for rostros in resultados.multi_face_landmarks:
                    # Dibujamos
                    mpDibujo.draw_landmarks(frame, rostros, mpMallaFacial.FACEMESH_CONTOURS, ConfDibu, ConfDibu)
                    #Extracción de puntos
                    for id, puntos in enumerate(rostros.landmark):
                        al, an, c = frame.shape
                        x, y = int(puntos.x*an), int(puntos.y*al)
                        px.append(x)
                        py.append(y)
                        lista.append([ id, x, y ])
                        if len(lista) == 468:
                            #Ojo derecho
                            x1, y1 = lista[145][1:]
                            x2, y2 = lista[159][1:]
                            # cx, cy = ( x1 + x2 )//2, ( y1 + y2 )//2
                            #Prueba de distancias
                            # cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), t)
                            # cv2.circle(frame, (x1, y1), r, (0, 0, 0), cv2.FILLED)
                            # cv2.circle(frame, (x2, y2), r, (0, 0, 0), cv2.FILLED)
                            # cv2.circle(frame, (cx, cy), r, (0, 0, 0), cv2.FILLED)
                            longitud1 = math.hypot(x2 - x1, y2 - y1)
                            # print(longitud1)

                            #Ojo izquierdo
                            x3, y3 = lista[374][1:]
                            x4, y4 = lista[386][1:]
                            # cx2, cy2 = (x3 + x4) // 2, (y3 + y4) // 2
                            longitud2 = math.hypot(x4 - x3, y4 - y3)
                            # print(longitud2)

                            #Conteo de parpadeos
                            cv2.putText(frame, f'Parpadeos: {int(conteo)}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                            cv2.putText(frame, f'Somnolencia: {int(conteo_sue)}', (380, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                            cv2.putText(frame, f'Duración: {int(muestra)}', (380, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                            if longitud1 <= 10 and longitud2 <= 10 and parpadeo == False: #Parpadeo
                                conteo = conteo + 1
                                parpadeo = True
                                inicio = time.time()

                            elif longitud2 > 10 and longitud1 > 10 and parpadeo == True: #Seguridad parpadeo
                                parpadeo = False
                                final = time.time()
                            #Temporizador
                            tiempo = round(final - inicio, 0)

                            #Contador de Micro Sueño
                            if tiempo >= 1:
                                conteo_sue = conteo_sue + 1
                                muestra = tiempo
                                inicio = 0
                                final = 0

            # Codificamos el video en bytes
            suc, encode = cv2.imencode( '.jpg', frame )
            frame = encode.tobytes()

        # yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

        yield(  b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )

# App principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta video 
@app.route('/video')
def video():
    return Response( gen_frame(), mimetype='multipart/x-mixed-replace; boundary=frame' )

# Ejecutamos la app
if __name__ == "__main__":
    app.run(debug=True)
