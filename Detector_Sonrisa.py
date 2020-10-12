import cv2

# Clasificador de rostro
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
#eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')

# captura de video la camara
webcam = cv2.VideoCapture(0)

# Muestra la toma actual
while True:
    # Lee la toma actual de la camara
    successful_frame_read, frame = webcam.read()

    # Si hay un error que salga del bucle
    if not successful_frame_read:
        break

    # Cambia a una escala de grises
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta la primera cara
    faces = face_detector.detectMultiScale(frame_grayscale)

    # Ejecute la detección de rostros dentro de cada uno de esos rostros
    for (x, y, w, h) in faces:

        # Pinta un rectangulo alrededor del rostro
        cv2.rectangle(frame, (x, y), (x+w, y+w), (100, 200, 50), 4)

        # Obtiene una subtrama (usando una matriz N-dimensional)
        the_face = frame[y:y+h, x:x+w]

        # Cambia a una escala de grises
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)

        #eyes = eye_detector.detectMultiScale(face_grayscale, scaleFactor=1.1, minNeighbors=10)

        # Encuentra la sonrisa en la cara
        #for (x_, y_, w_, h_) in the_face:
            
            # Dibuja un rectangulo alrededor de la sonrisa
            #cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_ + w_), (50, 50, 200), 4)

        # Encuentra los ojos en el rostro
        #for (x_, y_, w_, h_) in eyes:
            
            # Dibuja un rectangulo alrededor de los ojos
            #cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_ + w_), (255, 255, 255), 4)

        # Pinta una etiqueta cuando detecta que esta sonriendo
        if len(smiles) > 0:
            cv2.putText(frame, 'Sonriendo', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

    # Muestra el video actual
    cv2.imshow('Smile Detector', frame)

    # Display
    cv2.waitKey(1)

# Limpiar la pantalla
webcam.release()
cv2.destroyAllWindows()


# El codigo corre sin errores
print("code completed")