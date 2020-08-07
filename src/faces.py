import numpy as np 
import cv2

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
# load the parameters of trained model
recognizer.read("trainer.yml")

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # cascade works with gray images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # find all faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        print(x,y,w,h)
        # Save the portion of face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = gray[y:y+h, x:x+w]

        # Recognize 
        # the label:id_ and the confidence:conf
        id_, conf = recognizer.predict(roi_gray)
        if conf>=45 and conf<=85:
            print(id_)
        """
        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_color)
        """
        # color of the rectangle
        color = (255, 0, 0) #BGR 
        stroke = 2  # how thick do we want the line to actually be 
        # draw the rectangle
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
