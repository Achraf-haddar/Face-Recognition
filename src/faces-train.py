import os
import cv2
import numpy as np
from PIL import Image
import pickle

# Path of Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__ ))
# Path of images directory (join Base_dir + "images")
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
x_train = []
y_labels = []



# Python method walk() generates the file names in a directory
for root, dirs, files in os.walk(image_dir):
    for file in files: # files is a list of file's path
    # check if the file is a png or jpg file
        if file.endswith("png") or file.endswith("jpg"): 
            path = os.path.join(root, file) # get the path of the image
            # get the label from directory name
            label = os.path.basename(root).replace(" ", "-").lower()
            # Transforming label (string) into labels_id
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]

            # Read the image using pillow
            pil_image = Image.open(path).convert("L") # Grayscale
            # Store the image into array
            image_array = np.array(pil_image, 'uint8')
            # Identify faces
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            # Loop through faces and store into x_train and store their labels into y_labels
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

# print(y_labels)
# print(x_train)
with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")