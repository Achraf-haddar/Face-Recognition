import os
import numpy as np
from PIL import Image
# Path of Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__ ))
# Path of images directory (join Base_dir + "images")
image_dir = os.path.join(BASE_DIR, "images")

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
            print(label)
            # Read the image using pillow
            pil_image = Image.open(path).convert("L") # Grayscale
            # Store the image into array
            image_array = np.array(pil_image, 'uint8')
            print(image_array)