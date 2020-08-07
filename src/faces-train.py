import os
# Path of Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__ ))
# Path of images directory (join Base_dir + "images")
image_dir = os.path.join(BASE_DIR, "images")
# Python method walk() generates the file names in a directory
for root, dirs, files in os.walk(image_dir):
    for file in files: # files is a list of file's path
    # check if the file is a png or jpg file
        if file.endswith("png") or file.endswith("jpg"): 
            path = os.path.join(root, file) # get the path of the image
            print(path)