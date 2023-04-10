#Image detection model for hands
import ros_demo
from PIL import Image
import glob

#Colored Images for testing
image_list = []
for filename in glob.glob('../Downloads/imageDataset/0104T143721/*.jpg'): #replace '' with location of testing dataset
    im=Image.open(filename)
    image_list.append(im)

#Depth Images for testing
image_list_depth = []
for filename in glob.glob('../Downloads/imageDataset/0104T143721/*.png'): #replace '' with location of testing dataset
    im=Image.open(filename)
    image_list_depth.append(im)

#calls the run_network() from ros_demo.py
ros_demo.ImageListener.run_network()

# import glob
# import cv2

## create a list of image paths
# image_paths = glob.glob('../Downloads/imageDataset/0104T143721/*.jpg')

## create an instance of the class containing the run_network function
# network_runner = NetworkRunner()

## loop through each image and pass it to the run_network function
# for image_path in image_paths:
#     # read the image
#     im = cv2.imread(image_path)

#    # pass the image to the run_network function
#     network_runner.run_network(im)

