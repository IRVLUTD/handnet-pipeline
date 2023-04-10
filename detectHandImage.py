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
