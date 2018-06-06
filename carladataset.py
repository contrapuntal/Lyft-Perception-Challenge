import numpy as np
import os

from PIL import Image, ImageOps
import torch
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt
from torchvision.transforms.functional import crop
import torch.utils.data as data
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
THREE_CLASSES = True

def is_image(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class carla(data.Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        super(carla, self).__init__()
        self.images_root = os.path.join(root, 'CameraRGB')
        self.target_root = os.path.join(root, 'CameraSeg')

        self.filenames = [os.path.basename(os.path.splitext(f)[0])
            for f in os.listdir(self.target_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def preprocess_labels(self, label_image):
        # Identify lane marking pixels (label is 6)
        labels_new = np.array(label_image)
        lane_marking_pixels = (label_image[:,:,0] == 6).nonzero()

        # Identify all vehicle pixels
        vehicle_pixels = (label_image[:,:,0] == 10).nonzero()
        # Isolate vehicle pixels associated with the hood (y-position > 496)
        hood_indices = (vehicle_pixels[0] >= 496).nonzero()[0]
        hood_pixels = (vehicle_pixels[0][hood_indices], \
                       vehicle_pixels[1][hood_indices])
        #hood_pixels = hood_indices
        #print(label_image.shape)
        #print(len(hood_pixels[1]))
        # Set hood pixel labels to 0
        labels_new[hood_pixels] = 0
        if(THREE_CLASSES):
            # For all pixels that don't belong to a car and road, re-label as background

            exclude = [6, 7, 10]
            for i in range(13):
                if(i not in exclude):
                    pixels = (label_image[:,:,0] == i).nonzero()
                    labels_new[pixels] = 0
            
            # recode road and vehicles class labels to 1 and 2
            road_pixels = (label_image[:,:,0] == 7).nonzero()
            labels_new[road_pixels] = 1

            #vehicle_pixels = (label_image[:,:,0] == 10).nonzero()
            vehicle_pixels = (labels_new == 10).nonzero()
            labels_new[vehicle_pixels] = 2
            # Set lane marking pixels to road (label is 7)
            labels_new[lane_marking_pixels] = 1 #recode to 1
        else:
            #recode traffic signs(12) as roadlines(6)
            traffic_sign_pixels = (labels_new[:,:,0] == 12).nonzero()
            labels_new[traffic_sign_pixels] = 6 #recode to 1
        
            # Set lane marking pixels to road (label is 7)
            labels_new[lane_marking_pixels] = 7 
        
        # Return the preprocessed label image - 2d red channel only
        return labels_new
        
    def __getitem__(self, index):
        filename = self.filenames[index]
        seed = random.randint(0,2**32) #use same seed for random transformation for both input and target

        with open(os.path.join(self.images_root, f'{filename}.png'), 'rb') as f:
            image = Image.open(f).convert('RGB')

        with open(os.path.join(self.target_root, f'{filename}.png'), 'rb') as f:
            label = Image.open(f).convert('RGB')

        #Random translation 0-2 pixels (fill rest with padding
        #transX = random.randint(-2, 2) 
        #transY = random.randint(-2, 2)
        #image = ImageOps.expand(image, border=(transX,transY,0,0), fill=0)
        #label = ImageOps.expand(label, border=(transX,transY,0,0), fill=0) 
        #image = image.crop((0, 0, image.size[0]-transX, image.size[1]-transY))
        #label = label.crop((0, 0, label.size[0]-transX, label.size[1]-transY)) 

        label = np.array(label)[:,:,0][:, :, None]
        label = self.preprocess_labels(label)





        label = ToPILImage()(label)
        
        if self.input_transform is not None:
            random.seed(seed)
            image = self.input_transform(image) #[:,194:498,:] #[:,204:524,:]
        if self.target_transform is not None:
            random.seed(seed)
            label = self.target_transform(label) #[:,194:498,:] #[:,204:524,:]
            
        label = np.array(label).astype(int)[:,:,None]
        label = ToTensor()(label)

        return image, label, filename

    def __len__(self):
        return len(self.filenames)
