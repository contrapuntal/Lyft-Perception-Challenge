import sys
import json
import base64
import torch
import torch.nn.functional as F
from io import BytesIO, StringIO
from scipy import misc
from argparse import ArgumentParser
from torchvision.transforms import ToTensor
import numpy as np
from erfnet import Net
import cv2

use_crop = True

def eval(file, net, device):
    def encode(array):
        retval, buffer = cv2.imencode('.png', array)
        return base64.b64encode(buffer).decode("utf-8")

    
    # Video Processing
    video = cv2.VideoCapture(file)
    frame = 1
    

    net.eval()
    shape = (video.get(cv2.CAP_PROP_FRAME_HEIGHT), video.get(cv2.CAP_PROP_FRAME_WIDTH))
    ones = torch.ones(shape).to(device)
    zeros = torch.zeros(shape).to(device)
    answer_key = {}

    while(True):
        (grabbed, bgr_frame) = video.read()
        
        if not grabbed:
            break
 
        img = ToTensor()(bgr_frame)
        img = img.to(device)
        img = img[[2, 1, 0], :, :] # swap channel from BGR to RGB
        img = img.unsqueeze(0)
        if(use_crop):
            img = img[:,:,100:540,:]

        result = net(img)
        
        result = result.max(1)[1]
        if(use_crop):
          result = F.pad(result, (0,0,100,60,0,0))

        final = result[0]

        binary_car_result = torch.where(final==2,ones,zeros).cpu().detach().numpy() 
        binary_road_result = torch.where(final==1,ones,zeros).cpu().detach().numpy() 

        answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
        frame += 1

    # Print output in proper json format
    print(json.dumps(answer_key))
    return frame

def main(args):
    net = Net(num_classes=3)
    net = net.to(args.device)
    net.load_state_dict(torch.load(args.model))
    frames_processed = eval(args.video, net, args.device)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--device', action='store', default='cuda', help='choices={cpu, cuda, cuda:0, cuda:1, ...}')
    parser.add_argument('video', type=str, metavar='input_video', help='Input video')
    parser.add_argument('--model', default='model.pth', help='model file to load')

    main(parser.parse_args())
