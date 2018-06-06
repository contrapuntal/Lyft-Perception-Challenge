import sys
import skvideo.io
import json
import base64
import torch
from io import BytesIO, StringIO
from scipy import misc
from argparse import ArgumentParser
from torchvision.transforms import ToTensor
import numpy as np
from torch.autograd import Variable
# from PIL import Image
import timeit
import cv2


def eval(file, net, device, outfile):
    # process the video
    # Define encoder function
    # def encode(array):
    #        pil_img = Image.fromarray(array)
    #        buff = BytesIO()
    # pil_img.save(buff, format="PNG")
    #        return base64.b64encode(buff.getvalue()).decode("utf-8")
    def encode(array):
        retval, buffer = cv2.imencode('.png', array)
        return base64.b64encode(buffer).decode("utf-8")

    # Video Processing
    video = skvideo.io.vread(file)
  
    answer_key = {}

    # Frame numbering starts at 1
    frame = 1

    net.eval()
    shape = video.shape[1:3]
    ones = torch.ones(shape).to(device)
    zeros = torch.zeros(shape).to(device)
    
    for rgb_frame in video:
 
        img = ToTensor()(rgb_frame)
        img = img.to(device)
        img = img.unsqueeze(0)

        result = net(img)

        result = result.max(1)[1]
        #result = result.cpu().detach().numpy()[0]
        final = result[0]

        binary_car_result = torch.where(final==2,ones,zeros).cpu().detach().numpy() 
        binary_road_result = torch.where(final==1,ones,zeros).cpu().detach().numpy() 

        answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]

        # Increment frame
        frame += 1

    # Print output in proper json format
    if(outfile == None): #output to screen
        print(json.dumps(answer_key))
    else: #output to file
        with open(outfile, 'w') as outfile:
            json.dump(answer_key, outfile)
    return frame

def main(args):
    start = timeit.default_timer()

    from erfnet import Net

    net = Net(num_classes=3)

    net = net.to(args.device)

    net.load_state_dict(torch.load(args.model))
    stop = timeit.default_timer()

    frames_processed = eval(args.video, net, args.device, args.output)
    #print("FPS: ", frames_processed/ (stop-start))

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--device', action='store', default='cuda', help='choices={cpu, cuda, cuda:0, cuda:1, ...}')
    #parser.add_argument('--video', default='Example/test_video.mp4', help='Input video')
    parser.add_argument('video', type=str, metavar='input_video', help='Input video')
    parser.add_argument('--answer_file', default='Example/results.json', help='Correct json file (answer key)')
    parser.add_argument('--model', default='model.pth', help='model file to load')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--output', help='file to save the output json')

    main(parser.parse_args())
