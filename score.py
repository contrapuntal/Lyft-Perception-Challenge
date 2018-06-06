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

def score(in_file, target_file):

    student_output = in_file
    # score the result
    def decode(packet):
            img = base64.b64decode(packet)
            filename = 'image.png'
            with open(filename, 'wb') as f:
                            f.write(img)
            result = misc.imread(filename)
            return result

    with open(target_file) as json_data:
            ans_data = json.loads(json_data.read())
            json_data.close()

    # Load student data
    with open(student_output) as student_data:
            student_ans_data = json.loads(student_data.read())
            student_data.close()

    frames_processed = 0

    Car_TP = 0 # True Positives
    Car_FP = 0 # Flase Positives
    Car_TN = 0 # True Negatives
    Car_FN = 0 # True Negatives

    Road_TP = 0 # True Positives
    Road_FP = 0 # Flase Positives
    Road_TN = 0 # True Negatives
    Road_FN = 0 # True Negatives

    for frame in range(1,len(ans_data.keys())+1):
            truth_data_car =  decode(ans_data[str(frame)][0])
            truth_data_road =  decode(ans_data[str(frame)][1])
            student_data_car = decode(student_ans_data[str(frame)][0])
            student_data_road = decode(student_ans_data[str(frame)][1])
            
            # if(frames_processed == 2):
            #    plt.imshow(truth_data_car)

            Car_TP += np.sum(np.logical_and(student_data_car == 1, truth_data_car == 1))
            Car_FP += np.sum(np.logical_and(student_data_car == 1, truth_data_car == 0))
            Car_TN += np.sum(np.logical_and(student_data_car == 0, truth_data_car == 0))
            Car_FN += np.sum(np.logical_and(student_data_car == 0, truth_data_car == 1))

            Road_TP += np.sum(np.logical_and(student_data_road == 1, truth_data_road == 1))
            Road_FP += np.sum(np.logical_and(student_data_road == 1, truth_data_road == 0))
            Road_TN += np.sum(np.logical_and(student_data_road == 0, truth_data_road == 0))
            Road_FN += np.sum(np.logical_and(student_data_road == 0, truth_data_road == 1))

            frames_processed+=1
    # Generate results
    Car_precision = Car_TP/(Car_TP+Car_FP)/1.0
    Car_recall = Car_TP/(Car_TP+Car_FN)/1.0
    Car_beta = 2
    Car_F = (1+Car_beta**2) * ((Car_precision*Car_recall)/(Car_beta**2 * Car_precision + Car_recall))
    Road_precision = Road_TP/(Road_TP+Road_FP)/1.0
    Road_recall = Road_TP/(Road_TP+Road_FN)/1.0
    Road_beta = 0.5
    Road_F = (1+Road_beta**2) * ((Road_precision*Road_recall)/(Road_beta**2 * Road_precision + Road_recall))

    print ("Car F score: %05.3f  | Car Precision: %05.3f  | Car Recall: %05.3f  |\n\
    Road F score: %05.3f | Road Precision: %05.3f | Road Recall: %05.3f | \n\
    Averaged F score: %05.3f" %(Car_F,Car_precision,Car_recall,Road_F,Road_precision,Road_recall,((Car_F+Road_F)/2.0)))


def main(args):
    if(args.input != None):
        score(args.input, args.answer_file)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--answer_file', default='Example/results.json', help='Correct json file (answer key)')
    parser.add_argument('--input', default='output.json', help='input json file')

    main(parser.parse_args())
