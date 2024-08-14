import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from gazenet import GazeNet

import time
import os
import numpy as np
import json
import cv2
from PIL import Image, ImageOps
import random
from tqdm import tqdm
import operator
import itertools
from scipy.io import  loadmat
import logging

from scipy import signal

from utils import data_transforms
from utils import get_paste_kernel, kernel_map

from eye_position_norm import normalize_coordinates
from bounding import bound
import json

def generate_data_field(eye_point):
    """eye_point is (x, y) and between 0 and 1"""
    height, width = 224, 224
    x_grid = np.array(range(width)).reshape([1, width]).repeat(height, axis=0)
    y_grid = np.array(range(height)).reshape([height, 1]).repeat(width, axis=1)
    grid = np.stack((x_grid, y_grid)).astype(np.float32)

    x, y = eye_point
    x, y = x * width, y * height

    grid -= np.array([x, y]).reshape([2, 1, 1]).astype(np.float32)
    norm = np.sqrt(np.sum(grid ** 2, axis=0)).reshape([1, height, width])
    # avoid zero norm
    norm = np.maximum(norm, 0.1)
    grid /= norm
    return grid

def preprocess_image(image_path, eye):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # crop face
    x_c, y_c = eye
    x_0 = x_c - 0.15
    y_0 = y_c - 0.15
    x_1 = x_c + 0.15
    y_1 = y_c + 0.15
    if x_0 < 0:
        x_0 = 0
    if y_0 < 0:
        y_0 = 0
    if x_1 > 1:
        x_1 = 1
    if y_1 > 1:
        y_1 = 1

    h, w = image.shape[:2]
    face_image = image[int(y_0 * h):int(y_1 * h), int(x_0 * w):int(x_1 * w), :]
    # process face_image for face net
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = Image.fromarray(face_image)
    face_image = data_transforms['test'](face_image)
    # process image for saliency net
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = data_transforms['test'](image)

    # generate gaze field
    gaze_field = generate_data_field(eye_point=eye)
    sample = {'image' : image,
              'face_image': face_image,
              'eye_position': torch.FloatTensor(eye),
              'gaze_field': torch.from_numpy(gaze_field)}

    return sample


def test(net, test_image_path, eye):
    net.eval()
    heatmaps = []

    data = preprocess_image(test_image_path, eye)

    image, face_image, gaze_field, eye_position = data['image'], data['face_image'], data['gaze_field'], data['eye_position']
    image, face_image, gaze_field, eye_position = map(lambda x: Variable(x.unsqueeze(0).cuda(), volatile=True), [image, face_image, gaze_field, eye_position])

    _, predict_heatmap = net([image, face_image, gaze_field, eye_position])

    final_output = predict_heatmap.cpu().data.numpy()

    heatmap = final_output.reshape([224 // 4, 224 // 4])

    h_index, w_index = np.unravel_index(heatmap.argmax(), heatmap.shape)
    f_point = np.array([w_index / 56., h_index / 56.])


    return heatmap, f_point[0], f_point[1] 

def draw_result(image_path, eye, heatmap, gaze_point,k,i,bounds,paper):
    x1, y1 = eye
    x2, y2 = gaze_point
    im = cv2.imread(image_path)
    image_height, image_width = im.shape[:2]
    x1, y1 = image_width * x1, y1 * image_height
    x2, y2 = image_width * x2, y2 * image_height
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    cv2.circle(im, (x1, y1), 5, [255, 255, 255], -1)
    cv2.circle(im, (x2, y2), 5, [0, 0, 255], -1)
    cv2.line(im, (x1, y1), (x2, y2), [255, 0, 0], 3)

    for box in bounds:
        min_x = int(image_width * box['min_x'])
        max_x = int(image_width * box['max_x'])
        min_y = int(image_height * box['min_y'])
        max_y = int(image_height * box['max_y'])
        # Draw rectangle on image
        cv2.rectangle(im, (min_x, min_y), (max_x, max_y), [0, 255, 0], 2)

    min_x = int(image_width * paper['min_x'])
    max_x = int(image_width * paper['max_x'])
    min_y = int(image_height * paper['min_y'])
    max_y = int(image_height * paper['max_y'])
    # Draw rectangle on image
    cv2.rectangle(im, (min_x, min_y), (max_x, max_y), [0, 255, 255], 2)

    # heatmap visualization
    heatmap = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255).astype(np.uint8)
    heatmap = np.stack([heatmap, heatmap, heatmap], axis=2)
    heatmap = cv2.resize(heatmap, (image_width, image_height))

    heatmap = (0.8 * heatmap.astype(np.float32) + 0.2 * im.astype(np.float32)).astype(np.uint8)
    img = np.concatenate((im, heatmap), axis=1)
    cv2.imwrite(f'D:/group2_results/tmp_{k}_{i}.png', img)
    
    return img
    

def paper_coor(frame_width, frame_height):
    min_x = 540 / frame_width
    max_x = 1470 /frame_width
    min_y = 755 / frame_height
    max_y = 1060 /frame_height
    paper={
        'min_y': min_y,
        'max_y': max_y,
        'min_x': min_x,
        'max_x': max_x
    }
    return paper

def main():

    net = GazeNet()
    net = DataParallel(net)
    net.cuda()

    pretrained_dict = torch.load('../model/pretrained_model.pkl')
    frame_num = 61894 #number of frame
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    f= open("./log/group2_1_log.txt","w+")
    
    cnt=0
    personal_cnt=[0,0,0,0,0,0,0,0,0]
    flag=[0,0,0,0,0,0,0,0,0]
    personal_rcnt=[0,0,0,0,0,0,0,0,0]

    paper_cnt=0
    paper_personal_cnt=[0,0,0,0,0,0,0,0,0]
    paper_flag=[0,0,0,0,0,0,0,0,0]
    paper_personal_rcnt=[0,0,0,0,0,0,0,0,0]

    paper = paper_coor(1920,1080)
    for k in range(frame_num):
        test_image_path = f'D:/group2_imgs/frame_{k:05d}.png'
        input_path = f'../../../openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended/openpose/output/group2_{k:012d}_keypoints.json'
        print(input_path)
        coor = normalize_coordinates(input_path,1920,1080)
        f.write(f'<frame {k}>\n')
        is_gazing_1 = 0
        paper_gazing1 = 0
        for i,people in enumerate(coor):
            is_gazing=0
            paper_gazing=0
            x = people['norm_eye_x']
            y = people['norm_eye_y']
            heatmap, p_x, p_y = test(net, test_image_path, (x, y))
            f.write(f'person {i}\n{p_x} {p_y}\n')
            bounds = bound(input_path,1920,1080)
            draw_result(test_image_path, (x, y), heatmap, (p_x, p_y),k,i,bounds,paper)
            for j,peo in enumerate(bounds):
                if i!=j:
                    min_x = peo['min_x']
                    max_x = peo['max_x']
                    min_y = peo['min_y']
                    max_y = peo['max_y']
                    #print(min_x,max_x,min_y,max_y)
                    if p_x>=min_x and p_x<=max_x and p_y>=min_y and p_y<=max_y:
                        is_gazing=1
                        is_gazing_1=1
                    f.write(f"person {j}'s bounding box\n{min_x} {max_x} {min_y} {max_y}\n")
                    if p_x>=paper['min_x'] and p_x<=paper['max_x'] and p_y>=paper['min_y'] and p_y<=paper['max_y']:
                        paper_gazing1=1
                        paper_gazing=1
                        
            personal_cnt[i]+=is_gazing
            paper_personal_cnt[i]+=paper_gazing
            if is_gazing==1:
                if flag[i]==0:
                    personal_rcnt[i]+=1
                    flag[i]=1
            else:
                flag[i]=0

            if paper_gazing==1:
                if paper_flag[i]==0:
                    paper_personal_rcnt[i]+=1
                    paper_flag[i]=1
            else:
                paper_flag[i]=0
            f.write('\n')

        cnt+=is_gazing_1
        paper_cnt+=paper_gazing1

        f.write(f"personal cnt: ")
        for i in range(len(personal_cnt)):
            f.write(f"{i}:{personal_cnt[i]} ")
        print(personal_cnt)
        f.write(f"\npersonal real cnt: ")
        for i in range(len(personal_rcnt)):
            f.write(f"{i}:{personal_rcnt[i]} ")
        print(personal_rcnt)
        f.write(f'\ncnt: {cnt}\n\n')

        f.write(f"paper gaze personal cnt: ")
        for i in range(len(paper_personal_cnt)):
            f.write(f"{i}:{paper_personal_cnt[i]} ")
        print(paper_personal_cnt)
        f.write(f"\npaper gaze personal real cnt: ")
        for i in range(len(paper_personal_rcnt)):
            f.write(f"{i}:{paper_personal_rcnt[i]} ")
        print(paper_personal_rcnt)
        f.write(f'\npaper gaze cnt: {paper_cnt}\n\n')

        print(cnt)
        print(paper_cnt)
    f.close()
    print(cnt)

    
if __name__ == '__main__':
    main()

