
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" Offline data generation for the KITTI dataset."""

import os
from absl import app
from absl import flags
from absl import logging
import numpy as np
import cv2
import os, glob

import alignment
from alignment import compute_overlap
from alignment import align


SEQ_LENGTH = 3
WIDTH = 416   # the orignal version is 416 
HEIGHT = 128
STEPSIZE = 1

INPUT_DIR = '/Users/Ghada/PHD/subdataset-mask/'
OUTPUT_DIR = '/Users/Ghada/PHD/Mask-processed/2011_09_26/'

def get_line(file, start):
    file = open(file, 'r')
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]
    ret = None
    for line in lines:
        nline = line.split(': ')
        if nline[0]==start:
            ret = nline[1].split(' ')
            ret = np.array([float(r) for r in ret], dtype=float)
            ret = ret.reshape((3,4))[0:3, 0:3]
            break
    file.close()
    return ret


def crop(img, segimg, fx, fy, cx, cy):
    # Perform center cropping, preserving 50% vertically.
    middle_perc = 0.50
    left = 1-middle_perc
    half = left/2
    a = img[int(img.shape[0]*(half)):int(img.shape[0]*(1-half)), :]
    aseg = segimg[int(segimg.shape[0]*(half)):int(segimg.shape[0]*(1-half)), :]
    cy /= (1/middle_perc)

    # Resize to match target height while preserving aspect ratio.
    wdt = int((128*a.shape[1]/a.shape[0]))
    x_scaling = float(wdt)/a.shape[1]
    y_scaling = 128.0/a.shape[0]
    b = cv2.resize(a, (wdt, 128))
    bseg = cv2.resize(aseg, (wdt, 128))

    # Adjust intrinsics.
    fx*=x_scaling
    fy*=y_scaling
    cx*=x_scaling
    cy*=y_scaling

    # Perform center cropping horizontally.
    remain = b.shape[1] - 416
    cx /= (b.shape[1]/416)
    c = b[:, int(remain/2):b.shape[1]-int(remain/2)]
    cseg = bseg[:, int(remain/2):b.shape[1]-int(remain/2)]

    return c, cseg, fx, fy, cx, cy


def run_all():
  ct = 0
if not OUTPUT_DIR.endswith('/'):
    OUTPUT_DIR = OUTPUT_DIR + '/'

for d in glob.glob(INPUT_DIR + '/*/'):
    date = d.split('/')[-2]
    print('d = ', d)
    file_calibration = d + 'calib_cam_to_cam.txt'
    calib_raw = [get_line(file_calibration, 'P_rect_02'), get_line(file_calibration, 'P_rect_03')]

    for d2 in glob.glob(d + '*/'):
        print('d2= ',d2)
        seqname = d2.split('/')[-2]
        #for subfolder in ['image_02/data', 'image_03/data']:
        for subfolder in ['image_03/data']:    
            print('subfolder= ',subfolder)
            ct = 0
            seqname = d2.split('/')[-2] + subfolder.replace('image', '').replace('/data', '')
            print('seqname= ',seqname)
            
            if not os.path.exists(OUTPUT_DIR + seqname):
                os.mkdir(OUTPUT_DIR + seqname)
            
            calib_camera = calib_raw[0] if subfolder=='image_03/data' else calib_raw[1]
            folder = d2+subfolder 
            print('Processing folder', folder)
            #files = os.listdir(folder)
            files = glob.glob(folder + '/*.png')
            print('Processing files number= ',len(files))
            files = [file for file in files if not 'disp' in file and not 'flip' in file and not 'seg' in file]
            files = sorted(files)
            print('files number = ',len(files))
            count=0
            for i in range(SEQ_LENGTH, len(files)+1, STEPSIZE):
                imgnum = str(ct+1).zfill(10)
                #imgnum = str(count).zfill(10)
                if os.path.exists(OUTPUT_DIR + seqname + '/' + imgnum + '.png'):
                    ct+=1
                    print('ct= ',ct)
                    continue
                big_img = np.zeros(shape=(HEIGHT, WIDTH*SEQ_LENGTH, 3))
                #big_img = np.zeros(shape=(HEIGHT, 1248, 3))
                wct = 0
                wrt = 0

                for j in range(i-SEQ_LENGTH, i):  # Collect frames for this sample.
                    if wrt == 0:
                        img0 = cv2.imread(files[j])
                    elif wrt == 1:
                        img1 = cv2.imread(files[j])
                    elif wrt == 2:
                        img2 = cv2.imread(files[j])

                    wrt+=1


                    
                    ORIGINAL_HEIGHT, ORIGINAL_WIDTH, _ = img0.shape
                    
                    zoom_x = WIDTH/ORIGINAL_WIDTH
                    zoom_y = HEIGHT/ORIGINAL_HEIGHT
                   
                    # Adjust intrinsics.
                    calib_current = calib_camera.copy()
                    calib_current[0, 0] *= zoom_x
                    calib_current[0, 2] *= zoom_x
                    calib_current[1, 1] *= zoom_y
                    calib_current[1, 2] *= zoom_y

                    calib_representation = ','.join([str(c) for c in calib_current.flatten()])

                    if wrt == 3:
                        img0, img1, img2 = align(img0, img1, img2, threshold_same=0.5)
                        
                        img0 = cv2.resize(img0, (WIDTH, HEIGHT))
                        img1 = cv2.resize(img1, (WIDTH, HEIGHT))
                        img2 = cv2.resize(img2, (WIDTH, HEIGHT))
                        
                        big_img[:,0*WIDTH:(0+1)*WIDTH] = img0
                        big_img[:,1*WIDTH:(1+1)*WIDTH] = img1
                        big_img[:,2*WIDTH:(2+1)*WIDTH] = img2
                        count+=1

                #imgnum = imgnum[6:]
                
               
                cv2.imwrite(OUTPUT_DIR + seqname + '/' + imgnum + '-fseg.png', big_img)
                simpan = cv2.imread(OUTPUT_DIR + seqname + '/' + imgnum + '-fseg.png', 0)
                cv2.imwrite(OUTPUT_DIR + seqname + '/' + imgnum + '-fseg.png', simpan)
               
                ct+=1
              

def main(_):
  run_all()


if __name__ == '__main__':
  app.run(main)
