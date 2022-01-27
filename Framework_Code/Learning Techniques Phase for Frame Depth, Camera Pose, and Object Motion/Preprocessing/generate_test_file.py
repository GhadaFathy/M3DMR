
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


INPUT_DIR = '/mnt/e/PHD/test files'
OUTPUT_DIR = '/mnt/e/PHD/test files'




def run_all():
  ct = 0
if not OUTPUT_DIR.endswith('/'):
    OUTPUT_DIR = OUTPUT_DIR + '/'
#output_filepath = os.path.join(OUTPUT_DIR, 'sample1.txt')
for d in glob.glob(INPUT_DIR + '/*/'):
    date = d.split('/')[-2]
    ct = 0
    for d2 in glob.glob(d + '*/'):
        seqname = d2.split('/')[-2]
        print('Processing sequence', seqname)
        for subfolder in ['data/']:
             
            folder = d2+subfolder 
            print("folder = ", folder)
            files = glob.glob(folder + '/*.png')
            files = sorted(files)
            for i in files:
            
                x=files[ct].split('/')
                #increase image id with 1 to start from 1 need to modify dont forget
                image_path=x[5]+'/'+x[6]+'/'+x[7]+'/'+x[8]
                print("image_path = ", image_path)
                with open(OUTPUT_DIR+'sample1.txt', "a") as myfile:
                      myfile.write(image_path+ '\n')

                #current_output_handle.write(image_path+ '\n')
                ct+=1
#current_output_handle.close()
def main(_):
  run_all()


if __name__ == '__main__':
  app.run(main)
