import os
import numpy as np
import cv2
import os, glob

import glob

DIR2 = '/Users/Ghada/te_depth/new_Mask/Mask_RCNN-master/subdataset/2011_09_26_drive_0005_sync/image_03/data'

for filename in glob.glob(DIR2 + "*-fseg.png"):
    input_image = cv2.imread(filename)
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    nama = os.path.basename(filename)
    print('filename = ', nama)
    cv2.imwrite(DIR2 + nama, gray_image)
    #cv2.imwrite(OUTPUT_DIR + seqname2 + '/' + imgnum + '-fseg.png', big_img)

print('Done')
