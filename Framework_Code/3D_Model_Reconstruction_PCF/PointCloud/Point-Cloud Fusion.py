# Author Ghada M.Fathy 
# Informatics Research Institute, City for Scientific Research and Technological Applications,
# SRTA-City, Alexandria, Egypt
# 2020
# Generate 3D Model Point Cloud Fusion
#--------------------------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import time
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
from math import cos, sin, radians
import cv2
import sys
import open3d as o3d
from PIL import Image
import imageio
from mapPoint import mapPoint
from hashMapCoords import hashMapCoords
from OpenGL.GL import *
from OpenGL.GLU import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

NUM_SCALES=4
#gfile = tf.gfile
gfile=tf.io.gfile
#flipping_mode=reader.FLIP_RANDOM

flags.DEFINE_string('data_dir', None, 'Preprocessed data.')
flags.DEFINE_string('input_dir', None, 'npy depth.')
#flags.DEFINE_string('egomotion_dir', None, 'the direction of egomotion file.')
#flags.DEFINE_string('raw_image', None, 'the direction of raw image.')

flags.DEFINE_string('file_extension', 'png', 'Image data file extension.')
flags.DEFINE_integer('num_scales',4, 'NUM_SCALES.')
flags.DEFINE_integer('batch_size', 2, 'The size of a sample batch')
flags.DEFINE_integer('img_height', 128, 'Input frame height.')
flags.DEFINE_integer('img_width', 416, 'Input frame width.')
flags.DEFINE_integer('seq_length', 3, 'Number of frames in sequence.')
FLAGS = flags.FLAGS
flags.mark_flag_as_required('data_dir')
flags.mark_flag_as_required('input_dir')
#flags.mark_flag_as_required('egomotion_dir')
#flags.mark_flag_as_required('raw_image')

#........................................................................................................                                                
def trig(angle):
  r = radians(angle)
  return cos(r), sin(r) 
#........................................................................................................
def matrix(rotation, translation):
  xC, xS = trig(rotation[0])
  yC, yS = trig(rotation[1])
  zC, zS = trig(rotation[2])
  dX = translation[0]
  dY = translation[1]
  dZ = translation[2]
  Translate_matrix = np.array([[1, 0, 0, dX],
                               [0, 1, 0, dY],
                               [0, 0, 1, dZ],
                               [0, 0, 0, 1]])
  Rotate_X_matrix = np.array([[1, 0, 0, 0],
                              [0, xC, -xS, 0],
                              [0, xS, xC, 0],
                              [0, 0, 0, 1]])
  Rotate_Y_matrix = np.array([[yC, 0, yS, 0],
                              [0, 1, 0, 0],
                              [-yS, 0, yC, 0],
                              [0, 0, 0, 1]])
  Rotate_Z_matrix = np.array([[zC, -zS, 0, 0],
                              [zS, zC, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
  return np.dot(Rotate_Z_matrix,np.dot(Rotate_Y_matrix,np.dot(Rotate_X_matrix,Translate_matrix)))
#........................................................................................................  
# Points generator
def get_grid(x, y, homogenous=False):
    coords = np.indices((x, y)).reshape(2, -1)
    return np.vstack((coords, np.ones(coords.shape[1]))) if homogenous else coords
    
#........................................................................................................
def _meshgrid_abs(height, width):
  """Meshgrid in the absolute coordinates."""
  x_t = tf.matmul(
      tf.ones(shape=tf.stack([height, 1])),
      tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
  y_t = tf.matmul(
      tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
      tf.ones(shape=tf.stack([1, width])))
  x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
  y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
  x_t_flat = tf.reshape(x_t, (1, -1))
  y_t_flat = tf.reshape(y_t, (1, -1))
  ones = tf.ones_like(x_t_flat)
  grid = tf.concat([x_t_flat, y_t_flat, ones], axis=0)
  return grid  
#........................................................................................................
def _cam2pixel(cam_coords, proj_c2p):
  """Transform coordinates in the camera frame to the pixel frame."""
  pcoords = tf.matmul(proj_c2p, cam_coords)
  x = tf.slice(pcoords, [0, 0, 0], [-1, 1, -1])
  y = tf.slice(pcoords, [0, 1, 0], [-1, 1, -1])
  z = tf.slice(pcoords, [0, 2, 0], [-1, 1, -1])
  # Not tested if adding a small number is necessary
  x_norm = x / (z + 1e-10)
  y_norm = y / (z + 1e-10)
  pixel_coords = tf.concat([x_norm, y_norm], axis=1)
  return pixel_coords  
#........................................................................................................  
def arreqclose_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if elem.size == myarr.size and allclose(elem, myarr)), False)
#........................................................................................................   
def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if array_equal(elem, myarr)), False)
#........................................................................................................    
def save_image(img_file, im, file_extension):
  """Save image from disk. Expected input value range: [0,1]."""
  im = (im * 255.0).astype(np.uint8)
  with gfile.Open(img_file, 'w') as f:
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    _, im_data = cv2.imencode('.%s' % file_extension, im)
    #dim = (1280, 385)
    #im_data = cv2.resize(im_data, dim, interpolation = cv2.INTER_AREA)
    f.write(im_data.tostring())
    
#........................................................................................................    
def calc_avg_weight(depth,segma=0.6):
     depth_norm=tf.linalg.norm(depth)                                           # normalization of current frame depth 
     depth=depth/depth_norm
     avg_w_array=tf.math.square(depth)*-1
     s=2*segma*segma
     avg_w_array=avg_w_array/s
     avrege_weight=tf.math.exp(avg_w_array)
     print('avrege_weight',avrege_weight.shape)
     return avrege_weight
#........................................................................................................     
def drow_point(pointCloud,pointColor):
    print('pointCloud=',pointCloud.shape)
    points=np.array(pointCloud)
    #points=pointCloud.eval(session=tf.compat.v1.Session())
    print('X= ',points[0,0])
    
    R=(pointColor[:,0]/255)*1.0
    G=(pointColor[:,1]/255)*1.0
    B=(pointColor[:,2]/255)*1.0
    gl.glPointSize(53248)
   
    glBegin(GL_POINTS)
    for i in range(0,53248):
        glColor3d(R[i], G[i], B[i])
        glVertex3d(points[0,i], points[1,i], points[2,i])
    glEnd()
#........................................................................................................    
def save_pointCloud(pointCloud,pointColor):
    print('pointCloud=',pointCloud.shape)
    points=np.array(pointCloud)
    print('points_size= ',points.shape) 
    R=(pointColor[:,0]/255)*1.0
    G=(pointColor[:,1]/255)*1.0
    B=(pointColor[:,2]/255)*1.0
    f= open("/Users/Ghada/estimate_depth/new_Mask/Mask_RCNN-master/points.xyz","w+")
    for i in range(0,53248):
        #f.write("{} {} {} {} {} {}\n".format(points[0,i], points[1,i], points[2,i],R[i],G[i],B[i]))
        f.write("{} {} {} {} {}\n".format(points[0,i], points[1,i],R[i],G[i],B[i]))
        #np.savetxt(,(R[i], G[i], B[i]))
        #glVertex3d(points[0,i], points[1,i], points[2,i])
    f.close()
#........................................................................................................    
def draw_pointCloud():
    input_path="/Users/Ghada/estimate_depth/new_Mask/Mask_RCNN-master/points.xyz"
    output_path="/Users/Ghada/estimate_depth/new_Mask/Mask_RCNN-master"
    point_cloud= np.loadtxt(input_path, skiprows=1, max_rows=53249) 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(point_cloud[:,3:6])
    
#........................................................................................................    
def update_points(pointId,ref_pointcloud,pointcloud,current_weight,uncertainty,isStable,old_weight,MAX_WEIGHT_LIMIT,max_fusion_weight,map_objects):
    new_weight=1
    point_stable_thr=0.001
    stable_delay_loop_thr=0.001
    update_pointcloud=((old_weight*ref_pointcloud)+(new_weight*pointcloud))/(new_weight+old_weight)
    update_uncertainty=((new_weight*current_weight)+(old_weight*uncertainty))/(new_weight+old_weight)
    if update_uncertainty > MAX_WEIGHT_LIMIT:
       update_uncertainty=MAX_WEIGHT_LIMIT
    new_weight=new_weight+old_weight
    update_weight=min(new_weight,max_fusion_weight) 
    if isStable:
       stable_thr = point_stable_thr + stable_delay_loop_thr
       if update_uncertainty < stable_thr:
          update_isStable = True
       else:
          update_isStable = False
        
    else:
       stable_thr = point_stable_thr - stable_delay_loop_thr

       if update_uncertainty < stable_thr:
          update_isStable = True
       else:
          update_isStable = False
  
    map_objects.upadate_mapPoint(pointId,update_pointcloud,update_uncertainty,update_isStable,update_weight)
    
 
#*****************************************************************************************    
def main(_):
    global read
    global translation_matrix
    global intrinsic_mat
    global coords
    global depths
    Max_points_num=53248
    #motion_dir=FLAGS.egomotion_dir
    data_dir=FLAGS.data_dir
    input_dir=FLAGS.input_dir
    img_height=128
    img_width=416
    key_frame_num=1
    depth_list=[]
    image_list=[]
    egomotion_list=[]
    points_cloud=[]
    m_points=[]
    m_Color=[]
    Map=[]
    map_objects=mapPoint(m_points,m_Color)
    map_index=hashMapCoords(Map)
    MAX_WEIGHT_LIMIT=1.0
    point_dist_thr=0.1
    max_fusion_weight=100
    point_init_uncertainty=0.1
    pointID=0
    # ----------------------------read and create intrinsic_mat ------------------------------
    with open(os.path.join(data_dir, '0000000001_cam.txt'), 'r') as f:
           cam_ = f.read().split(',')
    intrinsic_mat = np.array(cam_, dtype=np.float32) 
    intrinsic_mat=np.reshape(intrinsic_mat,[3,3])
    intrinsic=np.reshape(intrinsic_mat,[1,3,3])    
    intrinsic_mat_inv=np.linalg.inv(intrinsic_mat) 
    hom_filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    hom_filler = tf.tile(hom_filler, [1, 1, 1])
    intrinsic_mat_hom = tf.concat([intrinsic, tf.zeros([1, 3, 1])], axis=2)
    intrinsic_mat_hom = tf.concat([intrinsic_mat_hom, hom_filler], axis=1)              # size 1x4x4
    print('****intrinsic_mat_hom= ',intrinsic_mat_hom.shape)
    #-------------------------read from files ------------------------------------------------
    for file in os.listdir(input_dir):
        if file.endswith(".png"):
           img = cv2.imread(input_dir+file) 
           img = cv2.resize(img, (img_height, img_width))
           image_list.append(img)
        elif file.endswith(".npy"):     
           depth_array=np.load(input_dir+file)
           depth_list.append(depth_array)
        elif file.endswith(".txt"): 
           with open(input_dir+file, 'r') as f:
                motion = f.read().split(',')        
           egomotion_list.append(motion)
           
    depths=np.array(depth_list) 
    images=np.array(image_list) 
    
    egomotion=np.array(egomotion_list) 
    egomotion_float = np.array(egomotion, dtype=np.float32) 
    homog_coords = _meshgrid_abs(img_height,img_width)                                   #   Tt(u) homogenouscoords [x,y,1]
 
    # -----------------------------loop on 20 frame to generate point cloud for each frame ------------------------------
    
    for i in range(0,20):                                                                # 20 frames the sequence of video 
         start = time.time()
         
         translation= egomotion_float[i,0:3]
         rotation=egomotion_float[i,3:6]
         translation_matrix = matrix(rotation, translation)                              # Tcw  which calculate from struct2depth 4x4
         translation_matrix_inv=np.linalg.inv(translation_matrix)                        # (Tcw) power -1 (4x4)
         translation_matrix_inv=tf.convert_to_tensor(translation_matrix_inv, np.float32) #(Tcw) power -1 in tensor shape
         depth_array=np.reshape(depths[i,:,:],(1,img_width*img_height))                  # Di the current depth map 1x53248
         depth_tens=tf.convert_to_tensor(depth_array, np.float32)                        # Di the current depth map in tensor shape
         p=tf.matmul(intrinsic_mat_inv, homog_coords) * depth_tens                       #K power -1 *ui*Di size 3x53248
    
         p=tf.reshape(p,[1,3,img_height * img_width])                                    # P shape= 1x3x53248 
         ones = tf.ones([1,1, img_height * img_width])
         points_coords_hom = tf.concat([p, ones], axis=1)                                #points_coords_hom =1x4x53248
         
         translation_matrix_inv=tf.reshape(translation_matrix_inv,[1,4,4])
         
                                                                                         # point_cloud refer to pw in paper for the current frame
         point_cloud=tf.matmul(translation_matrix_inv,points_coords_hom)                 # pw = Tcw power -1 * K power -1 *ui*Di output size 1x4x53248
        
         avrege_weight=calc_avg_weight(depth_tens)                                       # w (average weighet for pixel size= 1x53248) 
         
         # --------------------------------------------get RGB for each frame  for draw------------------------------------
         Color=images[i,:,:,:]
         Color=np.reshape(Color,[416*128,3])
         
         
         # -------------------------------------------project world coord to ref_keyframe coord----------------------------
         for k in range(0,5): #key_frame_num 
            start = time.time()
            print('key frame number: ',k)
            k_translation=egomotion_float[(k*5)+1,0:3]
            k_rotation=egomotion_float[(k*5)+1,3:6]
            k_transformation_matrix=matrix(k_rotation,k_translation)
            k_transformation_matrix=tf.convert_to_tensor(k_transformation_matrix,np.float32)
            k_transformation_matrix=tf.reshape(k_transformation_matrix,[1,4,4])
            pointc=tf.matmul(intrinsic_mat_hom,k_transformation_matrix)                  # 1x4x4 mul 1x4x4
            uk_coords=_cam2pixel(point_cloud,pointc)                                     # uk for current keyframe size size 1x2x53248
            uk_coords=tf.dtypes.cast(uk_coords, tf.int32)
            #save_pointCloud(uk_coords[0,:,:],Color)
            #----------------------------- now prepation for updates by compare each pixel with the selected keyframes---------------
            
            for pixel in range(0,Max_points_num) :                                                    # loop for each pixel and search for it in map 53248
                
                start1 = time.time()
                uv_ref_coords=uk_coords[0,:,pixel]                                       # note source paper convert it to integer ?? hashmap([x,y],pointID)
                inti_avg_weight=avrege_weight[0,pixel]
                P=point_cloud[0,:3,pixel]
                map_size=map_objects.map_point_len()
                
                pointID=map_index.search_in_Map(uv_ref_coords)
                
                if pointID!=-1 and pointID < map_size:                                   # if pointID found in hashmap get with point id the following (ref_pointCloud[x,y,z](
                   
                   #print('point found into index_map update !!!') 
                   ref_pointCloud,uncertinity,isStable,avg_weight=map_objects.get_mapPoint(pointID)
                   dist=ref_pointCloud-P
                   dist_len=np.dot(dist,dist)
                   if dist_len < MAX_WEIGHT_LIMIT:
                      Current_weight=dist_len
                   else:
                      Current_weight=MAX_WEIGHT_LIMIT
                      
                   if dist_len < point_dist_thr :
                       update_points(pointID,ref_pointCloud,P,Current_weight,uncertinity,isStable,avg_weight,MAX_WEIGHT_LIMIT,max_fusion_weight,map_objects)
                else:                                                                     # not exist, then create new pointcloud 
                    #print('point not found add new point!!!')
      
                    pointId = map_size
                    uncertinity=point_init_uncertainty
                    P=point_cloud[0,:3,pixel]
                    isStable = False
                    uncertainty =point_init_uncertainty
                    avg_weight=inti_avg_weight
                    map_objects.add_mapPoint(pointId,P,uncertinity,isStable,avg_weight)
                    map_objects.add_color(pointId,Color[pixel,:])
                    
                    map_index.add_mapPoint(uv_ref_coords,pointId)
                    end1=time.time()
                #print('iteration= (%s )time= (%s)  id = (%s)',(pixel,end1 - start1,pointID))
                start1=0
                end1=0
            end= time.time()
            print('time for frame =',(end - start))
            start=0
            end=0
              
    
    print('map_size=',map_objects.map_point_len())  
    print('map_Color_size=',map_objects.map_index_len()) 
    print('map_index_size=',map_index.hash_len())             
    map_objects.save_pointCloud(Color,map_objects.map_point_len())
     
         
    #---------------------------------------------------------------------
     
    
if __name__ == '__main__':
  app.run(main) 