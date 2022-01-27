# Author Ghada M.Fathy 
# Informatics Research Institute, City for Scientific Research and Technological Applications,
# SRTA-City, Alexandria, Egypt
# 2020
# Ealuation Metrics Localization Accuracy Error L_E, FPE (False positive error),
# FNE (False Negative error), and mean relative error (MRE)
#--------------------------------------------------------------------------------------------------
from __future__ import division
import numpy as np


def load_from_file(file_path,max_r):
    point_cloud= np.loadtxt(file_path, skiprows=1, max_rows=max_r)   
    points=point_cloud[:,:3]
    return points

def load_from_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    # ignore reflectivity info
    return obj[:,:3]
    
def Euclidean_distance(groundTruth,predicted,r):
    gr,gc=groundTruth.shape
    Na,pc=predicted.shape
    min_dist=0
    dist_list=[]
    
    for i in range(0,Na):
      dist=np.sqrt((groundTruth[i,0]-predicted[:,0])**2+(groundTruth[i,1]-predicted[:,1])**2+(groundTruth[i,2]-predicted[i,2])**2)
      min_dist=np.min(dist)
      if(min_dist<=r):
        dist_list.append(min_dist)
    
    
    corrected_points=np.array(dist_list)
    Nc=len(corrected_points)#corrected_points.shape
    Le=np.sum(corrected_points)
    Le=Le/Nc
    print('Loclization Error=',Le)
    
    FNE=1-(Nc/gr)
    print('FNE= ',FNE)
    FPE=(Na-Nc)/Na
    print('FPE=',FPE)
    
def Mean_retrivale_Error(groundTruth,predicted,r,num_point):
    gr,gc=groundTruth.shape
    Na,pc=predicted.shape
    list=[]
    z=np.abs(predicted[:,2]-groundTruth[:,2])
    for i in range(0,Na):
      if z[i]!=0 and groundTruth[i,2]!=0:
       p=z[i]/(groundTruth[i,2])
       list.append(p)
    t=np.array(list)
    sum_array=np.sum(t[np.isfinite(t)])
      
    MRE=(sum_array/num_point)
    print('MRE=',MRE)  

     
