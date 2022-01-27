# Author Ghada M.Fathy 
# Informatics Research Institute, City for Scientific Research and Technological Applications,
# SRTA-City, Alexandria, Egypt
# 2020
#--------------------------------------------------------------------------------------------------
from absl import app
import tensorflow as tf
import numpy as np
class hashMapCoords(object):
     def __init__(self,M_point):
         self.M_point=M_point
         
     def add_mapPoint(self,point,pointId): 
         #points=np.array(point)     
         self.M_point.append((True,point,pointId))
     
     def search_in_Map(self,point):
         pointId=0
         #points=np.array(point)
         for x in self.M_point:
            if tf.math.reduce_all(tf.equal(x[1], point)):
            #if np.array_equal(x[1], point):
               pointId=x[2]
               break
            else: 
               pointId=-1 
         return pointId
                 
     
     def hash_len(self):
        return len(self.M_point) 
