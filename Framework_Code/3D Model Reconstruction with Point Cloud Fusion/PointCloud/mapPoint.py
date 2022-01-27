# Author Ghada M.Fathy 
# Informatics Research Institute, City for Scientific Research and Technological Applications,
# SRTA-City, Alexandria, Egypt
# 2020
#--------------------------------------------------------------------------------------------------
from absl import app
import numpy as np
class mapPoint(object):
     def __init__(self,M_point,P_Color):
         self.M_point=M_point
         self.P_Color=P_Color
     #...........................................................................................    
     def add_mapPoint(self,pointId,point,uncertinity,isStable,avg_weight): 
         
         self.M_point.append((True,pointId,point,uncertinity,isStable,avg_weight))
     #...........................................................................................       
     def add_color(self,pointId,RGB_Color):
         self.P_Color.append((True,pointId,RGB_Color))
     #...........................................................................................       
     def upadate_mapPoint(self,pointId,point,uncertinity,isStable,avg_weight):
         for x in self.M_point:
            if pointId==x[1]:
               self.M_point[pointId]=True,pointId,point,uncertinity,isStable,avg_weight
               
     #...........................................................................................             
     def map_point_len(self):
        return len(self.M_point) 
     #........................................................................................... 
     def map_index_len(self):
        return len(self.P_Color)
     #...........................................................................................
     def get_mapPoint(self,point_id):
         p=self.M_point[point_id]
         point=p[2]
         uncertinity=p[3]
         isStable=p[4]
         avg_weight=p[5]
         return point,uncertinity,isStable,avg_weight
     #...........................................................................................       
     def print_map(self):
        print(self.M_point)
     #...........................................................................................
     def save_pointCloud(self,pointColor,index):
        
        R=(pointColor[:,0]/255)*1.0
        G=(pointColor[:,1]/255)*1.0
        B=(pointColor[:,2]/255)*1.0
        f= open("/Users/Ghada/estimate_depth/new_Mask/Mask_RCNN-master/test.xyz","w+")
        j=0
        for i in self.M_point:
          point=i[2]
          #print(points[0])
          f.write("{} {} {} {} {} {}\n".format(point[0], point[1], point[2],R[j],G[j],B[j]))
          j=j+1
        
        f.close()     
