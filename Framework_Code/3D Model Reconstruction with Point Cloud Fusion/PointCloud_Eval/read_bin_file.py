# Author Ghada M.Fathy 
# Informatics Research Institute, City for Scientific Research and Technological Applications,
# SRTA-City, Alexandria, Egypt
# 2020
# Registration between Ground Truth and predected frames 
#--------------------------------------------------------------------------------------------------
import numpy as np
import open3d as o3d
import copy

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp,target_temp])
def registration():
   source = o3d.io.read_point_cloud('/Users/Ghada/Eval/fames/GroundTruth/GT_full.xyz')
   target = o3d.io.read_point_cloud('/Users/Ghada/Eval/fames/Predict_fames/P_F.xyz')
   threshold = 0.16
   trans_init = np.asarray([[7.533745e-03, -9.999714e-01, -6.166020e-04, -0.00406977],
                         [1.480249e-02,  7.280733e-04, -9.998902e-01, -0.07631618],
                         [9.998621e-01,  7.523790e-03,  1.480755e-02, -0.2717806],
                         [0.0, 0.0, 0.0, 1.0]])
   print("Initial alignment")
   evaluation = o3d.registration.evaluate_registration(source, target, threshold, trans_init)
   print(evaluation)
   print("Apply point-to-point ICP")
   reg_p2p = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint())
   print(reg_p2p)
   #print("Transformation is:")
   #print(reg_p2p.transformation)
   
   #draw_registration_result(source, target, reg_p2p.transformation)

def down_Sampling(fileName):
  print("Load a ply point cloud, print it, and render it")
  pcd = o3d.io.read_point_cloud(fileName)
 
  downpcd = pcd.voxel_down_sample(voxel_size=0.047) 
  print(points.shape)
  return points

def load_velodyne_points(file_name):
    points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
    print(points.shape)
    return points
    
def save_Frame_pointCloud(points,fileName):
    poin_num,_=points.shape

    f= open("/Users/Ghada/Eval/fames/"+fileName,"w+")
    for i in range(1,poin_num):
        f.write("{} {} {} \n".format(points[i,0], points[i,1], points[i,2]))
        
        
    f.close()  
def load_from_file(file_path,max_r):
    point_cloud= np.loadtxt(file_path, skiprows=1, max_rows=max_r)   
    points=point_cloud[:,:3]
    return points
    


