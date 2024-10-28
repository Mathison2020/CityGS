import os
from plyfile import PlyData, PlyElement
import numpy as np
import pandas as pd


current_dir = os.path.dirname(__file__)
file_dir = current_dir + '/output/my_scene/point_cloud/iteration_30000/point_cloud.ply'
plydata = PlyData.read(file_dir)
data = plydata.elements[0].data
data_pd = pd.DataFrame(data)
data_np = np.zeros(data_pd.shape, dtype=float)
property_names = data[0].dtype.names
for i, name in enumerate(property_names):
    data_np[:, i] = data_pd[name]
print(data_np.shape)