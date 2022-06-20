import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import matplotlib.pyplot as plt
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr
import pytransform3d.rotations as pr
import pytransform3d.camera as pc

every_n_frames = 10

intrinsic_matrix = np.array([[367.3756, 0.0, 0.0], [0.0, 367.3756, 0.0], [256.5331, 205.893295, 1.0 ]])

pose_true = pd.read_csv('testing_results'  + '/pose_true.csv')
pose_true = pose_true.to_numpy()
pose_estim = pd.read_csv('testing_results'  + '/pose_estim.csv')
pose_estim = pose_estim.to_numpy()

cam2world_trajectory_true = ptr.transforms_from_pqs(pose_true)

plt.figure(figsize=(5, 5))
ax = pt.plot_transform(s=0.3)
ax = ptr.plot_trajectory(ax, P=pose_true, s=0.1)

image_size = np.array([960, 540])

for i in range (0, len(pose_true), every_n_frames):
    pc.plot_camera(ax, intrinsic_matrix, cam2world_trajectory_true[i],
                   sensor_size=image_size, virtual_image_distance=0.2, c="g") # green is for actual

cam2world_trajectory_estim = ptr.transforms_from_pqs(pose_estim)
#ax = ptr.plot_trajectory(ax, P=pose_estim, s=0.1, n_frames=every_n_frames)

for i in range (0, len(pose_true), every_n_frames):
    pc.plot_camera(ax, intrinsic_matrix, cam2world_trajectory_estim[i],
                   sensor_size=image_size, virtual_image_distance=0.2, c="r") # red is for prediction
plt.show()