import glob
import numpy as np
from math import sqrt, pi, sin, cos, asin, acos, atan2, exp, log
import pathlib

def read_poses_from_log(traj_log):
	import numpy as np

	trans_arr = []
	array = np.loadtxt(traj_log, delimiter=' ', skiprows=1)
	array=np.delete(array,0,1)

	return array


input_folder = "./train/"
pose_file = input_folder+"poses.txt"
image_dir = input_folder+"rgb/"

pose_array = read_poses_from_log(pose_file)
image_array = sorted(glob.glob(image_dir+'*.png'))

f = open(input_folder+'/sequence-train.txt', 'w')
f.write("Custom Posenet Dataset, in tue format of King's College Dataset\n'")
f.write('ImageFile, Camera Position [X Y Z W P Q R]\n') 
f.write('\n') 

g = open(input_folder+'/sequence-test.txt', 'w')
g.write("Custom Posenet Dataset, in tue format of King's College Dataset\n'")
g.write('ImageFile, Camera Position [X Y Z W P Q R]\n') 
g.write('\n') 
every_n = 10
count = 0
for current_index in range (len(image_array)):
	current_all = pose_array[current_index]
	current_image_path = image_array[current_index]
	current_all  = np.insert(current_all, 3, current_all[6])
	current_all = current_all[:-1]
	print(str(pathlib.Path(*pathlib.Path(current_image_path[0:]).parts[1:])) + " " + str(np.frombuffer(current_all, dtype=float))[1:-1]+ " \n")
	if count%every_n == 0:
		g.write(str(pathlib.Path(*pathlib.Path(current_image_path[0:]).parts[1:])) + " " + str(np.frombuffer(current_all, dtype=float))[1:-1].replace('\n', '')+ "\n")
	else:
		f.write(str(pathlib.Path(*pathlib.Path(current_image_path[0:]).parts[1:])) + " " + str(np.frombuffer(current_all, dtype=float))[1:-1].replace('\n', '')+ "\n")
	count = count + 1
f.close()
g.close()
