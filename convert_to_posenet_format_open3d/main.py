import glob
import numpy as np
from math import sqrt, pi, sin, cos, asin, acos, atan2, exp, log
import pathlib

def read_poses_from_log(traj_log):
	import numpy as np

	trans_arr = []
	with open(traj_log) as f:
		content = f.readlines()

		# Load .log file.
		for i in range(0, len(content), 5):
			print(i)
			# format %d (src) %d (tgt) %f (fitness)
			data = list(map(float, content[i].strip().split(' ')))
			ids = (int(data[0]), int(data[1]))
			fitness = data[2] 

			# format %f x 16
			T_gt = np.array(
				list(map(float, (''.join(
					content[i + 1:i + 5])).strip().split()))).reshape((4, 4))

			trans_arr.append(T_gt)

	return trans_arr

def rotation_matrix_to_quaternion(matrix):
	# stolen from pyquaternion
	m = matrix.conj().transpose() # This method assumes row-vector and postmultiplication of that vector
	if m[2, 2] < 0:
		if m[0, 0] > m[1, 1]:
			t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
			q = [m[1, 2]-m[2, 1],  t,  m[0, 1]+m[1, 0],  m[2, 0]+m[0, 2]]
		else:
			t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
			q = [m[2, 0]-m[0, 2],  m[0, 1]+m[1, 0],  t,  m[1, 2]+m[2, 1]]
	else:
		if m[0, 0] < -m[1, 1]:
			t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
			q = [m[0, 1]-m[1, 0],  m[2, 0]+m[0, 2],  m[1, 2]+m[2, 1],  t]
		else:
			t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
			q = [t,  m[1, 2]-m[2, 1],  m[2, 0]-m[0, 2],  m[0, 1]-m[1, 0]]
	q = np.array(q).astype('float64')
	q *= 0.5 / sqrt(t)
	return q

input_folder = "./validation/"
pose_file = input_folder+"trajectory.log"
image_dir = input_folder+"color/"

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
print(len(image_array), len(pose_array))
if len(image_array) == len(pose_array):
	count = 0
	for current_index in range (len(pose_array)):
		current_4x4_pose = pose_array[current_index]
		#current_4x4_pose = np.array([[0.211, -0.306, -0.928, 0.789], [0.662, 0.742, -0.0947, 0.147], [0.718,-0.595, 0.360, 3.26],[0,0,0,1]])
		current_image_path = image_array[current_index]
		current_3x3_rotation_matix = current_4x4_pose[:-1][:,:-1] 
		transform_matrix = current_4x4_pose[0:3:,3] 

		quaternion = rotation_matrix_to_quaternion(current_3x3_rotation_matix)
		coordinates = np.ascontiguousarray(transform_matrix)
		print(coordinates)
		#print(current_image_path)
		if count%every_n == 0:
			g.write(str(pathlib.Path(*pathlib.Path(current_image_path[0:]).parts[1:])) + " " + str(np.frombuffer(coordinates, dtype=float))[1:-1] +
			" " + str(str(np.frombuffer(quaternion, dtype=float))[1:-1]) + " \n")
		else:
			f.write(str(pathlib.Path(*pathlib.Path(current_image_path[0:]).parts[1:])) + " " + str(np.frombuffer(coordinates, dtype=float))[1:-1] +
			" " + str(str(np.frombuffer(quaternion, dtype=float))[1:-1]) + " \n")
		count = count + 1
	f.close()
	g.close()
else:
	print("error, not same number of poses and images")
