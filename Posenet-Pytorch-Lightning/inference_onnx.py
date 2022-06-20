from data_loader import get_loader
import torch
import onnxruntime
from pose_utils import *
import time
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()



image_path = "./validation/"
#model_name = "Googlenet"
model_name = "Resnet"
metadata_path = "./validation/sequence-train.txt"
summary_save_path = './testing_results/'
mode = "test"
batch_size = 1
shuffle = True
data_loader = get_loader(model_name, image_path, metadata_path, mode, batch_size,
                             shuffle)
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_session = onnxruntime.InferenceSession("efficient.onnx", so, providers=['CPUExecutionProvider'])
device = torch.device("cpu")

total_pos_loss = 0
total_ori_loss = 0
pos_loss_arr = []
ori_loss_arr = []
true_pose_list = []
estim_pose_list = []


num_data = len(data_loader)
start_time = time.time()
torch.backends.cudnn.benchmark=True
with torch.inference_mode():
    for i, (inputs, poses) in enumerate(data_loader):
        print(i)
        
        inputs = inputs.to(device)
        
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}
        pos_out, ori_out = ort_session.run(None, ort_inputs)#model(inputs)
        pos_out = pos_out[0]
        #ori_out = F.normalize(ori_out, p=2, dim=1)
        ori_out = quat_to_euler(ori_out.squeeze(0))
        print('pos out', pos_out)
        print('ori_out', ori_out)

        pos_true = poses[:, :3].squeeze(0).numpy()
        ori_true = poses[:, 3:].squeeze(0).numpy()

        ori_true = quat_to_euler(ori_true)
        print('pos true', pos_true)
        print('ori true', ori_true)
        loss_pos_print = array_dist(pos_out, pos_true)
        loss_ori_print = array_dist(ori_out, ori_true)

        true_pose_list.append(np.hstack((pos_true, ori_true)))
        
        if loss_pos_print < 20:
            estim_pose_list.append(np.hstack((pos_out, ori_out)))


        print(pos_out)
        print(pos_true)

        total_pos_loss += loss_pos_print
        total_ori_loss += loss_ori_print

        pos_loss_arr.append(loss_pos_print)
        ori_loss_arr.append(loss_ori_print)
position_error = np.median(pos_loss_arr)
rotation_error = np.median(ori_loss_arr)

print('=' * 20)
print('Overall median pose errer {:.3f} / {:.3f}'.format(position_error, rotation_error))
print('Overall average pose errer {:.3f} / {:.3f}'.format(np.mean(pos_loss_arr), np.mean(ori_loss_arr)))
print("--- %s batches per seconds ---" % (len(data_loader)/(time.time() - start_time)))

f_true = summary_save_path + '/pose_true.csv'
f_estim = summary_save_path + '/pose_estim.csv'
np.savetxt(f_true, true_pose_list, delimiter=',')
np.savetxt(f_estim, estim_pose_list, delimiter=',')
