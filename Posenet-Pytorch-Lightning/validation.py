from model.model import GoogleNetPoseNet, ResNet34PoseNet, EfficientNetB0PoseNet, DenseNet121PoseNet, InceptionResNetV2PoseNet
from model.model_transformer import DeiTTinyPoseNet, DeiTSmallPoseNet, DeiTBasePoseNet
import pytorch_lightning as pl
from data_loader import get_loader
import torch
import torch.nn.functional as F
import torch.onnx
from pose_utils import *
import time
import os




image_path = "./train/"
#model_name = "Googlenet"
model_name = "Resnet"
metadata_path = "./train/sequence-train.txt"
summary_save_path = './testing_results/'
try:
    os.mkdir(summary_save_path)
except:
    pass
mode = "test"
batch_size = 1
shuffle = False
data_loader = get_loader(model_name, image_path, metadata_path, mode, batch_size,
                             shuffle)
checkpoint_name = "resnet.ckpt"
model = ResNet34PoseNet.load_from_checkpoint(checkpoint_name)
model.eval()
f = open(summary_save_path+'/test_result.csv', 'w')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
model = model.to(device)
x = torch.randn(batch_size, 3, 224, 224, requires_grad=True).cuda() 
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  checkpoint_name[:-5]+".onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True)  # whether to execute constant folding for optimization)
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

        torch.cuda.synchronize()
        pos_out, ori_out = model(inputs)
        torch.cuda.synchronize()

        pos_out = pos_out.squeeze(0).detach().cpu().numpy()
        ori_out_quad = F.normalize(ori_out, p=2, dim=1)
        ori_out = quat_to_euler(ori_out_quad.squeeze(0).detach().cpu().numpy())
        print('pos out', pos_out)
        print('ori_out', ori_out)

        pos_true = poses[:, :3].squeeze(0).numpy()
        ori_true_quad = poses[:, 3:].squeeze(0).numpy()

        ori_true = quat_to_euler(ori_true_quad)
        print('pos true', pos_true)
        print('ori true', ori_true)
        loss_pos_print = array_dist(pos_out, pos_true)
        loss_ori_print = array_dist(ori_out, ori_true)
        true_pose_list.append(np.concatenate((pos_true, ori_true_quad), axis=0))        
        if loss_pos_print < 20:
            print(ori_out_quad.cpu().numpy())
            estim_pose_list.append(np.concatenate((pos_out, ori_out_quad.cpu().numpy()[0]), axis=0))


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
f.close()
print("--- %s batches per seconds ---" % (len(data_loader)/(time.time() - start_time)))

f_true = summary_save_path + '/pose_true.csv'
f_estim = summary_save_path + '/pose_estim.csv'
np.savetxt(f_true, true_pose_list, delimiter=',')
np.savetxt(f_estim, estim_pose_list, delimiter=',')
