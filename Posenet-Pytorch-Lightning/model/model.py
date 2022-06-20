import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, datasets
from densenet import DenseNet
from inceptionresnet import Inception_ResNetv2
import torch.nn.functional as F
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet

class PoseLoss(nn.Module):
	def __init__(self, device, sx=0.0, sq=0.0, learn_beta=False):
		super(PoseLoss, self).__init__()
		self.learn_beta = learn_beta

		if not self.learn_beta:
			self.sx = 0
			self.sq = -6.25
			
		self.sx = nn.Parameter(torch.Tensor([sx]), requires_grad=self.learn_beta)
		self.sq = nn.Parameter(torch.Tensor([sq]), requires_grad=self.learn_beta)

		# if learn_beta:
		#     self.sx.requires_grad = True
		#     self.sq.requires_grad = True
		#
		# self.sx = self.sx.to(device)
		# self.sq = self.sq.to(device)

		self.loss_print = None

	def forward(self, pred_x, pred_q, target_x, target_q):
		pred_q = F.normalize(pred_q, p=2, dim=1)
		loss_x = F.l1_loss(pred_x, target_x)
		loss_q = F.l1_loss(pred_q, target_q)

			
		loss = torch.exp(-self.sx)*loss_x \
			   + self.sx \
			   + torch.exp(-self.sq)*loss_q \
			   + self.sq

		self.loss_print = [loss.item(), loss_x.item(), loss_q.item()]

		return loss, loss_x.item(), loss_q.item()
		
    
class ResNet34PoseNet(pl.LightningModule):
	""" PoseNet using Resnet-34 """
	def __init__(self, pretrained=False, fixed_weight=False, dropout_rate = 0.2):
		super(ResNet34PoseNet, self).__init__()
		base_model = models.resnet34(pretrained=pretrained)

		self.criterion = PoseLoss(self.device)
		self.dropout_rate = dropout_rate
		feat_in = base_model.fc.in_features

		self.base_model = nn.Sequential(*list(base_model.children())[:-1])
		# self.base_model = base_model

		if fixed_weight:
			for param in self.base_model.parameters():
				param.requires_grad = False

		self.fc_last = nn.Linear(feat_in, 2048, bias=True)
		
		self.fc_position = nn.Linear(2048, 3, bias=True)
		self.fc_rotation = nn.Linear(2048, 4, bias=True)

		init_modules = [self.fc_last, self.fc_position, self.fc_rotation]

		for module in init_modules:
			if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
				nn.init.kaiming_normal_(module.weight)
				if module.bias is not None:
					nn.init.constant_(module.bias, 0)

		# nn.init.normal_(self.fc_last.weight, 0, 0.01)
		# nn.init.constant_(self.fc_last.bias, 0)
		#
		# nn.init.normal_(self.fc_position.weight, 0, 0.5)
		# nn.init.constant_(self.fc_position.bias, 0)
		#
		# nn.init.normal_(self.fc_rotation.weight, 0, 0.01)
		# nn.init.constant_(self.fc_rotation.bias, 0)

	def forward(self, x):
		x = self.base_model(x)
		#print(x.shape)
		x = x.view(x.size(0), -1)
		x_fully = self.fc_last(x)
		x = F.relu(x_fully)

		dropout_on = self.training 
		if self.dropout_rate > 0:
			x = F.dropout(x, p=self.dropout_rate, training=dropout_on)
		#print(x.shape)
		position = self.fc_position(x)
		rotation = self.fc_rotation(x)

		return position, rotation
	
	def configure_optimizers(self):
		opt = torch.optim.AdamW(self.parameters(), lr=1e-4)
		return opt
	def training_step(self, batch, batch_idx):
		input_image, poses  = batch
		pos_true = poses[:, :3]
		ori_true = poses[:, 3:]
		
		pos_pred, ori_pred = self(input_image)
		loss, _, _ = self.criterion(pos_pred, ori_pred, pos_true, ori_true)
		
		self.log("Training Loss", loss, on_epoch=True, logger=True)
		return loss
	def validation_step(self, batch, batch_idx):
		input_image, poses  = batch
		pos_true = poses[:, :3]
		ori_true = poses[:, 3:]
		
		pos_pred, ori_pred = self(input_image)

		loss, _, _ = self.criterion(pos_pred, ori_pred, pos_true, ori_true)
		loss_print = self.criterion.loss_print[0]
		loss_pos_print = self.criterion.loss_print[1]
		loss_ori_print = self.criterion.loss_print[2]
		#self.log("Testing Overall Loss", loss, on_epoch=True, logger=True)
		#self.log("Testing Position Loss", loss, on_epoch=True, logger=True)
		#self.log("Testing Orientation Loss", loss, on_epoch=True, logger=True)
		self.log("losses", {"Testing Overall Loss": loss, 
		"Testing Position Loss": loss_pos_print, 
		"Testing Orientation Loss": loss_ori_print})
		return loss
    
class EfficientNetB0PoseNet(pl.LightningModule):
	""" PoseNet using EfficientNet-B0 """
	def __init__(self, pretrained=True, fixed_weight=False, dropout_rate = 0.2):
		super(EfficientNetB0PoseNet, self).__init__()
		self.base_model = EfficientNet.from_name('efficientnet-b0')
		self.base_model.set_swish(memory_efficient=False)

		self.criterion = PoseLoss(self.device)
		self.dropout_rate = dropout_rate


		if fixed_weight:
			for param in self.base_model.parameters():
				param.requires_grad = False

		self.fc_last = nn.Linear(1000, 1000, bias=True)
		
		self.fc_position = nn.Linear(1000, 3, bias=True)
		self.fc_rotation = nn.Linear(1000, 4, bias=True)


	def forward(self, x):
		x = self.base_model(x)
		x = x.view(x.size(0), -1)
		x_fully = self.fc_last(x)
		x = F.relu(x_fully)

		dropout_on = self.training 
		if self.dropout_rate > 0:
			x = F.dropout(x, p=self.dropout_rate, training=dropout_on)
		#print(x.shape)
		position = self.fc_position(x)
		rotation = self.fc_rotation(x)

		return position, rotation
	
	def configure_optimizers(self):
		opt = torch.optim.AdamW(self.parameters(), lr=1e-4)
		return opt
	def training_step(self, batch, batch_idx):
		input_image, poses  = batch
		pos_true = poses[:, :3]
		ori_true = poses[:, 3:]
		
		pos_pred, ori_pred = self(input_image)
		
		loss, _, _ = self.criterion(pos_pred, ori_pred, pos_true, ori_true)
		self.log("Training Loss", loss, on_epoch=True, logger=True)
		return loss
	def validation_step(self, batch, batch_idx):
		input_image, poses  = batch
		pos_true = poses[:, :3]
		ori_true = poses[:, 3:]
		
		pos_pred, ori_pred = self(input_image)
		
		loss, _, _ = self.criterion(pos_pred, ori_pred, pos_true, ori_true)
		loss_print = self.criterion.loss_print[0]
		loss_pos_print = self.criterion.loss_print[1]
		loss_ori_print = self.criterion.loss_print[2]
		#self.log("Testing Overall Loss", loss, on_epoch=True, logger=True)
		#self.log("Testing Position Loss", loss, on_epoch=True, logger=True)
		#self.log("Testing Orientation Loss", loss, on_epoch=True, logger=True)
		self.log("losses", {"Testing Overall Loss": loss, 
		"Testing Position Loss": loss_pos_print, 
		"Testing Orientation Loss": loss_ori_print})
		return loss



class DenseNet121PoseNet(pl.LightningModule):
	""" PoseNet using DenseNet-121 """
	def __init__(self, pretrained=True, fixed_weight=False, dropout_rate = 0.2):
		super(DenseNet121PoseNet, self).__init__()
		
		self.dropout_rate =dropout_rate

		self.base_model = DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,drop_rate=dropout_rate, num_classes= 2048)
		#base_model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=False)

		#self.base_model = nn.Sequential(*list(base_model.children())[:-1]) # does NOT include all layers in DenseNet
		
		self.criterion = PoseLoss(self.device)

		

		#if fixed_weight:
		#	for param in self.base_model.parameters():
		#		param.requires_grad = False

		# Out 2
		self.pos2 = nn.Linear(2048, 3, bias=True) # change from 2048 since more features
		self.ori2 = nn.Linear(2048, 4, bias=True)

	def forward(self, x):
		#print(x.shape)
		x = self.base_model(x) # need stuff after cause sequential does NOT include all of the stuff in model
		#x = F.relu(x, inplace=True) # get rid of relu layer originally found in densenet, it limits to 0 to 1
		#x = F.adaptive_avg_pool2d(x, (1, 1))
		#print(x.shape)
		#x = torch.flatten(x, 1)
		#print(x.shape)
		#print(x.shape)
		#x = x[0]
		#x = x.view(x.size(0), -1)
		#print(x.shape)

		pos = self.pos2(x)
		ori = self.ori2(x)

		return pos, ori
	
	def configure_optimizers(self):
		opt = torch.optim.AdamW(self.parameters(), lr=1e-4)
		return opt
	def training_step(self, batch, batch_idx):
		input_image, poses  = batch
		pos_true = poses[:, :3]
		ori_true = poses[:, 3:]
		
		pos_pred, ori_pred = self(input_image)
		
		loss, _, _ = self.criterion(pos_pred, ori_pred, pos_true, ori_true)
		self.log("Training Loss", loss, on_epoch=True, logger=True)
		return loss
	def validation_step(self, batch, batch_idx):
		input_image, poses  = batch
		pos_true = poses[:, :3]
		ori_true = poses[:, 3:]
		
		pos_pred, ori_pred = self(input_image)
		
		loss, _, _ = self.criterion(pos_pred, ori_pred, pos_true, ori_true)
		loss_print = self.criterion.loss_print[0]
		loss_pos_print = self.criterion.loss_print[1]
		loss_ori_print = self.criterion.loss_print[2]
		#self.log("Testing Overall Loss", loss, on_epoch=True, logger=True)
		#self.log("Testing Position Loss", loss, on_epoch=True, logger=True)
		#self.log("Testing Orientation Loss", loss, on_epoch=True, logger=True)
		self.log("losses", {"Testing Overall Loss": loss, 
		"Testing Position Loss": loss_pos_print, 
		"Testing Orientation Loss": loss_ori_print})
		return loss

class GoogleNetPoseNet(pl.LightningModule):
	""" PoseNet using Inception V3 """
	def __init__(self, pretrained=False, fixed_weight=False, dropout_rate = 0.2):
		super(GoogleNetPoseNet, self).__init__()
		
		self.dropout_rate =dropout_rate

		base_model = models.inception_v3(pretrained=False)

		self.criterion = PoseLoss(self.device)

		model = []
		model.append(base_model.Conv2d_1a_3x3)
		model.append(base_model.Conv2d_2a_3x3)
		model.append(base_model.Conv2d_2b_3x3)
		model.append(nn.MaxPool2d(kernel_size=3, stride=2))
		model.append(base_model.Conv2d_3b_1x1)
		model.append(base_model.Conv2d_4a_3x3)
		model.append(nn.MaxPool2d(kernel_size=3, stride=2))
		model.append(base_model.Mixed_5b)
		model.append(base_model.Mixed_5c)
		model.append(base_model.Mixed_5d)
		model.append(base_model.Mixed_6a)
		model.append(base_model.Mixed_6b)
		model.append(base_model.Mixed_6c)
		model.append(base_model.Mixed_6d)
		model.append(base_model.Mixed_6e)
		model.append(base_model.Mixed_7a)
		model.append(base_model.Mixed_7b)
		model.append(base_model.Mixed_7c)
		self.base_model = nn.Sequential(*model)

		if fixed_weight:
			for param in self.base_model.parameters():
				param.requires_grad = False

		# Out 2
		self.pos2 = nn.Linear(2048, 3, bias=True)
		self.ori2 = nn.Linear(2048, 4, bias=True)

	def forward(self, x):
		# 299 x 299 x 3
		x = self.base_model(x)
		# 8 x 8 x 2048
		x = F.avg_pool2d(x, kernel_size=8)
		# 1 x 1 x 2048
		x = F.dropout(x, p=self.dropout_rate, training=self.training)
		# 1 x 1 x 2048
		x = x.view(x.size(0), -1)
		# 2048
		pos = self.pos2(x)
		ori = self.ori2(x)

		return pos, ori
	
	def configure_optimizers(self):
		opt = torch.optim.AdamW(self.parameters(), lr=1e-4)
		return opt
	def training_step(self, batch, batch_idx):
		input_image, poses  = batch
		pos_true = poses[:, :3]
		ori_true = poses[:, 3:]
		
		pos_pred, ori_pred = self(input_image)
		
		loss, _, _ = self.criterion(pos_pred, ori_pred, pos_true, ori_true)
		self.log("Training Loss", loss, on_epoch=True, logger=True)
		return loss
	def validation_step(self, batch, batch_idx):
		input_image, poses  = batch
		pos_true = poses[:, :3]
		ori_true = poses[:, 3:]
		
		pos_pred, ori_pred = self(input_image)
		
		loss, _, _ = self.criterion(pos_pred, ori_pred, pos_true, ori_true)
		loss_print = self.criterion.loss_print[0]
		loss_pos_print = self.criterion.loss_print[1]
		loss_ori_print = self.criterion.loss_print[2]
		#self.log("Testing Overall Loss", loss, on_epoch=True, logger=True)
		#self.log("Testing Position Loss", loss, on_epoch=True, logger=True)
		#self.log("Testing Orientation Loss", loss, on_epoch=True, logger=True)
		self.log("losses", {"Testing Overall Loss": loss, 
		"Testing Position Loss": loss_pos_print, 
		"Testing Orientation Loss": loss_ori_print})
		return loss


class InceptionResNetV2PoseNet(pl.LightningModule):
	""" PoseNet using InceptionResetV2 """
	def __init__(self, pretrained=False, fixed_weight=False, dropout_rate = 0.2):
		super(InceptionResNetV2PoseNet, self).__init__()
		
		self.dropout_rate =dropout_rate

		self.base_model = Inception_ResNetv2(in_channels=3, classes=2048, k=256, l=256, m=384, n=384)
		

		self.criterion = PoseLoss(self.device)


		if fixed_weight:
			for param in self.base_model.parameters():
				param.requires_grad = False

		# Out 2
		self.pos2 = nn.Linear(2048, 3, bias=True)
		self.ori2 = nn.Linear(2048, 4, bias=True)

	def forward(self, x):
		# 299 x 299 x 3
		x = self.base_model(x)
		#print(x.shape)
		# 1 x 1 x 2048
		x = F.dropout(x, p=self.dropout_rate, training=self.training)
		# 1 x 1 x 2048
		x = x.view(x.size(0), -1)
		# 2048
		pos = self.pos2(x)
		ori = self.ori2(x)

		return pos, ori
	def configure_optimizers(self):
		opt = torch.optim.AdamW(self.parameters(), lr=1e-4)
		return opt
	def training_step(self, batch, batch_idx):
		input_image, poses  = batch
		pos_true = poses[:, :3]
		ori_true = poses[:, 3:]
		
		pos_pred, ori_pred = self(input_image)
		
		loss, _, _ = self.criterion(pos_pred, ori_pred, pos_true, ori_true)
		self.log("Training Loss", loss, on_epoch=True, logger=True)
		return loss
	def validation_step(self, batch, batch_idx):
		input_image, poses  = batch
		pos_true = poses[:, :3]
		ori_true = poses[:, 3:]
		
		pos_pred, ori_pred = self(input_image)
		
		loss, _, _ = self.criterion(pos_pred, ori_pred, pos_true, ori_true)
		loss_print = self.criterion.loss_print[0]
		loss_pos_print = self.criterion.loss_print[1]
		loss_ori_print = self.criterion.loss_print[2]
		#self.log("Testing Overall Loss", loss, on_epoch=True, logger=True)
		#self.log("Testing Position Loss", loss, on_epoch=True, logger=True)
		#self.log("Testing Orientation Loss", loss, on_epoch=True, logger=True)
		self.log("losses", {"Testing Overall Loss": loss, 
		"Testing Position Loss": loss_pos_print, 
		"Testing Orientation Loss": loss_ori_print})
		return loss

''
# this function works to find precise parameters needed, as well as flops!
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table

input_tensor = torch.rand(size=(1,3,299,299))
model = EfficientNetB0PoseNet()
flops = FlopCountAnalysis(model, input_tensor)
print(flop_count_table(flops))

