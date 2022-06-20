import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, datasets
import torch.nn.functional as F
import pytorch_lightning as pl
from warmup_scheduler import GradualWarmupScheduler
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from vision_transformer import _create_vision_transformer
import timm
import copy

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
		


class DeiTTinyPoseNet(pl.LightningModule):
	""" PoseNet using DeiT Tiny """
	def __init__(self, pretrained=False, fixed_weight=False, dropout_rate = 0.1, training=False):
		super(DeiTTinyPoseNet, self).__init__()
		self.lr_warmup_epochs = 20
		self.dropout_rate =dropout_rate
		self.learning_rate = 1e-4
		model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
		self.base_model = _create_vision_transformer('deit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)#torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=False)		

		self.criterion = PoseLoss(self.device)
		# Out 2
		self.pos2 = nn.Linear(1000, 3, bias=True)
		self.ori2 = nn.Linear(1000, 4, bias=True)

	def forward(self, x):
		x = self.base_model(x)
		x = F.dropout(x, p=self.dropout_rate, training=self.training)
		
		x = x.view(x.size(0), -1)
		# 2048
		pos = self.pos2(x)
		ori = self.ori2(x)

		return pos, ori

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
		# only works since did not override default warmup step size, which is on epoch end
		scheduler  = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps=100,
                                          cycle_mult=2.0,
                                          max_lr=self.learning_rate,
                                          min_lr=2e-5,
                                          warmup_steps=self.lr_warmup_epochs,
                                          gamma=0.5)# steps is really epochs!
		

		return [optimizer], [scheduler]
	
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
class DeiTSmallPoseNet(pl.LightningModule):
	""" PoseNet using DeiT Small """
	def __init__(self, pretrained=False, fixed_weight=False, dropout_rate = 0.1):
		super(DeiTSmallPoseNet, self).__init__()
		self.lr_warmup_steps = 20000
		self.dropout_rate =dropout_rate
		self.learning_rate = 3e-5

		self.base_model = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True)								

		self.criterion = PoseLoss(self.device)
		# Out 2
		self.pos2 = nn.Linear(1000, 3, bias=True)
		self.ori2 = nn.Linear(1000, 4, bias=True)

	def forward(self, x):
		x = self.base_model(x)
		x = F.dropout(x, p=self.dropout_rate, training=self.training)
		x = x.view(x.size(0), -1)
		# 2048
		pos = self.pos2(x)
		ori = self.ori2(x)

		return pos, ori

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

		return optimizer
	
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

class DeiTBasePoseNet(pl.LightningModule):
	""" PoseNet using DeiT Base """
	def __init__(self, pretrained=False, fixed_weight=False, dropout_rate = 0.1):
		super(DeiTBasePoseNet, self).__init__()
		self.lr_warmup_steps = 20000
		self.dropout_rate =dropout_rate
		self.learning_rate = 3e-5

		self.base_model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)								

		self.criterion = PoseLoss(self.device)
		# Out 2
		self.pos2 = nn.Linear(1000, 3, bias=True)
		self.ori2 = nn.Linear(1000, 4, bias=True)

	def forward(self, x):
		x = self.base_model(x)
		x = F.dropout(x, p=self.dropout_rate, training=self.training)
		x = x.view(x.size(0), -1)
		# 2048
		pos = self.pos2(x)
		ori = self.ori2(x)

		return pos, ori

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
		return optimizer
	
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


# this function works to find precise parameters needed, as well as flops!

from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table

input_tensor = torch.rand(size=(1,3,224,224))
model = DeiTTinyPoseNet()
flops = FlopCountAnalysis(model, input_tensor)
print(flop_count_table(flops))

