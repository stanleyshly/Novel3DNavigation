from data_loader import get_loader
import pytorch_lightning as pl
from model.model import GoogleNetPoseNet, ResNet34PoseNet, EfficientNetB0PoseNet, DenseNet121PoseNet, InceptionResNetV2PoseNet
from model.model_transformer import DeiTTinyPoseNet, DeiTSmallPoseNet, DeiTBasePoseNet
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.plugins import DeepSpeedPlugin
from pytorch_lightning.callbacks import LearningRateMonitor


image_path = "./train/"
#model = "Googlenet"
model = "Resnet"
metadata_path = "./train/sequence-train.txt"
mode = "train"
batch_size = 32
shuffle = True
dataloaders = get_loader(model, image_path, metadata_path, mode, batch_size,
                             shuffle)
train_loader = dataloaders['train']
test_loader = dataloaders['val']


lr_monitor = LearningRateMonitor(logging_interval='step')
logger = TensorBoardLogger("TBLogs") # this line if for testing before final run, avoids cluttering up the weights and bias dashboard 
ModelPoseNet = ResNet34PoseNet()

#trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=100) 
trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=1000, precision=16, callbacks=[lr_monitor])


trainer.fit(ModelPoseNet, train_loader, test_loader)
