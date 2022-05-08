import argparse
from typing import Optional
import torch
import torch.nn as nn
import a2j.resnet as resnet
from a2j.anchor import A2J_loss, post_process
import pytorch_lightning as pl

from utils.utils import get_e2e_loaders

class DepthRegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=16, num_classes=15, feature_size=256):
        super(DepthRegressionModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(feature_size)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors*num_classes, kernel_size=3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.act4(out)
        out = self.output(out)

        # out is B x C x W x H, with C = 3*num_anchors
        out1 = out.permute(0, 3, 2, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        return out2.contiguous().view(out2.shape[0], -1, self.num_classes)

class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=16, num_classes=15, feature_size=256):
        super(RegressionModel, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(feature_size)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors*num_classes*2, kernel_size=3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.act4(out)
        out = self.output(out)

        # out is B x C x W x H, with C = 3*num_anchors
        out1 = out.permute(0, 3, 2, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes, 2)
        return out2.contiguous().view(out2.shape[0], -1, self.num_classes, 2)

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=16, num_classes=15, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(feature_size)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors*num_classes, kernel_size=3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.act4(out)
        out = self.output(out)
    
        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 3, 2, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class ResNetBackBone(nn.Module):
    def __init__(self):
        super(ResNetBackBone, self).__init__()
        
        modelPreTrain50 = resnet.resnet50(pretrained=True)
        self.model = modelPreTrain50
        
    def forward(self, x): 
        n, c, h, w = x.size()  # x: [B, 1, H ,W]
        
        x = x[:,0:1,:,:]  # depth
        x = x.expand(n,3,h,w)
              
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x1 = self.model.layer1(x)
        x2 = self.model.layer2(x1)
        x3 = self.model.layer3(x2)
        x4 = self.model.layer4(x3)
        
        return x3,x4  

class A2JModel(nn.Module):
    def __init__(self, num_classes, crop_height, crop_width, is_3D=True, spatial_factor=0.5,):
        super(A2JModel, self).__init__()
        self.is_3D = is_3D 
        self.Backbone = ResNetBackBone() # 1 channel depth only, resnet50 
        self.regressionModel = RegressionModel(2048, num_classes=num_classes)
        self.classificationModel = ClassificationModel(1024, num_classes=num_classes)
        if is_3D:
            self.DepthRegressionModel = DepthRegressionModel(2048, num_classes=num_classes)
        self.criterion = A2J_loss(shape=[crop_height//16,crop_width//16],thres = [16.0,32.0],stride=16,\
        spatialFactor=spatial_factor,img_shape=[crop_height, crop_width],P_h=None, P_w=None)
        self.post_process = post_process(shape=[crop_height//16,crop_width//16],stride=16,P_h=None, P_w=None)
        self.reg_loss_factor = 3

    def eager_outputs(self, heads, gt=None):
        output = torch.FloatTensor()
        pred_keypoints = self.post_process(heads, voting=False)
        output = torch.cat([output, pred_keypoints.data.cpu()], 0)

        if gt is not None:
            cls_loss, reg_loss =  self.criterion(heads, gt)
            reg_loss *= self.reg_loss_factor
            losses = {
                "classification": cls_loss,
                "regression": reg_loss,
                "total_loss": cls_loss + reg_loss
            }
            return losses, output
        return output

    
    def forward(self, x, gt=None): 
        x3,x4 = self.Backbone(x)
        classification  = self.classificationModel(x3)
        regression = self.regressionModel(x4)
        if self.is_3D:
            depth_regression  = self.DepthRegressionModel(x4)
            return self.eager_outputs((classification, regression, depth_regression), gt)
        return self.eager_outputs((classification, regression), gt)

class A2JModelLightning(pl.LightningModule):
    def __init__(self, num_classes: int=21, crop_height:int=176, crop_width:int=176, is_3D:bool=True, spatial_factor:float=0.5):

        """A2J Model for joint estimation

        Args:
            num_classes (int): number of classes
            crop_height (int): height of the input image
            crop_width (int): width of the input image
            is_3D (bool): if True, use depth regression branch model
            spatial_factor (float): spatial factor for the loss
        """
        super().__init__()
        self.save_hyperparameters()
        self.a2j = A2JModel(num_classes, crop_height, crop_width, is_3D, spatial_factor)

    def training_step(self, batch, batch_idx):
        im, jt_uvd_gt, dexycb_id, color_im, _, _ = batch
        losses, outputs = self.a2j(im, jt_uvd_gt)
        self.log('train_loss', losses['total_loss'], prog_bar=True)
        return losses['total_loss']

    def validation_step(self, batch, batch_idx):
        im, jt_uvd_gt, dexycb_id, color_im, _, _ = batch
        losses, outputs = self.a2j(im, jt_uvd_gt)
        self.log('val_loss', losses['total_loss'], prog_bar=True)
        return losses['total_loss']

class A2JDataModule(pl.LightningDataModule):
    def __init__(self, batch_size:int=64, workers:int=8, aspect_ratio_group_factor:int=0,):
        super().__init__()

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch_size', type=int, default=batch_size)
        parser.add_argument('--workers', type=int, default=workers)
        parser.add_argument('--aspect_ratio_group_factor', type=int, default=aspect_ratio_group_factor)
        args = parser.parse_args([])

        self.args = args
    
    def setup(self, stage=None):
        data_loader, data_loader_test, data_loader_val = get_e2e_loaders(self.args, a2j=True)
        self.data_loader = data_loader
        self.data_loader_val = data_loader_val
        self.data_loader_test = data_loader_test
    
    def train_dataloader(self):
        return self.data_loader
    
    def val_dataloader(self):
        return self.data_loader_val

    def test_dataloader(self):
        return self.data_loader_test