import argparse
from typing import Optional
from dex_ycb_toolkit.hpe_eval import HPEEvaluator
import torch
import torch.nn as nn
import a2j.resnet as resnet
from a2j.anchor import A2J_loss, post_process
import pytorch_lightning as pl
from datasets3d.a2jdataset import uvd2xyz

from utils.utils import get_e2e_loaders, vis_minibatch
from utils.vistool import VisualUtil
import torchvision.transforms as T
import numpy as np
import os

def convert_joints(jt_uvd_pred, jt_uvd_gt, box, paras, cropWidth, cropHeight, is_left=False, image_width=0, image_height=0):
    jt_uvd_pred = jt_uvd_pred.reshape(-1, 3)
    if jt_uvd_gt is not None:
        jt_uvd_gt = jt_uvd_gt.reshape(-1, 3)
    box = box.reshape(4)
    if paras is not None:
        paras = paras.reshape(4)

    X_min, Y_min, X_max, Y_max = box[0], box[1], box[2], box[3]

    jt_xyz_pred = np.ones_like(jt_uvd_pred)
    jt_xyz_pred[:, 0] = jt_uvd_pred[:, 0] * (X_max - X_min) / cropWidth + X_min
    jt_xyz_pred[:, 1] = jt_uvd_pred[:, 1] * (Y_max - Y_min) / cropHeight + Y_min
    jt_xyz_pred[:, 2] = jt_uvd_pred[:, 2]

    if is_left:
        jt_xyz_pred[0, 0] = abs(image_width - jt_xyz_pred[0, 0])
        # jt_xyz_pred[0, 1] = abs(image_height - jt_xyz_pred[0, 1])

    if paras is not None:
        jt_xyz_pred = uvd2xyz(jt_xyz_pred, paras) * 1000.

    if jt_uvd_gt is not None:
        jt_xyz_gt = np.ones_like(jt_uvd_gt)
        jt_xyz_gt[:, 0] = jt_uvd_gt[:, 0] * (X_max - X_min) / cropWidth + X_min
        jt_xyz_gt[:, 1] = jt_uvd_gt[:, 1] * (Y_max - Y_min) / cropHeight + Y_min
        jt_xyz_gt[:, 2] = jt_uvd_gt[:, 2]
        if paras is not None:
            jt_xyz_gt = uvd2xyz(jt_xyz_gt, paras) * 1000.

        return jt_xyz_pred, jt_xyz_gt
    return jt_xyz_pred
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
    def __init__(self, channel_in):
        super(ResNetBackBone, self).__init__()
        
        modelPreTrain50 = resnet.resnet50(pretrained=True)
        self.model = modelPreTrain50
        self.channel_in = channel_in
        if channel_in == 4:
            self.model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
    def forward(self, x): 
        n, c, h, w = x.size()  # x: [B, 1, H ,W]
        
        x = x[:,0:self.channel_in,:,:]  # depth
        if self.channel_in == 1:
            x = x.expand(n,3,h,w) # only for depth
              
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
    def __init__(self, num_classes, crop_height, crop_width, is_3D=True, is_RGBD=False, spatial_factor=0.5,):
        super(A2JModel, self).__init__()
        self.is_3D = is_3D 
        self.Backbone = ResNetBackBone(channel_in=4 if is_RGBD else 1) # 1 channel depth only, resnet50 
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
    def __init__(
        self, 
        num_classes: int=21, 
        crop_height:int=176, 
        crop_width:int=176, 
        is_3D:bool=True, 
        is_RGBD:bool=False, 
        spatial_factor:float=0.5,
        display_freq:int=5000,
        output_dir:str='models/a2j',
    ):

        """A2J Model for joint estimation

        Args:
            num_classes (int): number of classes
            crop_height (int): height of the input image
            crop_width (int): width of the input image
            is_3D (bool): if True, use depth regression branch model
            spatial_factor (float): spatial factor for the loss
            is_RGBD (bool): if True, use RGBD model
        """
        super().__init__()
        self.save_hyperparameters()
        self.a2j = A2JModel(num_classes, crop_height, crop_width, is_3D, is_RGBD, spatial_factor)
        self.a2j.load_state_dict(torch.load('models/a2j_dexycb_1/a2j_25.pth', map_location="cpu")['model'], strict=False)
        self.rgbd = is_RGBD
        self.display_freq = display_freq
        self.vistool = VisualUtil('dexycb')
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir,'a2j_test_metrics/'), exist_ok=True)
        os.makedirs(os.path.join(output_dir,'dexycb_metrics'), exist_ok=True)

    def forward(self, x, gt=None):
        return self.a2j(x, gt)

    def training_step(self, batch, batch_idx):
        im, jt_uvd_gt, dexycb_id, color_im, _, _, combined_im = batch
        if self.rgbd:
            losses, outputs = self.a2j(combined_im, jt_uvd_gt)
        else:
            losses, outputs = self.a2j(im, jt_uvd_gt)
        self.log('train_loss', losses['total_loss'])
        if (batch_idx % self.display_freq == 0):
            img = vis_minibatch(
                np.array([ np.array(T.ToPILImage()(i)) for i in color_im ])[:, :, :, ::-1],
                im.detach().cpu().numpy(),
                jt_uvd_gt.detach().cpu().numpy(),
                self.vistool,
                dexycb_id.detach().cpu().numpy(),
                path=None,
                jt_pred=outputs.numpy()
            )
            self.logger.log_image(key="samples", images=[img])
        return losses['total_loss']

    def validation_step(self, batch, batch_idx):
        im, jt_uvd_gt, dexycb_id, color_im, _, _, combined_im = batch
        if self.rgbd:
            losses, outputs = self.a2j(combined_im, jt_uvd_gt)
        else:
            losses, outputs = self.a2j(im, jt_uvd_gt)
        self.log('val_loss', losses['total_loss'])

        # calculate rmse per batch
        rmse = torch.sqrt(torch.mean(torch.square(jt_uvd_gt.cpu() - outputs)))
        self.log('test_rmse', rmse.item())

        if (batch_idx % self.display_freq == 0):
            img = vis_minibatch(
                np.array([ np.array(T.ToPILImage()(i)) for i in color_im ])[:, :, :, ::-1],
                im.detach().cpu().numpy(),
                jt_uvd_gt.detach().cpu().numpy(),
                self.vistool,
                dexycb_id.detach().cpu().numpy(),
                path=None,
                jt_pred=outputs.numpy()
            )
            self.logger.log_image(key="samples", images=[img])
        return losses['total_loss']

    def test_step(self, batch, batch_idx):
        im, jt_uvd_gt, dexycb_id, color_im, box, paras, combined_im = batch
        if self.rgbd:
            outputs = self.a2j(combined_im)
        else:
            outputs = self.a2j(im)

        jt_xyz_pred, jt_xyz_gt = convert_joints(
            outputs.numpy(), 
            jt_uvd_gt.cpu().numpy(), 
            box.cpu().numpy(), 
            paras.cpu().numpy(), 
            176,
            176
        )

        # calculate rmse per batch
        rmse = np.sqrt(np.mean(np.square(jt_xyz_gt - jt_xyz_pred)))
        self.log('test_rmse', rmse)

        j_text = ''
        for j in jt_xyz_pred:
            j_text += str(list(j)).strip()[1:-1] + ','
        j_text = j_text.replace(" ", "")[:-1]

        epoch_output = os.path.join(self.output_dir, f'a2j_test_metrics/s0_test_{self.current_epoch}.txt')
        
        with open(epoch_output, 'a') as output:
            print(str(dexycb_id[0].cpu().numpy())[1:-1] + ',' + j_text, file=output)

    def test_epoch_end(self, outputs):
        hpe_eval = HPEEvaluator('s0_test')
        hpe_eval.evaluate(self.current_epoch, os.path.join(self.output_dir, f'a2j_test_metrics/s0_test_{self.current_epoch}.txt'), os.path.join(self.output_dir, 'dexycb_metrics/'))

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