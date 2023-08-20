import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init,constant_init,kaiming_init
import torch
from ..registry import NECKS
import numpy as np
import torch
from torch import nn
from torch.nn import init
from ..utils.conv_module import ConvModule

from ..utils.bifpnThree import BiFPNThree
# from ..utils.scenet import ScfeaturePyramidNetwork
  
@NECKS.register_module
class YzcBiHighFPNRetinanet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 group=1,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 train_with_auxiliary=False,
                 normalize=None,
                 activation=None):
        super(YzcBiHighFPNRetinanet, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.with_bias = normalize is None
        self.train_with_auxiliary = train_with_auxiliary
        self.extra_convs_on_inputs = extra_convs_on_inputs
        self.relu_before_extra_convs = relu_before_extra_convs
        
        # self.extra_input_conv = ConvModule(
        #         2048,
        #         out_channels,
        #         3,
        #         padding=1,
        #         normalize=normalize,
        #         bias=self.with_bias,
        #         activation=self.activation,
        #         inplace=False
        #     )
        self.Bpn1 = BiFPNThree(num_channels=self.out_channels,conv_channels=[256,256,256])
        self.Bpn2 = BiFPNThree(num_channels=self.out_channels,conv_channels=[256,256,256])
        self.Bpn3 = BiFPNThree(num_channels=self.out_channels,conv_channels=[256,256,256])
        
        
        #--------------验证f_fpnconv3*3作用-------------------------
        self.inoutconv1 = ConvModule(
                2048,
                out_channels,
                3,
                padding=1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)
        
        self.inoutconv2 = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)
        
        self.inoutconv3 = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)
        
        #--------------验证f_fpnconv3*3作用-------------------------
        
        # self.scenet = ScfeaturePyramidNetwork(2048)
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            print("i=",i)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)
            self.fpn_convs.append(fpn_conv)
        # for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    padding=0,
                    normalize=normalize,
                    bias=self.with_bias,
                    activation=self.activation,
                    inplace=False)
            self.lateral_convs.append(l_conv)
        self.adaptive_pool_output_ratio = [0.1, 0.2, 0.3]
        self.high_lateral_conv = nn.ModuleList()
        self.high_lateral_conv.extend(
            [nn.Conv2d(in_channels[-1], out_channels, 1) for k in range(len(self.adaptive_pool_output_ratio))])
        self.high_lateral_conv_attention = nn.Sequential(
            nn.Conv2d(out_channels * (len(self.adaptive_pool_output_ratio)), out_channels, 1), nn.ReLU(),
            nn.Conv2d(out_channels, len(self.adaptive_pool_output_ratio), 3, padding=1))
        # add extra conv layers (e.g., RetinaNet
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                # in_channels = (self.in_channels[self.backbone_end_level - 1]
                #               if i == 0 else out_channels)
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpnconv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    normalize=normalize,
                    bias=self.with_bias,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpnconv)
                

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        for m in self.high_lateral_conv_attention.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        print("AugFPN+3Bifpn no downupsum+RFA houmian+no fpn3*3+config中add_extra_convs==Fasle")
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # laterals = list(inputs)
        
        h, w = inputs[-1].size(2), inputs[-1].size(3)

    
        AdapPool_Features = [self.high_lateral_conv[j](F.adaptive_avg_pool2d(inputs[-1], output_size=(
        max(1, int(h * self.adaptive_pool_output_ratio[j])), max(1, int(w * self.adaptive_pool_output_ratio[j]))))) 
                             for j in range(len(self.adaptive_pool_output_ratio))]
        AdapPool_Features = [F.upsample(feat, size=(h, w), mode='bilinear', align_corners=True) for feat in
                             AdapPool_Features] 

        Concat_AdapPool_Features = torch.cat(AdapPool_Features, dim=1)
        fusion_weights = self.high_lateral_conv_attention(Concat_AdapPool_Features)
        fusion_weights = F.sigmoid(fusion_weights)
        high_pool_fusion = 0
        
        for i in range(3):
            high_pool_fusion += torch.unsqueeze(fusion_weights[:, i, :, :], dim=1) * AdapPool_Features[i]

        raw_laternals = [laterals[i].clone() for i in range(len(laterals))]
        
#         # build top-down path

        # high_pool_fusion += global_pool
        # laterals[-1] += high_pool_fusion
#         print('len(laterals)=',len(laterals))
        #********************************交叉连接***************************
        used_backbone_levels = len(laterals)
        # for i in range(used_backbone_levels - 1, 0, -1):
        #     laterals[i - 1] += F.interpolate(
        #         laterals[i], scale_factor=2, mode='nearest')
        
        af_Bifn_outs_one = self.Bpn1(laterals)
        af_Bifn_outs_two = self.Bpn2(af_Bifn_outs_one)
        af_Bifn_outs_three = self.Bpn3(af_Bifn_outs_two)
        # af_Bifn_outs_foure = self.Bpn4(af_Bifn_outs_three)
        # af_Bifn_outs_five = self.Bpn5(af_Bifn_outs_foure)
#         #***************Bifpn****************************************
        

        m_Bifn_outs = list(af_Bifn_outs_three)
#         print('outs[-1].shape = ',outs[-1].shape)
# #         print('outs[0]=',outs[0].shape)
#         used_backbone_levels = len(laterals)
        
        # fe_out = self.scenet(laterals)
        # raw_laternals = [fe_out[i].clone() for i in range(len(fe_out))]
        # build outputs
        # part 1: from original levels
        # print("len(fe_out)=",len(fe_out))
        m_Bifn_outs[-1] += high_pool_fusion
        
        outs = m_Bifn_outs
        
        # outs = [
        #     self.fpn_convs[i](m_Bifn_outs[i]) for i in range(used_backbone_levels)
        # ]
        
        # outs = list(fe_out)
        print('sucess!')
        # af_Bifn_outs = self.Bpn(outs)
        # outs = list(af_Bifn_outs)
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.inoutconv1(orig))
                    print("self.extra_convs_on_inputs")
                else:
                    outs.append(self.inoutconv2(outs[-1]))
                    print("self.extra_convs_on_inputs")

                pool_noupsample_fusion = F.adaptive_avg_pool2d(high_pool_fusion, (1, 1))
                outs[-1] += pool_noupsample_fusion
                
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.inoutconv3(outs[-1]))
                        
        if self.train_with_auxiliary:
            print("end!win!")
            return tuple(outs), tuple(raw_laternals)
        else:
            return tuple(outs)
