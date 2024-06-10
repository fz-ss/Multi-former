import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from depth_distribution.main.utils.misc import resize
# from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM


# @HEADS.register_module()
class UPerHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(UPerHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        # self.bottleneck = nn.Sequential(
        #     nn.Conv2d(self.in_channels[-1] + len(pool_scales) * self.channels,self.channels,3,padding=1),
        #     nn.BatchNorm2d(self.channels),
        #     nn.ReLU()
        # )
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            # l_conv = nn.Sequential(
            #     nn.Conv2d(in_channels, self.channels, 1, stride=1),
            #     nn.BatchNorm2d(self.channels),
            #     nn.ReLU()
            # )
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            # fpn_conv = nn.Sequential(
            #     nn.Conv2d(self.channels,self.channels,3,padding=1),
            #     nn.BatchNorm2d(self.channels),
            #     nn.ReLU()
            # )
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        # self.fpn_bottleneck=nn.Sequential(
        #     nn.Conv2d(len(self.in_channels) * self.channels,self.channels,3,padding=1),
        #     nn.BatchNorm2d(self.channels),
        #     nn.ReLU()
        # )
        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        #depth model
        # self.enc5_1 = nn.ModuleList()
        # self.enc5_2 = nn.ModuleList()
        # self.enc5_3 = nn.ModuleList()
        # for in_channels in self.in_channels:  # skip the top layer
        #     # l_conv = nn.Sequential(
        #     #     nn.Conv2d(in_channels, self.channels, 1, stride=1),
        #     #     nn.BatchNorm2d(self.channels),
        #     #     nn.ReLU()
        #     # )
        #     conv5_1 = ConvModule(
        #         in_channels,
        #         384,
        #         1,
        #         conv_cfg=self.conv_cfg,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=self.act_cfg,
        #         inplace=False)
        #     # fpn_conv = nn.Sequential(
        #     #     nn.Conv2d(self.channels,self.channels,3,padding=1),
        #     #     nn.BatchNorm2d(self.channels),
        #     #     nn.ReLU()
        #     # )
        #     conv5_2 = ConvModule(
        #         384,
        #         128,
        #         3,
        #         padding=1,
        #         conv_cfg=self.conv_cfg,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=self.act_cfg,
        #         inplace=False)
        #     conv5_3 = ConvModule(
        #         128,
        #         64,
        #         1,
        #         conv_cfg=self.conv_cfg,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=self.act_cfg,
        #         inplace=False)
        #     self.enc5_1.append(conv5_1)
        #     self.enc5_2.append(conv5_2)
        #     self.enc5_3.append(conv5_3)
        # self.catdepth = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=True)
        # self.catmixdepth = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.enc5_1 = nn.Conv2d(sum(self.in_channels), 512, kernel_size=1, stride=1, padding=0, bias=True)
        self.enc5_2 = nn.Conv2d(512, 384, kernel_size=3, stride=1, padding=1, bias=True)
        self.enc5_3 = nn.Conv2d(384, 128, kernel_size=1, stride=1, padding=0, bias=True)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]

        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        """Forward function."""
        

        inputs = self._transform_inputs(inputs)
        interp = nn.Upsample( size=(inputs[0].shape[-2], inputs[0].shape[-1]),mode="bilinear",align_corners=True)
        interp_mix = nn.Upsample(size =(inputs[-2].shape[-2], inputs[-2].shape[-1]),mode="bilinear",align_corners=True)
        #depth
        # out_depth = []
        # mix_depth = []
        # for i in range(1):
        #     inputs[i] = interp(inputs[i])
        #     out_depth.append(self.enc5_1[i](inputs[i]))
        #     mix_depth.append(out_depth[i])
        #     mix_depth[i] = interp_mix(mix_depth[i])
        #     out_depth[i] = self.relu(out_depth[i])
        #     out_depth[i] = self.enc5_2[i](out_depth[i])
        #     out_depth[i] = self.relu(out_depth[i])
        #     out_depth[i] = self.enc5_3[i](out_depth[i])
        #     out_depth[i] = torch.mean(out_depth[i],dim=1,keepdim=True)
            

        # out_depth = torch.stack(out_depth)
        # out_depth = out_depth.squeeze(1)
        # out_depth = out_depth.transpose(0,1)
        # # out_depth = torch.from_numpy(np.array(out_depth))
        # mix_depth = torch.stack(mix_depth)
        # mix_depth = mix_depth.squeeze(1)
        # mix_depth = mix_depth.transpose(0,1)
        # depth = self.catdepth(out_depth)
        # mix_depth = self.catmixdepth(mix_depth)
        input_depth = []
        for i in range(4):
            input_depth.append(interp(inputs[i]))
        input_depth  = torch.cat(input_depth, dim=1)

        x5_enc = self.enc5_1(input_depth)
        x5_enc = self.relu(x5_enc)
        x5_enc = self.enc5_2(x5_enc)   #1,512,41,81
        x5_enc = self.relu(x5_enc)   #1,512,41,81
        mix_depth = x5_enc
        mix_depth = interp_mix(mix_depth)
        x5_enc = self.enc5_3(x5_enc)   #1,128,41,81
        depth = torch.mean(x5_enc, dim=1, keepdim=True)  # depth output  1,1,41,81


        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1) #1,2048,80,160
        output = self.fpn_bottleneck(fpn_outs) #1,512,80,160;one conv
        output = self.cls_seg(output) #1,7,80,160;one conv
        return output,depth,mix_depth   #深度输出的上一层

