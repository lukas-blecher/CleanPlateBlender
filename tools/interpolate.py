#!/usr/bin/env python
# modified from: https://github.com/sniklaus/softmax-splatting
import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from tools import softsplat
import numpy as np
import cv2
import torch


class Interpolater:
    def __init__(self, args):
        self.args = args
        self.img_root = args.img_root
        self.mask_root = args.mask_root
        self.flow_root = args.flow_root
        self.inter_root = args.inter_root
        os.makedirs(self.inter_root, exist_ok=True)
        self.ending = args.imgending
        self.steps = args.interpolation_steps
        self.device = args.device
        self.flows = os.listdir(self.flow_root)
        self.img_shape = args.img_shape

        self.frame_name_list = sorted(os.listdir(self.img_root))
        self.frames_num = len(self.frame_name_list)

        self.index = 0

    def __len__(self):
        return self.frames_num-1

    def step(self):
        self.interpolate()
        self.move_idx(self.index)
        self.index += 1
        if self.index > self.frames_num:
            self.move_idx(self.index)
            return True
        return False

    def move_idx(self, idx):
        os.replace(self.frame_name_list[idx], '%s/%08.3f.%s' % (self.inter_root, idx, self.ending))
        # create empty mask
        cv2.imwrite('%s/%08.3f.%s' % (self.mask_root, self.index, self.ending), np.zeros(self.img_shape, dtype=np.uint8))

    def read_flo(self, strFile):
        with open(strFile, 'rb') as objFile:
            strFlow = objFile.read()
        # end

        assert(np.frombuffer(strFlow, dtype=np.float32, count=1, offset=0) == 202021.25)

        intWidth = np.frombuffer(strFlow, dtype=np.int32, count=1, offset=4)[0]
        intHeight = np.frombuffer(strFlow, dtype=np.int32, count=1, offset=8)[0]

        return np.frombuffer(strFlow, dtype=np.float32, count=intHeight * intWidth * 2, offset=12).reshape([intHeight, intWidth, 2])

    backwarp_tenGrid = {}

    def backwarp(self, tenInput, tenFlow):
        if str(tenFlow.shape) not in self.backwarp_tenGrid:
            tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
            tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

            self.backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1).to(self.device)

        tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

        return torch.nn.functional.grid_sample(input=tenInput, grid=(self.backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)

    def interpolate(self):
        tenFirst = torch.FloatTensor(np.ascontiguousarray(cv2.imread(filename=self.frame_name_list[self.index],
                                                                     flags=-1).transpose(2, 0, 1)[None, :, :, :].astype(np.float32) * (1.0 / 255.0))).to(self.device)
        tenSecond = torch.FloatTensor(np.ascontiguousarray(cv2.imread(filename=self.frame_name_list[self.index+1],
                                                                      flags=-1).transpose(2, 0, 1)[None, :, :, :].astype(np.float32) * (1.0 / 255.0))).to(self.device)
        tenFlow = torch.FloatTensor(np.ascontiguousarray(self.read_flo(self.flows[self.index]).transpose(2, 0, 1)[None, :, :, :])).to(self.device)

        tenMetric = torch.nn.functional.l1_loss(input=tenFirst, target=self.backwarp(tenInput=tenSecond, tenFlow=tenFlow), reduction='none').mean(1, True)

        for intTime, fltTime in enumerate(np.linspace(0, 1, self.steps+1, endpoint=False).tolist()[1:]):
            # -20.0 is a hyperparameter, called 'beta' in the paper, that could be learned using a torch.Parameter
            tenSoftmax = softsplat.FunctionSoftsplat(tenInput=tenFirst, tenFlow=tenFlow * fltTime, tenMetric=-20.0 * tenMetric, strType='softmax')
            img = tenSoftmax[0, :, :, :].cpu().numpy().transpose(1, 2, 0)
            cv2.imwrite('%s/%08.3f.%s' % (self.inter_root, fltTime+self.index, self.ending), (255*img).astype(np.uint8))
            cv2.imwrite('%s/%08.3f.%s' % (self.mask_root, fltTime+self.index, self.ending), (255*(img.sum(-1) == 0.)).astype(np.uint8))
