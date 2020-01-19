from tools.frame_inpaint import DeepFillv1
from utils import flow as flo
import numpy as np
import cv2
import torch
import sys
import argparse
import os
import time
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

#from mmcv import ProgressBar


class modal_propagation:
    def __init__(self, args, frame_inpaint_model=None):
        # Setup dataset
        self.args = args
        self.img_root = args.img_root
        self.mask_root = args.mask_root
        self.flow_root = args.flow_root
        self.output_root = args.output_root_propagation
        self.frame_inpaint_model = frame_inpaint_model
        # the shape list may be changed in the below, pls check it
        self.img_shape = args.img_shape
        self.th_warp = args.th_warp
        self.video_list = os.listdir(self.flow_root)
        self.video_list.sort()
        self.st_time = time.time()
        self.flow_no_list = [int(x[:5]) for x in os.listdir(self.flow_root) if '.flo' in x]
        self.flow_start_no = min(self.flow_no_list)
        # print('Flow Start no', flow_start_no)
        if not os.path.exists(self.output_root):
            os.makedirs(self.output_root)

        self.frame_name_list = sorted(os.listdir(self.img_root))
        self.frames_num = len(self.frame_name_list)
        self.frame_inpaint_seq = np.ones(self.frames_num-1)
        self.masked_frame_num = np.sum((self.frame_inpaint_seq > 0).astype(np.int))
        print(self.masked_frame_num, 'frames need to be inpainted.')

        self.image = cv2.imread(os.path.join(self.img_root, self.frame_name_list[0]))
        if self.img_shape[0] < 1:
            self.shape = self.image.shape
        else:
            self.shape = self.img_shape
        print('The output shape is:', self.shape)

        self.image = cv2.resize(self.image, (self.shape[1], self.shape[0]))
        self.iter_num = 0
        self.change_iter_num = False
        self.result_pool = [
            np.zeros(self.image.shape, dtype=self.image.dtype)
            for _ in range(self.frames_num)
        ]
        self.label_pool = [
            np.zeros(self.image.shape, dtype=self.image.dtype)
            for _ in range(self.frames_num)
        ]
        self.complete = False
        self.state = 0
        self.index = [0, 0, 0]
        self.calls = [self.forward, self.backward, self.merge]
        self.file_ending='png'

    def __len__(self):
        return self.frames_num*3-3

    def step(self):
        i = self.index[self.state]
        if i == 0:
            self.start_round()
        self.calls[self.state](i)
        self.index[self.state] += 1
        if self.index[self.state] == self.frames_num - 1:
            self.state += 1
        if self.state == 3:
            self.end_round()
        return self.complete

    def start_round(self):
        if self.state == 0:
            self.results = [
                np.zeros(self.image.shape + (2,), dtype=self.image.dtype)
                for _ in range(self.frames_num)
            ]
            self.time_stamp = [
                -np.ones(self.image.shape[:2] + (2,), dtype=int)
                for _ in range(self.frames_num)
            ]

            print('Iter', self.iter_num, 'Forward Propagation')
            # forward
            if self.iter_num == 0:
                self.image = cv2.imread(os.path.join(self.img_root, self.frame_name_list[0]))
                self.image = cv2.resize(self.image, (self.shape[1], self.shape[0]))

                self.label = cv2.imread(
                    os.path.join(self.mask_root, '%05d.png' % (0 + self.flow_start_no)), cv2.IMREAD_COLOR)
                self.label = cv2.resize(self.label, (self.image.shape[1], self.image.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                self.image = self.result_pool[0]
                self.label = self.label_pool[0]

            if len(self.label.shape) == 3:
                self.label = self.label[:, :, 0]
            if self.args.enlarge_mask and self.iter_num == 0:
                kernel = np.ones((self.args.enlarge_kernel, self.args.enlarge_kernel), np.uint8)
                self.label = cv2.dilate(self.label, kernel, iterations=1)

            self.label = (self.label > 0).astype(np.uint8)
            self.image[self.label > 0, :] = 0

            self.results[0][..., 0] = self.image
            self.time_stamp[0][self.label == 0, 0] = 0
        elif self.state == 1:

            sys.stdout.write('\n')
            print('Iter', self.iter_num, 'Backward Propagation')
            # backward
            if self.iter_num == 0:
                self.image = cv2.imread(os.path.join(self.img_root, self.frame_name_list[self.frames_num - 1]))
                self.image = cv2.resize(self.image, (self.shape[1], self.shape[0]))

                self.label = cv2.imread(
                    os.path.join(self.mask_root, '%05d.png' % (self.frames_num - 1 + self.flow_start_no)),
                    cv2.IMREAD_COLOR)

                self.label = cv2.resize(self.label, (self.image.shape[1], self.image.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                self.image = self.result_pool[-1]
                self.label = self.label_pool[-1]

            if len(self.label.shape) == 3:
                self.label = self.label[:, :, 0]
            if self.args.enlarge_mask and self.iter_num == 0:
                kernel = np.ones((self.args.enlarge_kernel, self.args.enlarge_kernel), np.uint8)
                self.label = cv2.dilate(self.label, kernel, iterations=1)

            self.label = (self.label > 0).astype(np.uint8)
            self.image[(self.label > 0), :] = 0

            self.results[self.frames_num - 1][..., 1] = self.image
            self.time_stamp[self.frames_num - 1][self.label == 0, 1] = self.frames_num - 1
        elif self.state == 2:
            sys.stdout.write('\n')
            self.tmp_label_seq = np.zeros(self.frames_num-1)
            print('Iter', self.iter_num, 'Merge Results')

    def forward(self, th):
        th = th+1
        if self.iter_num == 0:
            self.image = cv2.imread(os.path.join(self.img_root, self.frame_name_list[th]))
            self.image = cv2.resize(self.image, (self.shape[1], self.shape[0]))
        else:
            self.image = self.result_pool[th]

        flow1 = flo.readFlow(os.path.join(self.flow_root, '%05d.flo' % (th - 1 + self.flow_start_no)))
        flow2 = flo.readFlow(os.path.join(self.flow_root, '%05d.rflo' % (th + self.flow_start_no)))
        flow1 = flo.flow_tf(flow1, self.image.shape)
        flow2 = flo.flow_tf(flow2, self.image.shape)

        if self.iter_num == 0:
            self.label = cv2.imread(
                os.path.join(self.mask_root, '%05d.png' % (th + self.flow_start_no)), cv2.IMREAD_COLOR)
            self.label = cv2.resize(self.label, (self.image.shape[1], self.image.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            self.label = self.label_pool[th]

        if len(self.label.shape) == 3:
            self.label = self.label[:, :, 0]

        if self.args.enlarge_mask and self.iter_num == 0:
            kernel = np.ones((self.args.enlarge_kernel, self.args.enlarge_kernel), np.uint8)
            self.label = cv2.dilate(self.label, kernel, iterations=1)

        self.label = (self.label > 0).astype(np.uint8)
        self.image[(self.label > 0), :] = 0

        temp1 = flo.get_warp_label(flow1, flow2,
                                   self.results[th - 1][..., 0],
                                   th=self.th_warp)
        temp2 = flo.get_warp_label(flow1, flow2,
                                   self.time_stamp[th - 1],
                                   th=self.th_warp,
                                   value=-1)[..., 0]

        self.results[th][..., 0] = temp1
        self.time_stamp[th][..., 0] = temp2

        self.results[th][self.label == 0, :, 0] = self.image[self.label == 0, :]
        self.time_stamp[th][self.label == 0, 0] = th

    def backward(self, th):
        # for th in tqdm(range(self.frames_num - 2, -1, -1)):
        th = self.frames_num - 2 - th
        if self.iter_num == 0:
            self.image = cv2.imread(os.path.join(self.img_root, self.frame_name_list[th]))
            self.image = cv2.resize(self.image, (self.shape[1], self.shape[0]))
            self.label = cv2.imread(
                os.path.join(self.mask_root, '%05d.png' % (th + self.flow_start_no)), cv2.IMREAD_COLOR)
            self.label = cv2.resize(self.label, (self.image.shape[1], self.image.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            self.image = self.result_pool[th]
            self.label = self.label_pool[th]

        flow1 = flo.readFlow(os.path.join(self.flow_root, '%05d.rflo' % (th + 1 + self.flow_start_no)))
        flow2 = flo.readFlow(os.path.join(self.flow_root, '%05d.flo' % (th + self.flow_start_no)))
        flow1 = flo.flow_tf(flow1, self.image.shape)
        flow2 = flo.flow_tf(flow2, self.image.shape)

        if len(self.label.shape) == 3:
            self.label = self.label[:, :, 0]
        if self.args.enlarge_mask and self.iter_num == 0:
            kernel = np.ones((self.args.enlarge_kernel, self.args.enlarge_kernel), np.uint8)
            self.label = cv2.dilate(self.label, kernel, iterations=1)

        self.la = (self.label > 0).astype(np.uint8)
        self.image[(self.label > 0), :] = 0

        temp1 = flo.get_warp_label(flow1, flow2,
                                   self.results[th + 1][..., 1],
                                   th=self.th_warp)
        temp2 = flo.get_warp_label(
            flow1, flow2, self.time_stamp[th + 1],
            value=-1,
            th=self.th_warp,
        )[..., 1]

        self.results[th][..., 1] = temp1
        self.time_stamp[th][..., 1] = temp2

        self.results[th][self.label == 0, :, 1] = self.image[self.label == 0, :]
        self.time_stamp[th][self.label == 0, 1] = th

    def merge(self, th):
        v1 = (self.time_stamp[th][..., 0] == -1)
        v2 = (self.time_stamp[th][..., 1] == -1)

        hole_v = (v1 & v2)

        result = self.results[th][..., 0].copy()
        result[v1, :] = self.results[th][v1, :, 1].copy()

        v3 = ((v1 == 0) & (v2 == 0))

        dist = self.time_stamp[th][..., 1] - self.time_stamp[th][..., 0]
        dist[dist < 1] = 1

        w2 = (th - self.time_stamp[th][..., 0]) / dist
        w2 = (w2 > 0.5).astype(np.float)

        result[v3, :] = (self.results[th][..., 1] * w2[..., np.newaxis] +
                         self.results[th][..., 0] * (1 - w2)[..., np.newaxis])[v3, :]

        self.result_pool[th] = result.copy()

        tmp_mask = np.zeros_like(result)
        tmp_mask[hole_v, :] = 255
        self.label_pool[th] = tmp_mask.copy()
        self.tmp_label_seq[th] = np.sum(tmp_mask)

    def end_round(self):
        sys.stdout.write('\n')
        self.frame_inpaint_seq[self.tmp_label_seq == 0] = 0
        masked_frame_num = np.sum((self.frame_inpaint_seq > 0).astype(np.int))
        print(masked_frame_num)
        self.iter_num += 1
        self.state = 0
        self.index = [0, 0, 0]
        self.change_iter_num = True
        if masked_frame_num > 0:
            key_frame_ids = get_key_ids(self.frame_inpaint_seq)
            print(key_frame_ids)
            for id in key_frame_ids:
                with torch.no_grad():
                    tmp_inpaint_res = self.frame_inpaint_model.forward(self.result_pool[id], self.label_pool[id])
                self.label_pool[id] = self.label_pool[id] * 0.
                self.result_pool[id] = tmp_inpaint_res
        else:
            print(self.frames_num, 'frames have been inpainted by', self.iter_num, 'iterations.')
            self.complete = True

        tmp_label_seq = np.zeros(self.frames_num - 1)
        for th in range(0, self.frames_num - 1):
            tmp_label_seq[th] = np.sum(self.label_pool[th])
        self.frame_inpaint_seq[tmp_label_seq == 0] = 0
        masked_frame_num = np.sum((self.frame_inpaint_seq > 0).astype(np.int))
        print(masked_frame_num)

        if self.complete:
            print('Writing frames to:', self.output_root)

            if not os.path.exists(self.output_root):
                os.makedirs(self.output_root)

            for th in range(0, self.frames_num-1):
                cv2.imwrite(os.path.join(self.output_root, '%05d.%s' % (th + self.flow_start_no, self.file_ending)),
                            self.result_pool[th].astype(np.uint8))

            print('Propagation has been finished')
            pro_time = time.time() - self.st_time
            print(pro_time)


def get_key_ids(seq):
    st_pointer = 0
    end_pointer = len(seq) - 1

    st_status = False
    end_status = False
    key_id_list = []

    for i in range((len(seq)+1) // 2):
        if st_pointer > end_pointer:
            break

        if not st_status and seq[st_pointer] > 0:
            key_id_list.append(st_pointer)
            st_status = not st_status
        elif st_status and seq[st_pointer] <= 0:
            key_id_list.append(st_pointer-1)
            st_status = not st_status

        if not end_status and seq[end_pointer] > 0:
            key_id_list.append(end_pointer)
            end_status = not end_status
        elif end_status and seq[end_pointer] <= 0:
            key_id_list.append(end_pointer+1)
            end_status = not end_status

        st_pointer += 1
        end_pointer -= 1

    return sorted(list(set(key_id_list)))
