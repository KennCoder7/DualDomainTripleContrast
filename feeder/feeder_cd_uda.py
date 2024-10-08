from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
import numpy as np
import pdb
import os, pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm


try:
    from tools import (random_choose, f_random_rot, valid_crop_resize, f_random_shear, rotate_to_vertical,
                       rotate_to_face_front, rotate_around_y, flip_skeleton_ntu, normalize_skeleton_to_fixed_distance,
                       rotate_to_vertical_per_frame, rotate_to_face_front_per_frame,
                       f_random_shift, f_random_move, f_random_scale, f_temporal_crop, f_padding_none,
                       random_time_flip, random_spatial_flip, gaus_noise, gaus_filter, axis_mask)
except:
    from .tools import (random_choose, f_random_rot, valid_crop_resize, f_random_shear, rotate_to_vertical,
                        rotate_to_face_front, rotate_around_y, flip_skeleton_ntu, normalize_skeleton_to_fixed_distance,
                        rotate_to_vertical_per_frame, rotate_to_face_front_per_frame,
                        f_random_shift, f_random_move, f_random_scale, f_temporal_crop, f_padding_none,
                        random_time_flip, random_spatial_flip, gaus_noise, gaus_filter, axis_mask)


class Feeder(Dataset):
    def __init__(self, dataset, data_path, target_dataset=None, target_data_path=None,
                 label_path=None, p_interval=1, split='train', random_choose=False,
                 random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=-1, debug=False, use_mmap=False,
                 bone=False, vel=False, fold=1, unpaired_cross_dataset_path=None, resize_crop=True, random_crop=False,
                 xy=False, coordinate_transfer_type=-1, random_shear=False, temporal_crop=False,
                 vertical=False, fixed_direction=False, fixed_rot=None, random_scale=False, padding_none=False,
                 fixed_distance=False, get_pairs=False, strong_aug_method=None, preprocess=True, align_st=False,
                 ada=False):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """

        self.std_map = None
        self.mean_map = None
        self.dataset = dataset
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.random_shear = random_shear
        self.bone = bone
        self.vel = vel
        self.fold = fold
        self.unpaired_cross_dataset_path = unpaired_cross_dataset_path
        self.resize_crop = resize_crop
        self.random_crop = random_crop
        self.xy = xy
        self.coordinate_transfer_type = coordinate_transfer_type
        self.target_data_path = target_data_path
        self.target_dataset = target_dataset
        self.vertical = vertical
        self.fixed_direction = fixed_direction
        self.fixed_rot = fixed_rot
        self.skeleton_info = {}
        self.random_scale = random_scale
        self.temporal_crop = temporal_crop
        self.padding_none = padding_none
        self.fixed_distance = fixed_distance
        self.get_pairs = get_pairs
        self.strong_aug_method = strong_aug_method
        self.preprocess = preprocess
        self.align_st = align_st
        self.ada = ada

        if self.preprocess:
            file_name = self.data_path.split('/')[-1]
            preprocessed_file_name = "preprocessed_" + file_name
            preprocessed_file_path = self.data_path.replace(file_name, preprocessed_file_name)
            if os.path.exists(preprocessed_file_path):
                self.data_path = preprocessed_file_path
                self.preprocess = False
                print(f'load preprocessed data from {preprocessed_file_path}')

        if dataset == 'ntu':
            self.data, self.label = self.load_data_ntu()
        elif dataset == 'ntu2posetics3d':
            self.data, self.label = self.load_data_ntu(transform='posetics3d')
        elif dataset == 'ntu2kinetics':
            self.data, self.label = self.load_data_ntu(transform='kinetics')
        elif dataset == 'ntu2sbu':
            self.data, self.label = self.load_data_ntu(transform='sbu')
            # self.pairs = ntu_pairs
        elif dataset == 'sbu':
            self.data, self.label = self.load_data_sbu()
            # self.pairs = sbu_pairs
        elif dataset == 'kinetics':
            self.data, self.label = self.load_data_kinetics()
            # self.pairs = kinetics_pairs
        elif dataset == 'posetics2d' or dataset == 'posetics3d':
            self.data, self.label = self.load_data_posetics(axis='2d' if dataset == 'posetics2d' else '3d')
        elif dataset == 'posetics':
            self.data, self.label = self.load_data_posetics(axis='5d')
        elif dataset == 'posetics2dto3d':
            self.data, self.label = self.load_data_posetics(axis='2dto3d')
        elif dataset == 'pkummd':
            self.data, self.label = self.load_data_pkummd()
        elif dataset == 'orgbd' or dataset == 'msrda':
            self.data, self.label = self.load_data_orgbd_msrda()
        else:
            raise NotImplementedError('dataset not supported: {}'.format(dataset))
        print(f'{split} Dataset: {dataset}, data shape: {self.data.shape}, label shape: {self.label.shape}')

        if target_dataset is not None:
            if preprocess:
                target_file_name = target_data_path.split('/')[-1]
                preprocessed_target_file_name = "preprocessed_" + target_file_name
                preprocessed_target_file_path = target_data_path.replace(target_file_name,
                                                                         preprocessed_target_file_name)
                if os.path.exists(preprocessed_target_file_path):
                    target_data_path = preprocessed_target_file_path
                    self.preprocess = False
                    print(f'load preprocessed target data from {preprocessed_target_file_path}')
                else:
                    self.preprocess = True

        if target_dataset == 'ntu':
            self.data_path = target_data_path
            self.target_data, self.target_label = self.load_data_ntu()
        elif target_dataset == 'sbu':
            self.data_path = target_data_path
            self.target_data, self.target_label = self.load_data_sbu()
        elif target_dataset == 'kinetics':
            self.data_path = target_data_path
            self.target_data, self.target_label = self.load_data_kinetics()
        elif target_dataset == 'ntu2kinetics':
            self.data_path = target_data_path
            self.target_data, self.target_label = self.load_data_ntu(transform='kinetics')
        elif target_dataset == 'posetics2d' or target_dataset == 'posetics3d':
            self.data_path = target_data_path
            self.target_data, self.target_label = self.load_data_posetics(
                axis='2d' if target_dataset == 'posetics2d' else '3d')
        elif target_dataset == 'posetics':
            self.data_path = target_data_path
            self.target_data, self.target_label = self.load_data_posetics(axis='5d')
        elif target_dataset == 'pkummd':
            self.data_path = target_data_path
            self.target_data, self.target_label = self.load_data_pkummd()
        elif target_dataset == 'orgbd' or target_dataset == 'msrda':
            self.data_path = target_data_path
            self.target_data, self.target_label = self.load_data_orgbd_msrda()
        if target_dataset is not None:
            print(f'Target dataset: {target_dataset}, data shape: {self.target_data.shape}, '
                  f'label shape: {self.target_label.shape}')
            if align_st:
                self.align_N()

    def load_data_ntu(self, transform=None):
        # data: N C V T M
        if self.use_mmap:
            npz_data = np.load(self.data_path, mmap_mode='r')
        else:
            npz_data = np.load(self.data_path)
        if self.split == 'train':
            data = npz_data['x_train']
            try:
                label = np.where(npz_data['y_train'] > 0)[1]
                # sample_name = ['train_' + str(i) for i in range(len(data))]
            except:
                label = npz_data['y_train']
        elif self.split == 'test':
            data = npz_data['x_test']
            try:
                label = np.where(npz_data['y_test'] > 0)[1]
                # sample_name = ['test_' + str(i) for i in range(len(data))]
            except:
                label = npz_data['y_test']
        else:
            raise NotImplementedError('data split only supports train/test')
        try:
            N, T, _ = data.shape
            data = data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)  # N C T V M
        except:
            pass
        self.skeleton_info['top'] = 20  # chest
        self.skeleton_info['bottom'] = 0  # pelvis
        self.skeleton_info['left'] = 4  # left shoulder
        self.skeleton_info['right'] = 8  # right shoulder
        self.skeleton_info['thigh_left_top'] = 12
        self.skeleton_info['thigh_left_bottom'] = 13
        self.skeleton_info['thigh_right_top'] = 16
        self.skeleton_info['thigh_right_bottom'] = 17
        self.skeleton_info['calf_left_top'] = 13
        self.skeleton_info['calf_left_bottom'] = 14
        self.skeleton_info['calf_right_top'] = 17
        self.skeleton_info['calf_right_bottom'] = 18

        if transform == 'posetics3d':
            N, C, T, V, M = data.shape
            posetics_data = np.zeros((N, 3, T, 17, 2), dtype=data.dtype)
            posetics_data[:, :, :, 0, :] = data[:, :, :, 3, :]
            posetics_data[:, :, :, 1, :] = data[:, :, :, 2, :]
            posetics_data[:, :, :, 2, :] = data[:, :, :, 20, :]
            posetics_data[:, :, :, 3, :] = data[:, :, :, 1, :]
            posetics_data[:, :, :, 4, :] = data[:, :, :, 0, :]
            posetics_data[:, :, :, 5, :] = data[:, :, :, 4, :]
            posetics_data[:, :, :, 6, :] = data[:, :, :, 8, :]
            posetics_data[:, :, :, 7, :] = data[:, :, :, 5, :]
            posetics_data[:, :, :, 8, :] = data[:, :, :, 9, :]
            posetics_data[:, :, :, 9, :] = data[:, :, :, 7, :]
            posetics_data[:, :, :, 10, :] = data[:, :, :, 11, :]
            posetics_data[:, :, :, 11, :] = data[:, :, :, 12, :]
            posetics_data[:, :, :, 12, :] = data[:, :, :, 16, :]
            posetics_data[:, :, :, 13, :] = data[:, :, :, 13, :]
            posetics_data[:, :, :, 14, :] = data[:, :, :, 17, :]
            posetics_data[:, :, :, 15, :] = data[:, :, :, 14, :]
            posetics_data[:, :, :, 16, :] = data[:, :, :, 18, :]
            data = posetics_data
            self.skeleton_info['top'] = 2
            self.skeleton_info['bottom'] = 4
            self.skeleton_info['left'] = 5
            self.skeleton_info['right'] = 6
        elif transform == 'kinetics':
            N, C, T, V, M = data.shape
            # data = data[:, :2, :, :, :]
            kinetics_data = np.zeros((N, 3, T, 19, 2), dtype=data.dtype)
            kinetics_data[:, :, :, 0, :] = data[:, :, :, 2, :]
            kinetics_data[:, :, :, 1, :] = data[:, :, :, 20, :]
            kinetics_data[:, :, :, 2, :] = data[:, :, :, 8, :]
            kinetics_data[:, :, :, 3, :] = data[:, :, :, 9, :]
            kinetics_data[:, :, :, 4, :] = data[:, :, :, 10, :]
            kinetics_data[:, :, :, 5, :] = data[:, :, :, 4, :]
            kinetics_data[:, :, :, 6, :] = data[:, :, :, 5, :]
            kinetics_data[:, :, :, 7, :] = data[:, :, :, 6, :]
            kinetics_data[:, :, :, 8, :] = data[:, :, :, 16, :]
            kinetics_data[:, :, :, 9, :] = data[:, :, :, 17, :]
            kinetics_data[:, :, :, 10, :] = data[:, :, :, 18, :]
            kinetics_data[:, :, :, 11, :] = data[:, :, :, 12, :]
            kinetics_data[:, :, :, 12, :] = data[:, :, :, 13, :]
            kinetics_data[:, :, :, 13, :] = data[:, :, :, 14, :]
            kinetics_data[:, :, :, 14, :] = data[:, :, :, 3, :]
            kinetics_data[:, :, :, 15, :] = data[:, :, :, 3, :]
            kinetics_data[:, :, :, 16, :] = data[:, :, :, 3, :]
            kinetics_data[:, :, :, 17, :] = data[:, :, :, 3, :]
            kinetics_data[:, :, :, 18, :] = data[:, :, :, 0, :]
            kinetics_data[:, 2, :, :, :] = 0
            # kinetics_data[:, [1, 2], :, :, :] = kinetics_data[:, [2, 1], :, :, :]
            self.skeleton_info['top'] = 1
            self.skeleton_info['bottom'] = 18
            self.skeleton_info['left'] = 2
            self.skeleton_info['right'] = 5
            data = kinetics_data
        elif transform == 'sbu':
            N, C, T, V, M = data.shape
            sbu_data = np.zeros((N, 3, T, 15, 2), dtype=data.dtype)
            sbu_data[:, :, :, 0, :] = data[:, :, :, 3, :]
            sbu_data[:, :, :, 1, :] = data[:, :, :, 2, :]
            sbu_data[:, :, :, 2, :] = data[:, :, :, 1, :]
            sbu_data[:, :, :, 3, :] = data[:, :, :, 4, :]
            sbu_data[:, :, :, 4, :] = data[:, :, :, 5, :]
            sbu_data[:, :, :, 5, :] = data[:, :, :, 7, :]
            sbu_data[:, :, :, 6, :] = data[:, :, :, 8, :]
            sbu_data[:, :, :, 7, :] = data[:, :, :, 9, :]
            sbu_data[:, :, :, 8, :] = data[:, :, :, 11, :]
            sbu_data[:, :, :, 9, :] = data[:, :, :, 12, :]
            sbu_data[:, :, :, 10, :] = data[:, :, :, 13, :]
            sbu_data[:, :, :, 11, :] = data[:, :, :, 14, :]
            sbu_data[:, :, :, 12, :] = data[:, :, :, 16, :]
            sbu_data[:, :, :, 13, :] = data[:, :, :, 17, :]
            sbu_data[:, :, :, 14, :] = data[:, :, :, 18, :]
            self.skeleton_info['top'] = 1
            self.skeleton_info['bottom'] = 2
            self.skeleton_info['left'] = 3
            self.skeleton_info['right'] = 6
            data = sbu_data

        if self.preprocess:
            data = self.pre_process(data)
        if transform == 'kinetics':
            data = data[:, :, :, :15, :]
        return data, label

    def load_data_orgbd_msrda(self):
        # data: N C V T M
        if self.use_mmap:
            npz_data = np.load(self.data_path, mmap_mode='r')
        else:
            npz_data = np.load(self.data_path)
        if self.split == 'train':
            data = npz_data['data']
            label = npz_data['label']
        else:
            raise NotImplementedError('data split only supports train/test')
        N, T, V, C = data.shape
        data = data.transpose(0, 3, 1, 2)  # N C T V
        # data = data.reshape((N, C, T, V, 1))  # N C T V M
        data_new = np.zeros((N, 3, T, V, 2), dtype=np.float32)
        data_new[:, :, :, :, 0] = data
        data_new[:, :, :, :, 1] = data
        data = data_new

        self.skeleton_info['top'] = 2  # chest
        self.skeleton_info['bottom'] = 0  # pelvis
        self.skeleton_info['left'] = 4  # left shoulder
        self.skeleton_info['right'] = 8  # right shoulder

        if self.preprocess:
            data = self.pre_process(data)
        return data, label
    def load_data_pkummd(self):
        # data: N C T V M
        if self.use_mmap:
            npz_data = np.load(self.data_path, mmap_mode='r')
        else:
            npz_data = np.load(self.data_path)
        if self.split == 'train':
            data = npz_data['x_train']
            try:
                label = np.where(npz_data['y_train'] > 0)[1]
                # sample_name = ['train_' + str(i) for i in range(len(data))]
            except:
                label = npz_data['y_train']
        elif self.split == 'test':
            data = npz_data['x_test']
            try:
                label = np.where(npz_data['y_test'] > 0)[1]
                # sample_name = ['test_' + str(i) for i in range(len(data))]
            except:
                label = npz_data['y_test']
        else:
            raise NotImplementedError('data split only supports train/test')

        if self.xy:
            data = data[:, :2, :, :, :]
        self.skeleton_info['top'] = 20
        self.skeleton_info['bottom'] = 0
        self.skeleton_info['left'] = 4
        self.skeleton_info['right'] = 8
        self.skeleton_info['thigh_left_top'] = 12
        self.skeleton_info['thigh_left_bottom'] = 13
        self.skeleton_info['thigh_right_top'] = 16
        self.skeleton_info['thigh_right_bottom'] = 17
        self.skeleton_info['calf_left_top'] = 13
        self.skeleton_info['calf_left_bottom'] = 14
        self.skeleton_info['calf_right_top'] = 17
        self.skeleton_info['calf_right_bottom'] = 18
        if self.preprocess:
            data = self.pre_process(data)
        return data, label

    def load_data_kinetics(self):
        # data: N C T V M
        if self.split == 'train':
            phase = 'train'
        elif self.split == 'n12':
            phase = 'n12'
        else:
            phase = 'val'
        data_path = self.data_path + f'/{phase}_data.npy'
        label_path = self.data_path + f'/{phase}_label.npy'
        if self.use_mmap:
            data = np.load(data_path, mmap_mode='r')
        else:
            data = np.load(data_path)
        label = np.load(label_path)

        # if self.xy:
        #     xy_data = data[:, :2, :, :, :]
        #     data = xy_data
        # else:
        #     data_ = np.zeros_like(data)
        #     data_[:, 0] = data[:, 0]
        #     data_[:, 1] = data[:, 1]
        #     data = data_

        data_ = np.zeros_like(data)
        data_[:, 0] = data[:, 0]
        data_[:, 1] = data[:, 1]
        data = data_


        # data[:, :, :, 15, :] = (data[:, :, :, 8, :] + data[:, :, :, 11, :]) / 2.
        # data = data[:, :, :, :16, :]

        # self.skeleton_info['top'] = 1
        # self.skeleton_info['bottom'] = 15
        # self.skeleton_info['left'] = 5  # left shoulder
        # self.skeleton_info['right'] = 4  # right shoulder
        # if self.preprocess:
        #     self.vertical = False
        #     self.fixed_direction = False
        #     data = self.pre_process(data)
        #     # data = data[:, :2, :, :, :]
        data[:, :, :, 14, :] = (data[:, :, :, 14, :] + data[:, :, :, 15, :]) / 2.
        data = data[:, :, :, :15, :]
        if self.preprocess:
            self.vertical = False
            self.fixed_direction = False
            self.coordinate_transfer_type = -1
            data = self.pre_process(data)
            # data[:, 0] = - data[:, 0]
            data[:, 1] = - data[:, 1]
        return data, label

    def load_data_posetics(self, axis='3d'):
        # data: N C T V M
        if self.split == 'train':
            phase = 'train'
        elif self.split == 'n12':
            phase = 'n12'
        else:
            phase = 'val'
        data_path = self.data_path + f'/{phase}_data.npy'
        label_path = self.data_path + f'/{phase}_label.npy'
        if self.use_mmap:
            data = np.load(data_path, mmap_mode='r')
        else:
            data = np.load(data_path)
        label = np.load(label_path)

        if axis == '2d':
            xy_data = data[:, :2, :, :, :]
            data = xy_data
            assert data.shape[1] == 2
        elif axis == '3d':
            xyz_data = data[:, 2:, :, :, :]
            data = xyz_data
            assert data.shape[1] == 3
        elif axis == '5d':
            xyz_data = data[:, :, :, :, :]
            data = xyz_data
            assert data.shape[1] == 5
        elif axis == '2dto3d':
            xy_data = data[:, :2, :, :, :]
            N, C, T, V, M = xy_data.shape
            xyz_data = np.zeros((N, 3, T, V, M))
            xyz_data[:, 0] = -xy_data[:, 0]
            xyz_data[:, 1] = -xy_data[:, 1]
            data = xyz_data
            assert data.shape[1] == 3
        else:
            raise NotImplementedError('axis not supported: {}'.format(axis))
        self.skeleton_info['top'] = 2
        self.skeleton_info['bottom'] = 4
        self.skeleton_info['left'] = 5
        self.skeleton_info['right'] = 6
        self.skeleton_info['thigh_left_top'] = 11
        self.skeleton_info['thigh_left_bottom'] = 13
        self.skeleton_info['thigh_right_top'] = 12
        self.skeleton_info['thigh_right_bottom'] = 14
        self.skeleton_info['calf_left_top'] = 13
        self.skeleton_info['calf_left_bottom'] = 15
        self.skeleton_info['calf_right_top'] = 14
        self.skeleton_info['calf_right_bottom'] = 16
        if self.preprocess:
            data = self.pre_process(data)
        return data, label

    def load_data_sbu(self):
        root_folder = self.data_path
        fold = self.fold
        # phase = self.phase
        # fold = 0
        phase = 'train' if self.split == 'train' else 'eval'
        folds = {'train': list({1, 2, 3, 4, 5} - {fold}), 'eval': [fold]}

        for i in folds[phase]:
            data_path = '{}/fold{}_data.npy'.format(root_folder, i)
            if os.path.exists(data_path):
                fold = np.load(data_path)
                if i == folds[phase][0]:
                    data = fold
                else:
                    data = np.concatenate((data, fold), axis=0)
            else:
                raise ValueError()

            label_path = '{}/fold{}_label.pkl'.format(root_folder, i)
            if os.path.exists(label_path):
                with open(label_path, 'rb') as f:
                    label_ = pickle.load(f, encoding='latin1')
                    if i == folds[phase][0]:
                        label = label_
                    else:
                        label = np.concatenate((label, label_))
            else:
                raise ValueError()

        if self.xy:
            data = data[:, :2, :, :, :]

        self.skeleton_info['top'] = 1
        self.skeleton_info['bottom'] = 2
        self.skeleton_info['left'] = 3
        self.skeleton_info['right'] = 6
        if self.preprocess:
            data = self.pre_process(data)

        return data, label

    def pre_process(self, data):
        N, C, T, V, M = data.shape
        print(f'preprocess dataset {self.dataset} with normalization {self.normalization}, '
              f'coordinate_transfer {self.coordinate_transfer_type}, vertical {self.vertical}, '
              f'fixed_direction {self.fixed_direction}, fixed_rot {self.fixed_rot}, '
              f'padding_none {self.padding_none}')
        if self.padding_none:
            pp_data = f_padding_none(data)
        else:
            pp_data = data.copy()
        for i, data_numpy in enumerate(tqdm(pp_data)):
            valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
            data_numpy = data_numpy[:, :valid_frame_num, :, :]
            if self.coordinate_transfer_type != -1:
                data_numpy = self.coordinate_transfer(data_numpy)
            if self.vertical:
                data_numpy = rotate_to_vertical(data_numpy, self.skeleton_info)
            if self.fixed_direction:
                data_numpy = rotate_to_face_front(data_numpy, self.skeleton_info)
            if self.fixed_rot is not None:
                data_numpy = rotate_around_y(data_numpy, self.fixed_rot)
            if self.fixed_distance:
                data_numpy = normalize_skeleton_to_fixed_distance(data_numpy, self.skeleton_info)
            pp_data[i, :, :valid_frame_num, :, :] = data_numpy
        if self.normalization == 0:
            self.mean_map = pp_data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
            self.std_map = pp_data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M,
                                                                       C * V)).std(axis=0).reshape((C, 1, V, 1))
            pp_data = (pp_data - self.mean_map) / self.std_map
        return pp_data

    #
    # def normalize(self, data_numpy):
    #     C, T, V, M = data_numpy.shape
    #     if self.normalization == -1:
    #         new = data_numpy
    #     elif self.normalization == 0:
    #         new = (data_numpy - self.mean_map) / self.std_map
    #     elif self.normalization == 1:
    #         new = normalize_skeleton_to_fixed_distance(data_numpy, self.skeleton_info)
    #     else:
    #         raise NotImplementedError('normalization not supported: {}'.format(self.normalization))
    #     return new

    def coordinate_transfer(self, data_numpy):  # change here
        C, T, V, M = data_numpy.shape
        if self.coordinate_transfer_type == 2:
            #  take joints 3  of each person, as origins of each person
            origin_source = data_numpy[:, :, 1, :]
            new = data_numpy - origin_source[:, :, None, :]
        elif self.coordinate_transfer_type == 0:
            #  take joints 3  of first person, as origins of each person
            origin_source = data_numpy[:, :, 1, 0]
            new = data_numpy - origin_source[:, :, None, None]
        elif self.coordinate_transfer_type == 1:
            #  take joints 3  of second person, as origins of each person
            origin_source = data_numpy[:, :, 1, 1]
            new = data_numpy - origin_source[:, :, None, None]
            new = data_numpy - origin_source
        elif self.coordinate_transfer_type == 3:
            #  take mean joints  of first person, as origins of each person
            origin_source = data_numpy[:, :, :, 0].mean(axis=1, keepdims=True).mean(axis=3, keepdims=True)
            new = data_numpy - origin_source[:, :, :, None]
        elif self.coordinate_transfer_type == 4:
            #  take mean joints  of first person, as origins of each person
            origin_source = data_numpy[:, :, :, 0].mean(axis=2, keepdims=True)
            new = data_numpy - origin_source[:, :, :, None]
        elif self.coordinate_transfer_type == 5:
            #  take axis distance of first person, as origins of each person
            origin_source = data_numpy[:, :, :, 0].mean(axis=0, keepdims=True)
            new = data_numpy - origin_source[:, :, :, None]
        elif self.coordinate_transfer_type == 6:
            #  take mean joints  of first person at mean frame, as origins of each person
            origin_source = data_numpy[:, :, :, 0].mean(axis=2, keepdims=True).mean(axis=1, keepdims=True)
            new = data_numpy - origin_source[:, :, :, None]
        elif self.coordinate_transfer_type == 7:
            #  take mean joints of each person at mean frame, as origins of each person
            origin_source = data_numpy[:, :, :, :].mean(axis=2, keepdims=True).mean(axis=1, keepdims=True)
            new = data_numpy - origin_source[:, :, :, :]
        elif self.coordinate_transfer_type == 8:
            #  take mean joints of first person at first frame, as origins of each person
            origin_source = data_numpy[:, :1, :, 0].mean(axis=2, keepdims=True)
            new = data_numpy - origin_source[:, :, :, None]
        elif self.coordinate_transfer_type == 9:
            #  take mean joints of first person at first 5 frames, as origins of each person
            origin_source = data_numpy[:, :5, :, 0].mean(axis=2, keepdims=True).mean(axis=1, keepdims=True)
            new = data_numpy - origin_source[:, :, :, None]
        elif self.coordinate_transfer_type == 88:
            #  take bottom joints of first person at first frame, as origins of each person
            bottom = self.skeleton_info['bottom']
            origin_source = data_numpy[:, :1, bottom, 0]
            new = data_numpy - origin_source[:, :, None, None]
        elif self.coordinate_transfer_type == 888:
            #  take bottom joints of first person at first frame, as origins of each person
            bottom = self.skeleton_info['bottom']
            origin_source = data_numpy[:, :1, bottom, 0]
            new = np.zeros_like(data_numpy)
            for m in range(M):
                if np.sum(data_numpy[:, :, :, m].sum(0).sum(-1) != 0) > 0:
                    new[:, :, :, m] = data_numpy[:, :, :, m] - origin_source[:, :, None]
        elif self.coordinate_transfer_type == 87:
            #  take top joints of first person at first frame, as origins of each person
            top = self.skeleton_info['top']
            origin_source = data_numpy[:, :1, top, 0]
            new = data_numpy - origin_source[:, :, None, None]
        elif self.coordinate_transfer_type == 80:
            #  take bottom joints of first person at each frame, as origins of each person
            bottom = self.skeleton_info['bottom']
            origin_source = data_numpy[:, :, bottom, 0]
            new = data_numpy - origin_source[:, :, None, None]
        elif self.coordinate_transfer_type == 86:
            #  take bottom joints of each person at each frame, as origins of each person
            bottom = self.skeleton_info['bottom']
            origin_source = data_numpy[:, :, bottom, :]
            new = data_numpy - origin_source[:, :, None, :]
        elif self.coordinate_transfer_type == -1:
            new = data_numpy
            # print('no origin transfer')
        else:
            raise NotImplementedError(
                'coordinate_transfer_type not supported: {}'.format(self.coordinate_transfer_type))
        return new

    def _augment(self, data_numpy, dataset=''):
        if self.resize_crop:
            valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
            p_interval = self.p_interval
            if 'kinetics' in dataset and len(self.p_interval) == 2:
                p_interval = [0.75, 1]
            data_numpy = valid_crop_resize(data_numpy, valid_frame_num, p_interval, self.window_size)
        if self.random_crop:
            data_numpy = random_choose(data_numpy, self.window_size)
        if self.random_scale:
            data_numpy = f_random_scale(data_numpy)
        if self.random_shear:
            r = self.random_shear if isinstance(self.random_shear, float) else 0.3
            if 'kinetics' in dataset:
                r *= 0.1
            data_numpy = f_random_shear(data_numpy, r)
        if self.random_rot:
            r = self.random_rot if isinstance(self.random_rot, float) else 0.3
            data_numpy = f_random_rot(data_numpy, r)
        if self.random_shift:
            data_numpy = f_random_shift(data_numpy)
        if self.random_move:
            data_numpy = f_random_move(data_numpy)
        if self.temporal_crop:
            ratio = self.temporal_crop if isinstance(self.temporal_crop, float) else 6
            data_numpy = f_temporal_crop(data_numpy, ratio)
        return data_numpy

    def _strong_aug(self, data_numpy):
        if self.target_dataset is None:
            self.target_dataset = self.dataset
        if '0' in self.strong_aug_method:
            data_numpy = f_temporal_crop(data_numpy, 6)
        if '1' in self.strong_aug_method:
            data_numpy = random_spatial_flip(data_numpy, dataset=self.target_dataset)
        if '2' in self.strong_aug_method:
            r = 0.3
            if 'kinetics' in self.target_dataset:
                r *= 0.1
            data_numpy = f_random_rot(data_numpy, r=r)
        if '3' in self.strong_aug_method:
            data_numpy = gaus_noise(data_numpy)
        if '4' in self.strong_aug_method:
            data_numpy = gaus_filter(data_numpy)
        if '5' in self.strong_aug_method:
            data_numpy = axis_mask(data_numpy)
        if '6' in self.strong_aug_method:
            data_numpy = random_time_flip(data_numpy)

        return data_numpy

    def transform_modality(self, data_numpy):
        if self.bone:
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in self.pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0
        return data_numpy

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        data_numpy = self._augment(data_numpy, dataset=self.dataset)

        if self.ada:
            xw = data_numpy
            data_numpy_1 = self.data[index]
            data_numpy_1 = np.array(data_numpy_1)
            data_numpy_1 = self._augment(data_numpy_1)
            xe1 = self._strong_aug(data_numpy_1)
            data_numpy_2 = self.data[index]
            data_numpy_2 = np.array(data_numpy_2)
            data_numpy_2 = self._augment(data_numpy_2)
            xe2 = self._strong_aug(data_numpy_2)
            return xw, xe1, xe2

        if self.get_pairs:  # v2
            data_numpy_2 = self.data[index]
            data_numpy_2 = np.array(data_numpy_2)
            data_numpy_2 = self._augment(data_numpy_2, dataset=self.dataset)
            data_numpy = (data_numpy, data_numpy_2)

        if self.target_data_path is not None:
            target_index = self.index_s2t(index)
            target_data_numpy = self.target_data[target_index]
            target_label = self.target_label[target_index]
            target_data_numpy = np.array(target_data_numpy)
            target_data_numpy = self._augment(target_data_numpy, dataset=self.target_dataset)
            if self.get_pairs:
                target_data_numpy_2 = self.target_data[target_index]
                target_data_numpy_2 = np.array(target_data_numpy_2)
                target_data_numpy_2 = self._augment(target_data_numpy_2, dataset=self.target_dataset)
                if self.strong_aug_method is not None:
                    # target_data_numpy_e = self._strong_aug(target_data_numpy_2) # v1
                    # target_data_numpy_e = self._strong_aug(target_data_numpy) # v2
                    target_data_numpy_e = self.target_data[target_index]
                    target_data_numpy_e = np.array(target_data_numpy_e)
                    target_data_numpy_e = self._augment(target_data_numpy_e)
                    target_data_numpy_e = self._strong_aug(target_data_numpy_e)
                    target_data_numpy = (target_data_numpy, target_data_numpy_2, target_data_numpy_e)
                else:
                    target_data_numpy = (target_data_numpy, target_data_numpy_2)
            return data_numpy, label, target_data_numpy, target_label
        else:
            return data_numpy, label

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def index_s2t(self, index):
        if not self.align_st:
            target_index = random.randint(0, len(self.target_label) - 1)
        else:
            target_index = index
        return target_index

    def align_N(self):
        N_source = len(self.label)
        N_target = len(self.target_label)
        if N_source > N_target:
            roundtimes = N_source // N_target
            tail = N_source - N_target * roundtimes
            tail_index = random.sample(range(N_target), tail)
            self.target_data = np.concatenate(
                (np.tile(self.target_data, (roundtimes, 1, 1, 1, 1)),
                 self.target_data[tail_index, :, :, :, :]), axis=0)
            self.target_label = np.concatenate((np.tile(self.target_label, roundtimes),
                                                self.target_label[tail_index]), axis=0)
            print(f'source {N_source} > target {N_target}, padding target dataset to {len(self.target_label)}')
        else:
            roundtimes = N_target // N_source
            tail = N_target - N_source * roundtimes
            tail_index = random.sample(range(N_source), tail)
            self.data = np.concatenate(
                (np.tile(self.data, (roundtimes, 1, 1, 1, 1)),
                 self.data[tail_index, :, :, :, :]), axis=0)
            self.label = np.concatenate((np.tile(self.label, roundtimes),
                                         self.label[tail_index]), axis=0)
            print(f'source {N_source} < target {N_target}, padding source dataset to {len(self.label)}')
        assert len(self.label) == len(self.target_label)
        assert len(self.data) == len(self.target_data)

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


ntu_pairs = (
    (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4),
    (6, 5), (7, 6), (8, 20), (9, 8), (10, 9), (11, 10),
    (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16),
    (18, 17), (19, 18), (21, 22), (20, 20), (22, 7), (23, 24), (24, 11)
)
sbu_pairs = (
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6),
    (6, 7), (7, 8), (2, 9), (9, 10), (10, 11),
    (2, 12), (12, 13), (13, 14)
)
kinetics_pairs = (
    (4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
    (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
    (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)
)
posetics_pairs = (
    (0, 1), (1, 2), (2, 3), (2, 5), (2, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (3, 4),
    (4, 11), (4, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
)

if __name__ == '__main__':
    feeder = Feeder('ntu',
                    '/data/wangkun/project/datasets/NTU_dataset/ntu/ntu2kinetics/NTU60_CV.npz',
                    split='test', coordinate_transfer_type=0, normalization=1)
    feeder.get_mean_map()
    print(feeder.mean_map.shape, feeder.std_map.shape, feeder.min_map.shape, feeder.max_map.shape,
          feeder.mean_mean.shape)
