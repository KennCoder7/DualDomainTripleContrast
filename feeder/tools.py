import math
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch import nn
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def f_random_shear(data_numpy, r=0.3):
    s1_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]
    s2_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]

    R = np.array([[1, s1_list[0], s2_list[0]],
                  [s1_list[1], 1, s2_list[1]],
                  [s1_list[2], s2_list[2], 1]])

    R = R.transpose()
    data_numpy = np.dot(data_numpy.transpose([1, 2, 3, 0]), R)
    data_numpy = data_numpy.transpose(3, 0, 1, 2)
    return data_numpy


def f_temporal_crop(data_numpy, temporal_padding_ratio=6):
    C, T, V, M = data_numpy.shape
    padding_len = T // temporal_padding_ratio
    frame_start = np.random.randint(0, padding_len * 2 + 1)
    data_numpy = np.concatenate((data_numpy[:, :padding_len][:, ::-1],
                                 data_numpy,
                                 data_numpy[:, -padding_len:][:, ::-1]),
                                axis=1)
    data_numpy = data_numpy[:, frame_start:frame_start + T]
    return data_numpy


# def _rot(rot):
#     """
#     rot: T,3
#     """
#     cos_r, sin_r = rot.cos(), rot.sin()  # T,3
#     zeros = torch.zeros(rot.shape[0], 1)  # T,1
#     ones = torch.ones(rot.shape[0], 1)  # T,1
#
#     r1 = torch.stack((ones, zeros, zeros), dim=-1)  # T,1,3
#     rx2 = torch.stack((zeros, cos_r[:, 0:1], sin_r[:, 0:1]), dim=-1)  # T,1,3
#     rx3 = torch.stack((zeros, -sin_r[:, 0:1], cos_r[:, 0:1]), dim=-1)  # T,1,3
#     rx = torch.cat((r1, rx2, rx3), dim=1)  # T,3,3
#
#     ry1 = torch.stack((cos_r[:, 1:2], zeros, -sin_r[:, 1:2]), dim=-1)
#     r2 = torch.stack((zeros, ones, zeros), dim=-1)
#     ry3 = torch.stack((sin_r[:, 1:2], zeros, cos_r[:, 1:2]), dim=-1)
#     ry = torch.cat((ry1, r2, ry3), dim=1)
#
#     rz1 = torch.stack((cos_r[:, 2:3], sin_r[:, 2:3], zeros), dim=-1)
#     r3 = torch.stack((zeros, zeros, ones), dim=-1)
#     rz2 = torch.stack((-sin_r[:, 2:3], cos_r[:, 2:3], zeros), dim=-1)
#     rz = torch.cat((rz1, rz2, r3), dim=1)
#
#     rot = rz.matmul(ry).matmul(rx)
#     return rot
#
#
# def rot(data_numpy, theta=0.3, fixed_theta=-1):
#     """
#     data_numpy: C,T,V,M
#     """
#     data_torch = torch.from_numpy(data_numpy)
#     data_torch = data_torch.to(torch.float32)
#     C, T, V, M = data_torch.shape
#     data_torch = data_torch.permute(1, 0, 2, 3).contiguous().view(T, C, V * M)  # T,3,V*M
#     rot = torch.zeros(3).uniform_(-theta, theta)
#     if fixed_theta != -1:
#         rot = torch.zeros(3).fill_(fixed_theta)
#     rot = torch.stack([rot, ] * T, dim=0)
#     rot = _rot(rot)  # T,3,3
#     data_torch = torch.matmul(rot, data_torch)
#     data_torch = data_torch.view(T, C, V, M).permute(1, 0, 2, 3).contiguous()
#
#     return data_torch.numpy()


# def f_random_rot(data_numpy, rand_rotate=0.3):
#     # input: C,T,V,M
#     C, T, V, M = data_numpy.shape
#
#     R = np.eye(3)
#     for i in range(3):
#         theta = (np.random.rand() * 2 - 1) * rand_rotate * np.pi
#         Ri = np.eye(3)
#         Ri[C - 1, C - 1] = 1
#         Ri[0, 0] = np.cos(theta)
#         Ri[0, 1] = np.sin(theta)
#         Ri[1, 0] = -np.sin(theta)
#         Ri[1, 1] = np.cos(theta)
#         R = R * Ri
#
#     data_numpy = np.matmul(R, data_numpy.reshape(C, T * V * M)).reshape(C, T, V, M).astype('float32')
#     return data_numpy


def f_random_rot(data_numpy, r=0.3):  # v2 rot round y
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape

    R = np.eye(3)
    theta = (np.random.rand() * 2 - 1) * r * np.pi
    Ri = np.eye(3)
    Ri[1, 1] = 1
    Ri[0, 0] = np.cos(theta)
    Ri[0, 2] = np.sin(theta)
    Ri[2, 0] = -np.sin(theta)
    Ri[2, 2] = np.cos(theta)
    R = R * Ri

    data_numpy = np.matmul(R, data_numpy.reshape(C, T * V * M)).reshape(C, T, V, M).astype('float32')
    return data_numpy


def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]


def f_random_move(data_numpy,
                  angle_candidate=[-10., -5., 0., 5., 10.],
                  scale_candidate=[0.9, 1.0, 1.1],
                  transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                  move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def f_random_shift(data_numpy):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[
                                                             t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy


def top_k_by_category(label, score, top_k):
    instance_num, class_num = score.shape
    rank = score.argsort()
    hit_top_k = [[] for i in range(class_num)]
    for i in range(instance_num):
        l = label[i]
        hit_top_k[l].append(l in rank[i, -top_k:])

    accuracy_list = []
    for hit_per_category in hit_top_k:
        if hit_per_category:
            accuracy_list.append(sum(hit_per_category) * 1.0 / len(hit_per_category))
        else:
            accuracy_list.append(0.0)
    return accuracy_list


def calculate_recall_precision(label, score):
    instance_num, class_num = score.shape
    rank = score.argsort()
    confusion_matrix = np.zeros([class_num, class_num])

    for i in range(instance_num):
        true_l = label[i]
        pred_l = rank[i, -1]
        confusion_matrix[true_l][pred_l] += 1

    precision = []
    recall = []

    for i in range(class_num):
        true_p = confusion_matrix[i][i]
        false_n = sum(confusion_matrix[i, :]) - true_p
        false_p = sum(confusion_matrix[:, i]) - true_p
        precision.append(true_p * 1.0 / (true_p + false_p))
        recall.append(true_p * 1.0 / (true_p + false_n))

    return precision, recall


def valid_crop_resize(data_numpy, valid_frame_num, p_interval, window):
    if window == -1:
        return data_numpy
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin
    if valid_size == 0:
        valid_size = window

    # crop
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1 - p) * valid_size / 2)
        data = data_numpy[:, begin + bias:end - bias, :, :]  # center_crop
        cropped_length = data.shape[1]
    else:
        p = np.random.rand(1) * (p_interval[1] - p_interval[0]) + p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size * p)), 64),
                                    valid_size)  # constraint cropped_length lower bound as 64
        bias = np.random.randint(0, valid_size - cropped_length + 1)
        data = data_numpy[:, begin + bias:begin + bias + cropped_length, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)

    # resize
    data = torch.tensor(data, dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, None, :, :]
    data = F.interpolate(data, size=(C * V * M, window), mode='bilinear',
                         align_corners=False).squeeze()  # could perform both up sample and down sample
    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()

    return data


def rotate_to_vertical(data_numpy, skeleton_info, frame_num=1):
    C, T, V, M = data_numpy.shape
    top_index, bottom_index, left_index, right_index = (skeleton_info['top'],
                                                        skeleton_info['bottom'],
                                                        skeleton_info['left'],
                                                        skeleton_info['right'])
    # 获取第一个人的头部、胸部、左肩膀和右肩膀的坐标
    top = data_numpy[:, :frame_num, top_index, 0]  # C, T
    bottom = data_numpy[:, :frame_num, bottom_index, 0]  # C, T

    # 计算第一个人垂直于xz平面的朝向向量
    current_direction = top - bottom
    current_direction = current_direction.mean(axis=1)  # 平均化以获得稳定的方向

    # 目标是垂直于xz平面，即y轴方向
    target_direction = np.array([0, 1, 0])

    # 计算旋转矩阵
    r = R.align_vectors([target_direction], [current_direction.T])[0]
    rotation_matrix = torch.tensor(r.as_matrix(), dtype=torch.float32)  # 3x3

    # 应用旋转矩阵到所有人的数据
    rotated_data = torch.matmul(rotation_matrix, torch.tensor(data_numpy).reshape(C, -1)).reshape(C, T, V, M)

    return rotated_data.numpy()


def rotate_to_vertical_per_frame(data_numpy, skeleton_info):
    C, T, V, M = data_numpy.shape
    top_index, bottom_index = skeleton_info['top'], skeleton_info['bottom']

    rotated_data = np.zeros_like(data_numpy)

    for t in range(T):
        # 获取每一帧第一个人的头部和胸部的坐标
        top = data_numpy[:, t, top_index, 0]  # C
        bottom = data_numpy[:, t, bottom_index, 0]  # C

        # 计算第一个人垂直于xz平面的朝向向量
        current_direction = top - bottom

        # 目标是垂直于xz平面，即y轴方向
        target_direction = np.array([0, 1, 0])

        # 计算旋转矩阵，使current_direction对齐到target_direction
        r = R.align_vectors([target_direction], [current_direction])[0]
        rotation_matrix = torch.tensor(r.as_matrix(), dtype=torch.float32)  # 3x3

        # 应用旋转矩阵到所有人的数据
        frame_data = data_numpy[:, t].reshape(C, V * M)  # C x (V*M)
        rotated_frame_data = torch.matmul(rotation_matrix, torch.tensor(frame_data))  # C x (V*M)
        rotated_data[:, t] = rotated_frame_data.reshape(C, V, M)

    return rotated_data


def rotate_to_face_front(data_numpy, skeleton_info, frame_num=1, target_angle=0, dataset='ntu'):
    C, T, V, M = data_numpy.shape
    top_index, bottom_index, left_index, right_index = (skeleton_info['top'],
                                                        skeleton_info['bottom'],
                                                        skeleton_info['left'],
                                                        skeleton_info['right'])
    # 提取左右肩膀的坐标 (C, T, M)

    left = data_numpy[:, :frame_num, left_index, 0]
    right = data_numpy[:, :frame_num, right_index, 0]

    # 计算肩膀之间的向量 (C, T)
    shoulder_vector = right - left

    # 计算在 xz 平面上的投影 (T, 2)
    shoulder_xz = shoulder_vector[[0, 2], :]  # 仅取 x 和 z 分量

    # 计算当前向量与 x 轴之间的夹角
    current_angle = np.arctan2(shoulder_xz[1], shoulder_xz[0])  # 返回当前夹角 θ
    # print('Current angle:', current_angle.mean() * 180 / np.pi)
    # 计算所需旋转角度
    rotation_angle = current_angle.mean() - target_angle  # 得到最终所需旋转角度
    # print('Rotation angle:', rotation_angle * 180 / np.pi)

    # 构建旋转矩阵
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)

    rotation_matrix = np.array([
        [cos_theta, 0, sin_theta],
        [0, 1, 0],
        [-sin_theta, 0, cos_theta]
    ])

    # 将旋转矩阵应用到每个坐标点上
    rotated_data = np.tensordot(rotation_matrix, data_numpy, axes=(1, 0))

    # 如果旋转角度大于 90 度，那么我们需要翻转数据
    # if rotation_angle > np.pi / 2:
    #     if dataset == 'ntu':
    #         rotated_data = flip_skeleton_ntu(rotated_data)
    #     else:
    #         raise ValueError('Unknown dataset: {}'.format(dataset))
    return rotated_data


def rotate_to_face_front_per_frame(data_numpy, skeleton_info, target_angle=0, dataset='ntu'):
    C, T, V, M = data_numpy.shape
    top_index, bottom_index, left_index, right_index = (skeleton_info['top'],
                                                        skeleton_info['bottom'],
                                                        skeleton_info['left'],
                                                        skeleton_info['right'])

    rotated_data = np.zeros_like(data_numpy)

    for t in range(T):
        # 提取当前帧左右肩膀的坐标 (C, M)
        left = data_numpy[:, t, left_index, 0]
        right = data_numpy[:, t, right_index, 0]

        # 计算肩膀之间的向量 (C)
        shoulder_vector = right - left

        # 计算在 xz 平面上的投影 (2)
        shoulder_xz = shoulder_vector[[0, 2]]  # 仅取 x 和 z 分量

        # 计算当前向量与 x 轴之间的夹角
        current_angle = np.arctan2(shoulder_xz[1], shoulder_xz[0])  # 返回当前夹角 θ

        # 计算所需旋转角度
        rotation_angle = current_angle - target_angle  # 得到最终所需旋转角度

        # 构建旋转矩阵
        cos_theta = np.cos(rotation_angle)
        sin_theta = np.sin(rotation_angle)

        rotation_matrix = np.array([
            [cos_theta, 0, sin_theta],
            [0, 1, 0],
            [-sin_theta, 0, cos_theta]
        ])

        # 将旋转矩阵应用到当前帧的每个坐标点上
        rotated_frame_data = np.tensordot(rotation_matrix, data_numpy[:, t], axes=(1, 0))

        # 存储旋转后的数据
        rotated_data[:, t] = rotated_frame_data

    return rotated_data


def rotate_around_y(data_numpy, theta):
    """
    沿 y 轴旋转三维坐标数据。

    :param data_numpy: (C, T, V, M) 的四维数组，其中 C=3 表示 (x, y, z)
    :param theta: 沿 y 轴旋转的角度，单位为弧度
    :return: 旋转后的数据，形状与输入相同
    """
    # 旋转矩阵
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    rotation_matrix = np.array([
        [cos_theta, 0, sin_theta],
        [0, 1, 0],
        [-sin_theta, 0, cos_theta]
    ])

    # 将旋转矩阵应用到每个坐标点上
    # data_numpy 形状为 (C, T, V, M)，我们需要对 C 维度应用旋转
    rotated_data = np.tensordot(rotation_matrix, data_numpy, axes=(1, 0))

    return rotated_data


def flip_skeleton_ntu(data_numpy):
    """
    Flip skeleton data for NTU dataset.

    :param data_numpy: (C, T, V, M) 的四维数组，其中 C=3 表示 (x, y, z)
    :return: 翻转后的数据，形状与输入相同
    """
    # 交换左右肩膀
    data_numpy[:, :, [4, 8], :] = data_numpy[:, :, [8, 4], :]

    # 交换左右手腕
    data_numpy[:, :, [5, 9], :] = data_numpy[:, :, [9, 5], :]
    data_numpy[:, :, [6, 10], :] = data_numpy[:, :, [10, 6], :]
    data_numpy[:, :, [7, 11], :] = data_numpy[:, :, [11, 7], :]
    data_numpy[:, :, [21, 23], :] = data_numpy[:, :, [23, 21], :]
    data_numpy[:, :, [22, 24], :] = data_numpy[:, :, [24, 22], :]

    # 交换左右膝盖
    data_numpy[:, :, [12, 16], :] = data_numpy[:, :, [16, 12], :]
    data_numpy[:, :, [13, 17], :] = data_numpy[:, :, [17, 13], :]
    data_numpy[:, :, [14, 18], :] = data_numpy[:, :, [18, 14], :]
    data_numpy[:, :, [15, 19], :] = data_numpy[:, :, [19, 15], :]

    return data_numpy


def normalize_skeleton_to_fixed_distance(data_numpy, skeleton_info, target_distance=1.):
    """
    标准化每个人体的三维节点数据，使头部和腰部节点之间的距离为指定值。
    """
    x = torch.tensor(data_numpy, dtype=torch.float32)  # (C, T, V, M)
    top_index, bottom_index, left_index, right_index = (skeleton_info['top'],
                                                        skeleton_info['bottom'],
                                                        skeleton_info['left'],
                                                        skeleton_info['right'])
    thigh_left_top_index, thigh_left_bottom_index = (skeleton_info['thigh_left_top'],
                                                     skeleton_info['thigh_left_bottom'])
    thigh_right_top_index, thigh_right_bottom_index = (skeleton_info['thigh_right_top'],
                                                       skeleton_info['thigh_right_bottom'])
    calf_left_top_index, calf_left_bottom_index = (skeleton_info['calf_left_top'],
                                                   skeleton_info['calf_left_bottom'])
    calf_right_top_index, calf_right_bottom_index = (skeleton_info['calf_right_top'],
                                                     skeleton_info['calf_right_bottom'])

    # 提取头部和腰部节点的坐标
    # x: (C, T, V, M)
    head_position = x[:, :, top_index, :]  # 形状: (C, T, M)
    waist_position = x[:, :, bottom_index, :]  # 形状: (C, T, M)
    thigh_left_top_position = x[:, :, thigh_left_top_index, :]  # 形状: (C, T, M)
    thigh_left_bottom_position = x[:, :, thigh_left_bottom_index, :]  # 形状: (C, T, M)
    thigh_right_top_position = x[:, :, thigh_right_top_index, :]  # 形状: (C, T, M)
    thigh_right_bottom_position = x[:, :, thigh_right_bottom_index, :]  # 形状: (C, T, M)
    calf_left_top_position = x[:, :, calf_left_top_index, :]  # 形状: (C, T, M)
    calf_left_bottom_position = x[:, :, calf_left_bottom_index, :]  # 形状: (C, T, M)
    calf_right_top_position = x[:, :, calf_right_top_index, :]  # 形状: (C, T, M)
    calf_right_bottom_position = x[:, :, calf_right_bottom_index, :]  # 形状: (C, T, M)

    # 计算头部和腰部节点之间的欧氏距离
    # 计算每个样本、每个时间帧下的距离
    distance = torch.norm(head_position - waist_position, dim=0, keepdim=True)  # 形状: (1, T, M)
    distance_thigh_left = torch.norm(thigh_left_top_position - thigh_left_bottom_position, dim=0, keepdim=True)
    distance_thigh_right = torch.norm(thigh_right_top_position - thigh_right_bottom_position, dim=0, keepdim=True)
    distance_calf_left = torch.norm(calf_left_top_position - calf_left_bottom_position, dim=0, keepdim=True)
    distance_calf_right = torch.norm(calf_right_top_position - calf_right_bottom_position, dim=0, keepdim=True)
    distance += (distance_thigh_left + distance_thigh_right) / 2 + (distance_calf_left + distance_calf_right) / 2
    # 防止除以零，添加一个很小的值 epsilon
    epsilon = 1e-8
    distance = distance + epsilon
    distance = distance.unsqueeze(2)

    # 计算缩放因子，使得距离标准化为 target_distance
    scale = target_distance / distance
    # scale = scale.unsqueeze(0).expand_as(x)
    # 将缩放因子应用到所有节点
    x_normalized = x * scale  # 广播机制自动匹配维度
    x_normalized = x_normalized.numpy()

    return x_normalized


def f_random_scale(data_numpy, scale_range=(0.9, 1.1), bool_scale_z=True):
    """
    对人体骨骼数据在x和y轴上进行随机拉伸
    :param data_numpy: 输入数据的形状为 (C, T, V, M)
    :param scale_range: 缩放系数的范围，默认为 (0.9, 1.1)
    :return: 经过拉伸后的数据，形状不变
    """
    C, T, V, M = data_numpy.shape

    # 随机生成x和y轴的缩放系数
    scale_x = np.random.uniform(scale_range[0], scale_range[1])
    scale_y = np.random.uniform(scale_range[0], scale_range[1])
    scale_z = np.random.uniform(scale_range[0], scale_range[1])

    # 复制数据避免修改原数据
    data_scaled = np.copy(data_numpy)

    # 对x轴 (C=0) 和 y轴 (C=1) 进行拉伸
    data_scaled[0, :, :, :] *= scale_x  # 拉伸 x 轴
    data_scaled[1, :, :, :] *= scale_y  # 拉伸 y 轴
    if bool_scale_z:
        data_scaled[2, :, :, :] *= scale_z  # 拉伸 z 轴

    return data_scaled


def f_padding_none(data):
    s = data.copy()
    # print('pad the null frames with the previous frames')
    for i_s, skeleton in enumerate(tqdm(s)):  # pad
        if skeleton.sum() == 0:
            print(i_s, ' has no skeleton')
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            if person[0].sum() == 0:
                index = (person.sum(-1).sum(-1) != 0)
                tmp = person[index].copy()
                person *= 0
                person[:len(tmp)] = tmp
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    if person[i_f:].sum() == 0:
                        rest = len(person) - i_f
                        num = int(np.ceil(rest / i_f))
                        pad = np.concatenate([person[0:i_f] for _ in range(num)], 0)[:rest]
                        s[i_s, i_p, i_f:] = pad
                        break
                    else:
                        person[i_f] = person[i_f - 1]
    return s


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
        return np.eye(3)
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


transform_order = {
    'ntu': [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 16, 17, 18, 19, 12, 13, 14, 15, 20, 23, 24, 21, 22],
    'pkummd': [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 16, 17, 18, 19, 12, 13, 14, 15, 20, 23, 24, 21, 22],
    # 'kinetics': [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16],
    'kinetics': [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 14],
    'posetics': [0, 1, 2, 3, 4, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
}


def random_spatial_flip(seq, p=0.5, dataset='ntu'):
    if random.random() < p:
        # Do the left-right transform C,T,V,M
        if 'ntu' in dataset:
            index = transform_order['ntu']
        elif 'pkummd' in dataset:
            index = transform_order['pkummd']
        elif 'kinetics' in dataset:
            index = transform_order['kinetics']
        elif 'posetics' in dataset:
            index = transform_order['posetics']
        else:
            index = transform_order['ntu']
        trans_seq = seq[:, :, index, :]
        return trans_seq
    else:
        return seq


def random_time_flip(seq, p=0.5):
    T = seq.shape[1]
    if random.random() < p:
        time_range_order = [i for i in range(T)]
        time_range_reverse = list(reversed(time_range_order))
        return seq[:, time_range_reverse, :, :]
    else:
        return seq


def gaus_noise(data_numpy, mean=0, std=0.01, p=0.5):
    if random.random() < p:
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        noise = np.random.normal(mean, std, size=(C, T, V, M))
        return temp + noise
    else:
        return data_numpy


def gaus_filter(data_numpy):
    g = GaussianBlurConv(3)
    return g(data_numpy)


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3, kernel=15, sigma=[0.1, 2]):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        self.kernel = kernel
        self.min_max_sigma = sigma
        radius = int(kernel / 2)
        self.kernel_index = np.arange(-radius, radius + 1)

    def __call__(self, x):
        sigma = random.uniform(self.min_max_sigma[0], self.min_max_sigma[1])
        blur_flter = np.exp(-np.power(self.kernel_index, 2.0) / (2.0 * np.power(sigma, 2.0)))
        kernel = torch.from_numpy(blur_flter).unsqueeze(0).unsqueeze(0)
        # kernel =  kernel.float()
        kernel = kernel.double()
        kernel = kernel.repeat(self.channels, 1, 1, 1)  # (3,1,1,5)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

        prob = np.random.random_sample()
        x = torch.from_numpy(x)
        x = x.double()
        if prob < 0.5:
            x = x.permute(3, 0, 2, 1)  # M,C,V,T
            x = F.conv2d(x, self.weight, padding=(0, int((self.kernel - 1) / 2)), groups=self.channels)
            x = x.permute(1, -1, -2, 0)  #C,T,V,M
        x = x.float()
        return x.numpy()


class Zero_out_axis(object):
    def __init__(self, axis=None):
        self.first_axis = axis

    def __call__(self, data_numpy):
        if self.first_axis != None:
            axis_next = self.first_axis
        else:
            axis_next = random.randint(0, 2)

        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        x_new = np.zeros((T, V, M))
        temp[axis_next] = x_new
        return temp


def axis_mask(data_numpy, p=0.5):
    am = Zero_out_axis()
    if random.random() < p:
        return am(data_numpy)
    else:
        return data_numpy


if __name__ == '__main__':
    T = 128
    data = np.arange(1, T + 1).reshape(1, T, 1, 1)
    new = f_temporal_crop(data, 6).reshape(T)
    data = data.reshape(T)
    print(data)
    print(new)
