import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from utils import utils
import torchvision
import os



class Model(nn.Module):
    '''
    Input shape:
    Input shape should be (N, C, T, V, M)
    where N is the number of samples,
          C is the number of input channels,
          T is the length of the sequence,
          V is the number of joints
      and M is the number of people.
    '''

    def __init__(self,
                 in_channel=3,
                 num_joint=25,
                 num_person=2,
                 out_channel=64,
                 window_size=64,
                 num_class=60,
                 return_ft=False,
                 max_person_logits=True,
                 norm='bn',
                 ):
        super(Model, self).__init__()
        self.num_person = num_person
        self.num_class = num_class
        self.in_channel = in_channel
        self.num_joint = num_joint
        if norm == 'bn':
            self.data_bn_x = nn.BatchNorm1d(in_channel * num_joint)
            self.data_bn_m = nn.BatchNorm1d(in_channel * num_joint)
        elif norm == 'in':
            self.data_bn_x = nn.InstanceNorm1d(in_channel * num_joint)
            self.data_bn_m = nn.InstanceNorm1d(in_channel * num_joint)
        else:
            self.data_bn_x = lambda x: x
            self.data_bn_m = lambda x: x
        # position
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3, 1), stride=1,
                      padding=(1, 0)),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))
        # motion
        self.conv1m = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
        )
        self.conv2m = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3, 1), stride=1,
                      padding=(1, 0)),
        )

        self.conv3m = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4m = nn.Sequential(
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))

        # concatenate motion & position
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel * 2, out_channels=out_channel * 2, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel * 2, out_channels=out_channel * 4, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )

        self.fc7 = nn.Sequential(
            nn.Linear((out_channel * 4) * (window_size // 16) * (window_size // 16), 256 * 2),
            # 4*4 for window=64; 8*8 for window=128
            nn.PReLU(),
            nn.Dropout2d(p=0.5))
        self.fc8 = nn.Linear(256 * 2, num_class)
        self.return_ft = return_ft
        self.max_person_logits = max_person_logits
        self.pretrained_path = None
        # initial weight
        # utils.initial_model_weight(layers=list(self.children()))
        # print('weight initial finished!')

    def forward(self, x, return_jm=False):
        N, C, T, V, M = x.size()  # N0, C1, T2, V3, M4
        C = self.in_channel
        x = x[:, :C, :, :, :]
        V = self.num_joint
        x = x[:, :, :, :V, :]
        motion = x[:, :, 1::, :, :] - x[:, :, 0:-1, :, :]
        motion = motion.permute(0, 1, 4, 2, 3).contiguous().view(N, C * M, T - 1, V)
        motion = F.upsample(motion, size=(T, V), mode='bilinear',
                            align_corners=False).contiguous().view(N, C, M, T, V).permute(0, 1, 3, 4, 2)

        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn_x(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 3, 4, 2, 1).contiguous()

        motion = motion.permute(0, 4, 3, 1, 2).contiguous()
        motion = motion.view(N * M, V * C, T)
        motion = self.data_bn_m(motion)
        motion = motion.view(N, M, V, C, T)
        motion = motion.permute(0, 3, 4, 2, 1).contiguous()

        logits = []
        logits_j = []
        logits_m = []
        for i in range(self.num_person):
            # position
            # N0,C1,T2,V3 point-level
            out = self.conv1(x[:, :, :, :, i])
            #print('conv1',out.shape)
            out = self.conv2(out)
            #print('conv2',out.shape)
            # N0,V1,T2,C3, global level
            out = out.permute(0, 3, 2, 1).contiguous()
            out = self.conv3(out)
            #print('conv3',out.shape)
            out_p = self.conv4(out)
            #print('out_p',out_p.shape)
            logits_j.append(out_p)

            # motion
            # N0,T1,V2,C3 point-level
            out = self.conv1m(motion[:, :, :, :, i])
            out = self.conv2m(out)
            # N0,V1,T2,C3, global level
            out = out.permute(0, 3, 2, 1).contiguous()
            out = self.conv3m(out)
            out_m = self.conv4m(out)
            logits_m.append(out_m)

            # concat
            out = torch.cat((out_p, out_m), dim=1)
            #print('cat',out.shape)
            out = self.conv5(out)
            #print('conv5',out.shape)
            out = self.conv6(out)
            #print('conv6',out.shape)

            logits.append(out)

        # max out logits
        out = torch.max(logits[0], logits[1]) if self.max_person_logits else torch.mean(torch.stack(logits), 0)
        # out = torch.mean(torch.stack(logits), 0)
        if return_jm:
            out_j = torch.max(logits_j[0], logits_j[1]).view(out.size(0), -1) if self.max_person_logits else torch.mean(
                torch.stack(logits_j), 0).view(out.size(0), -1)
            out_m = torch.max(logits_m[0], logits_m[1]).view(out.size(0), -1) if self.max_person_logits else torch.mean(
                torch.stack(logits_m), 0).view(out.size(0), -1)
            # out_j = torch.mean(torch.stack(logits_j), 0).view(out.size(0), -1)
            # out_m = torch.mean(torch.stack(logits_m), 0).view(out.size(0), -1)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.fc7(out)
        #print('fc7',out.shape)
        cls = self.fc8(out)
        #print('fc8',out.shape)
        '''
        print(out)
        print(out.shape)
        '''

        t = cls
        assert not ((t != t).any())  # find out nan in tensor
        assert not (t.abs().sum() == 0)  # find out 0 tensor

        if self.return_ft:
            if return_jm:
                return cls, out, out_j, out_m
            return cls, out
        return cls

    def extract_feature(self, xs, xt):
        self.return_ft = True
        with torch.no_grad():
            cls_s, emb_s_q = self.forward(xs)
            cls_t, emb_t_q = self.forward(xt)
        return cls_s, cls_t, emb_s_q, emb_t_q


    def load_pretrained_model(self, pretrained_path=None):
        if pretrained_path:
            self.load_state_dict(torch.load(pretrained_path))
            print('load model from {}'.format(pretrained_path))
        else:
            print('no pretrained model')
        return True


if __name__ == '__main__':
    from thop import profile
    model = Model()
    input = torch.randn(1, 3, 64, 25, 2)
    flops, params = profile(model, inputs=(input,))
    print('GFLOPs: %.2fG' % (flops / 10 ** 9), 'Params: %.2fM' % (params / 10 ** 6))
    # GFLOPs: 0.39G Params: 2.65M