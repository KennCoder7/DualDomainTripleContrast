import torch
import torch.nn as nn
from .HCN import Model as HCN
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self, encoder_args,
                 queue_size,
                 queue_size_s,
                 momentum, latent_dim,
                 lambda_val=0.1,
                 lambda_center=0.5,
                 context=True,
                 Temperature=0.07, topk=1,
                 Temperature_s=0.07,
                 Temperature_uda=0.07, topk_uda=1,
                 maskout=True,
                 pretrained_path=None):
        super(Model, self).__init__()
        self.encoder_q = HCN(**encoder_args)
        self.encoder_k = HCN(**encoder_args)

        self.fc_q = nn.Sequential(nn.Linear(256 * 2, latent_dim),
                                  nn.ReLU(),
                                  nn.Linear(latent_dim, latent_dim))
        self.fc_k = nn.Sequential(nn.Linear(256 * 2, latent_dim),
                                  nn.ReLU(),
                                  nn.Linear(latent_dim, latent_dim))

        self.fc_j_q = nn.Sequential(nn.Linear(32768, latent_dim),
                                    nn.ReLU(),
                                    nn.Linear(latent_dim, latent_dim))
        self.fc_j_k = nn.Sequential(nn.Linear(32768, latent_dim),
                                    nn.ReLU(),
                                    nn.Linear(latent_dim, latent_dim))

        self.fc_m_q = nn.Sequential(nn.Linear(32768, latent_dim),
                                    nn.ReLU(),
                                    nn.Linear(latent_dim, latent_dim))
        self.fc_m_k = nn.Sequential(nn.Linear(32768, latent_dim),
                                    nn.ReLU(),
                                    nn.Linear(latent_dim, latent_dim))

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.fc_q.parameters(), self.fc_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.fc_j_q.parameters(), self.fc_j_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.fc_m_q.parameters(), self.fc_m_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue_t", torch.randn(latent_dim, queue_size))
        self.queue_t = F.normalize(self.queue_t, dim=0)
        self.register_buffer("queue_tj", torch.randn(latent_dim, queue_size))
        self.queue_tj = F.normalize(self.queue_tj, dim=0)
        self.register_buffer("queue_tm", torch.randn(latent_dim, queue_size))
        self.queue_tm = F.normalize(self.queue_tm, dim=0)
        self.register_buffer("queue_t_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_s", torch.randn(latent_dim, queue_size_s))
        self.queue_s = F.normalize(self.queue_s, dim=0)
        self.register_buffer("queue_sj", torch.randn(latent_dim, queue_size_s))
        self.queue_sj = F.normalize(self.queue_sj, dim=0)
        self.register_buffer("queue_sm", torch.randn(latent_dim, queue_size_s))
        self.queue_sm = F.normalize(self.queue_sm, dim=0)
        self.register_buffer("queue_s_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_s_label", torch.zeros(queue_size_s, dtype=torch.long))
        # self.register_buffer("centers", torch.zeros(encoder_args['num_class'], latent_dim))
        # self.register_buffer("centers_j", torch.zeros(encoder_args['num_class'], latent_dim))
        # self.register_buffer("centers_m", torch.zeros(encoder_args['num_class'], latent_dim))
        self.centers = torch.zeros(encoder_args['num_class'], latent_dim).cuda()
        self.centers_j = torch.zeros(encoder_args['num_class'], latent_dim).cuda()
        self.centers_m = torch.zeros(encoder_args['num_class'], latent_dim).cuda()

        self.latent_dim = latent_dim
        self.num_class = encoder_args['num_class']
        self.m = momentum
        self.Temperature = Temperature
        self.topk = topk
        self.queue_size_target = queue_size
        self.queue_size_source = queue_size_s
        self.pretrained_path = pretrained_path
        self.queue_s_label[:] = self.num_class + 1.
        self.context = context
        self.Temperature_s = Temperature_s
        self.maskout = maskout

        self.label_classifier = LabelClassifier(256*2, num_classes=self.num_class)
        self.domain_classifier = DomainClassifier(256*2)
        self.grl = GradientReversalLayer(lambda_val)
        self.lambda_val = lambda_val

        self.Temperature_uda = Temperature_uda
        self.topk_uda = topk_uda
        self.lambda_center = lambda_center

    def load_pretrained_model(self, pretrained_path=None):
        if pretrained_path or self.pretrained_path:
            pretrained_path = pretrained_path if pretrained_path else self.pretrained_path
            self.encoder_q.load_pretrained_model(pretrained_path)
            self.encoder_k.load_pretrained_model(pretrained_path)
            return True
        return False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.fc_q.parameters(), self.fc_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.fc_j_q.parameters(), self.fc_j_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.fc_m_q.parameters(), self.fc_m_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def copy_params(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
        for param_q, param_k in zip(self.fc_q.parameters(), self.fc_k.parameters()):
            param_k.data.copy_(param_q.data)
        for param_q, param_k in zip(self.fc_j_q.parameters(), self.fc_j_k.parameters()):
            param_k.data.copy_(param_q.data)
        for param_q, param_k in zip(self.fc_m_q.parameters(), self.fc_m_k.parameters()):
            param_k.data.copy_(param_q.data)
        print('copy params')

    @torch.no_grad()
    def _dequeue_and_enqueue_target(self, emb_t_k_mlp, emb_tj_k_mlp, emb_tm_k_mlp):
        batch_size = emb_t_k_mlp.shape[0]
        ptr_t = int(self.queue_t_ptr)
        self.queue_t[:, ptr_t:ptr_t + batch_size] = emb_t_k_mlp.T
        self.queue_tj[:, ptr_t:ptr_t + batch_size] = emb_tj_k_mlp.T
        self.queue_tm[:, ptr_t:ptr_t + batch_size] = emb_tm_k_mlp.T
        self.update_ptr_target(batch_size)

    @torch.no_grad()
    def update_ptr_target(self, batch_size):
        assert self.queue_size_target % batch_size == 0, 'queue size should be divisible by batch size'
        self.queue_t_ptr[0] = (self.queue_t_ptr[0] + batch_size) % self.queue_size_target

    @torch.no_grad()
    def _dequeue_and_enqueue_source(self, emb_s_k_mlp, label_s, emb_sj_k_mlp, emb_sm_k_mlp):
        batch_size = emb_s_k_mlp.shape[0]
        ptr_s = int(self.queue_s_ptr)
        self.queue_s[:, ptr_s:ptr_s + batch_size] = emb_s_k_mlp.T
        self.queue_sj[:, ptr_s:ptr_s + batch_size] = emb_sj_k_mlp.T
        self.queue_sm[:, ptr_s:ptr_s + batch_size] = emb_sm_k_mlp.T
        self.queue_s_label[ptr_s:ptr_s + batch_size] = label_s
        self.update_ptr_source(batch_size)

    @torch.no_grad()
    def update_ptr_source(self, batch_size):
        assert self.queue_size_source % batch_size == 0, 'queue size should be divisible by batch size'
        self.queue_s_ptr[0] = (self.queue_s_ptr[0] + batch_size) % self.queue_size_source

    @torch.no_grad()
    def update_centers(self):
        for i in range(self.num_class):
            self.centers[i] = torch.mean(self.queue_s[:, self.queue_s_label == i], dim=1)
            self.centers_j[i] = torch.mean(self.queue_sj[:, self.queue_s_label == i], dim=1)
            self.centers_m[i] = torch.mean(self.queue_sm[:, self.queue_s_label == i], dim=1)

    def forward(self, xs, xs2=None, label_s=None, xt1=None, xt2=None, xte=None,
                nnm=False, uda=False, center=False, nnm_uda=False):
        N, C, T, V, M = xs.shape
        _, emb_s_q, emb_sj_q, emb_sm_q = self.encoder_q(xs, return_jm=True)

        cls_s = self.label_classifier(emb_s_q)
        domain_s = self.domain_classifier(self.grl(emb_s_q))

        if xt1 is None:
            return cls_s, emb_s_q, domain_s
        
        exit(1)
        # the training code will be available after the paper is accepted

        self._dequeue_and_enqueue_target(emb_t_k_mlp, emb_tj_k_mlp, emb_tm_k_mlp)
        self._dequeue_and_enqueue_source(emb_s_k_mlp, label_s, emb_sj_k_mlp, emb_sm_k_mlp)

        return (cls_s, cls_t, ss_t_logits, ss_tj_logits, ss_tm_logits, ss_label_t, ss_te_logits, ss_label_ddm,
                ss_tej_logits, ss_label_ddm_j, ss_tem_logits, ss_label_ddm_m,
                ss_s_logits, ss_sj_logits, ss_sm_logits, ss_label_s,
                domain_s, domain_t,
                ss_t2s_logits, ss_j_t2s_logits, ss_m_t2s_logits, ss_label_t2s)

    def extract_feature(self, xs, xt):
        with torch.no_grad():
            _, emb_s_q = self.encoder_q(xs)
            _, emb_t_q = self.encoder_q(xt)
            cls_s = self.label_classifier(emb_s_q)
            cls_t = self.label_classifier(emb_t_q)
        return cls_s, cls_t, emb_s_q, emb_t_q


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambda_val = ctx.lambda_val
        grad_input = grad_output.neg() * lambda_val
        return grad_input, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_val=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)


class LabelClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=10):
        super(LabelClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.fc(x)


class DomainClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(DomainClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # 两个类别：源域和目标域
        )

    def forward(self, x):
        return self.fc(x)
