import numpy as np
import torch
import torch.nn as nn
from torchvision import models


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out



class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * \
                self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


def make_mask(labels, n_cls, mask_negatives):
    # return a one-hot  likt mask. can use scatter instead.
    device = labels.device
    labels = labels.detach().cpu().numpy()
    n_samples = labels.shape[0]
    if mask_negatives:
        mask_multi, target = np.zeros([n_cls, n_samples]), 1.0
    else:
        mask_multi, target = np.ones([n_cls, n_samples]), 0.0

    for c in range(n_cls):
        c_indices = np.where(labels == c)
        mask_multi[c, c_indices] = target

    return torch.tensor(mask_multi).type(torch.long).to(device)


def set_temperature(conditional_strategy, tempering_type, start_temperature, end_temperature, step_count, tempering_step, total_step):
    if conditional_strategy == 'ContraGAN':
        if tempering_type == 'continuous':
            t = start_temperature + step_count * \
                (end_temperature - start_temperature)/total_step
        elif tempering_type == 'discrete':
            tempering_interval = total_step//(tempering_step + 1)
            t = start_temperature + \
                (step_count//tempering_interval) * \
                (end_temperature-start_temperature)/tempering_step
        else:
            t = start_temperature
    else:
        t = 'no'
    return t


class Conditional_Contrastive_loss(torch.nn.Module):
    def __init__(self, batch_size, pos_collected_numerator):
        super(Conditional_Contrastive_loss, self).__init__()
        self.batch_size = batch_size
        self.pos_collected_numerator = pos_collected_numerator
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        # self.device = device

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def remove_diag(self, M):
        h, w = M.shape
        assert h == w, "h and w should be same"
        # mask = np.ones((h, w)) - np.eye(h)
        mask = torch.ones(h, w)-torch.eye(h)
        mask = (mask).type(torch.bool)
        return M[mask].view(h, -1)

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, inst_embed, proxy, negative_mask, labels, temperature, margin):
        # inst_embed: instance feature: l(x) in paper Eq.8
        # proxy: label embeding -> e(y) in Eq.8
        # negative_mask: shape?
        negative_mask = negative_mask.T  # batch first.
        similarity_matrix = self.calculate_similarity_matrix(
            inst_embed, inst_embed)
        instance_zone = torch.exp(
            (self.remove_diag(similarity_matrix) - margin)/temperature)  # 分母第二项

        inst2proxy_positive = torch.exp((self.cosine_similarity(
            inst_embed, proxy) - margin)/temperature)  # 分子和分母第一项
        if self.pos_collected_numerator:
            mask_4_remove_negatives = negative_mask[labels]
            mask_4_remove_negatives = self.remove_diag(mask_4_remove_negatives)
            inst2inst_positives = instance_zone*mask_4_remove_negatives  # 分子第二项

            numerator = inst2proxy_positive + inst2inst_positives.sum(dim=1)
        else:
            numerator = inst2proxy_positive  # no data-to-date, paper Eq.7

        denominator = torch.cat(
            [torch.unsqueeze(inst2proxy_positive, dim=1), instance_zone], dim=1).sum(dim=1)
        criterion = -torch.log(temperature*(numerator/denominator)).mean()
        return criterion


class LossManager(object):
    def __init__(self):
        self.total_loss = None
        self.all_losses = {}

    def add_loss(self, loss, name, weight=1.0, use_loss=True):
        cur_loss = loss * weight
        if use_loss:
            if self.total_loss is not None:
                self.total_loss += cur_loss
            else:
                self.total_loss = cur_loss

        self.all_losses[name] = cur_loss.data.item()

    def items(self):
        return self.all_losses.items()
