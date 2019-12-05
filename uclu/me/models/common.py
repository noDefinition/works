import torch
import torch.nn as nn
import torch.nn.functional as F


class Common(nn.Module):
    def __init__(self, device, *other_args):
        super(Common, self).__init__()
        self.device = device

    def assign_tensor(self, inputs, dtype=None):
        return torch.tensor(inputs, dtype=dtype).cuda(self.device)

    def get_mask(self, tensor, expand_last=False):
        mask = torch.gt(tensor, 0).cuda(self.device).float()  # (bs, tn)
        if expand_last:
            return mask.unsqueeze(-1)  # (bs, tn, 1)
        else:
            return mask  # (bs, tn)

    def max_margin_loss(self, pairwise):
        n = pairwise.size(0)  # batch size
        ones = torch.ones((n, n), dtype=torch.float).cuda(self.device)  # (n, n)
        eye = torch.eye(n, dtype=torch.float).cuda(self.device)  # (n, n)
        pointwise = torch.diag(pairwise).reshape(-1, 1)  # (n, 1)
        margin_point_pair = (ones - eye) - pointwise + pairwise
        margin_max = margin_point_pair.clamp(min=0)
        return margin_max.mean()

    @staticmethod
    def mutual_cos(t1, t2):
        t1_e0 = t1.unsqueeze(dim=0)  # (1, bs, dw)
        t2_e1 = t2.unsqueeze(dim=1)  # (bs, 1, dw)
        mut_cos = F.cosine_similarity(t1_e0, t2_e1, dim=2)  # (bs, bs)
        return mut_cos

    @staticmethod
    def mean_pooling(rep, mask):
        mask_size = mask.size()
        if len(mask_size) == 3:
            mask = mask.squeeze(2)
        elif len(mask_size) != 2:
            raise ValueError('unsupported size:', mask_size)
        # (bs, tn, dw) & (bs, tn) -> (bs, dw)
        return rep.sum(axis=1, keepdim=False) / mask.sum(axis=1, keepdim=True)

    @staticmethod
    def get_probs_recon(rep, embed, do_recon: bool):
        score = rep.matmul(embed.t())  # (bs, cn)
        probs = F.softmax(score, dim=1)  # (bs, cn)
        if do_recon:
            recon = probs.matmul(embed)  # (bs, dw)
            return probs, recon
        else:
            return probs

    def define_embeds(self, *args, **kwargs):
        pass

    def define_params(self):
        pass

    def define_optimizer(self):
        pass

    def forward(self, *args, **kwargs):
        pass

    """ runtime required below """

    def build(self, *args, **kwargs):
        pass

    def train_step(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass
