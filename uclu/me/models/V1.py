from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from uclu.me import UcluArgs
from uclu.data.document import Document


# # noinspection PyUnresolvedReferences,PyAttributeOutsideInit
# noinspection PyAttributeOutsideInit
class V1(nn.Module):
    def __init__(self, args: UcluArgs):
        super(V1, self).__init__()
        self.lr = args.lr
        self.device = torch.device('cuda:%d' % args.gi)

    def define_embeds(self, w_init, u_init, c_init):
        self.num_w, self.dim_w = w_init.shape
        self.num_u, self.dim_u = u_init.shape
        self.num_c, self.dim_c = c_init.shape
        get_embed = nn.Embedding.from_pretrained
        self.w_embed = get_embed(torch.tensor(w_init), freeze=False, padding_idx=0)
        self.u_embed = get_embed(torch.tensor(u_init), freeze=False, padding_idx=0)
        self.c_embed = get_embed(torch.tensor(c_init), freeze=False, padding_idx=0)

    def define_params(self):
        self.q_proj = nn.Sequential(nn.Linear(self.dim_w * 2, self.dim_w), nn.ReLU(True))
        # self.q_proj.cuda(self.device)

    def define_optimizer(self):
        self.adam = opt.Adam(params=self.parameters(), lr=self.lr)

    def build(self, w_init, u_init, c_init):
        self.define_embeds(w_init, u_init, c_init)
        self.define_params()
        self.define_optimizer()
        self.cuda(self.device)

    @staticmethod
    def mean_pooling(tensor, mask):
        # (bs, tn, dw) & (bs, tn, dw) -> (bs, dw)
        return tensor.sum(axis=1, keepdim=False) / mask.sum(axis=1, keepdim=True)

    def get_mask(self, tensor):
        return torch.gt(tensor, 0).type('torch.DoubleTensor').cuda(self.device)

    def get_question_rep(self, title_int: List[List[int]], body_int: List[List[int]]):
        title_tensor = torch.tensor(title_int).cuda(self.device)  # (bs, tn)
        body_tensor = torch.tensor(body_int).cuda(self.device)  # (bs, tn)
        title_mask = self.get_mask(title_tensor)  # (bs, tn)
        body_mask = self.get_mask(body_tensor)  # (bs, tn)
        title_lkup = self.w_embed(title_tensor)  # (bs, tn, dw)
        body_lkup = self.w_embed(body_tensor)  # (bs, tn, dw)

        title_mean = self.mean_pooling(title_lkup, title_mask)  # (bs, dw)
        body_mean = self.mean_pooling(body_lkup, body_mask)  # (bs, dw)
        que_rep = 0.3 * title_mean + 0.7 * body_mean
        # que_cat = torch.cat([title_mean, body_mean], dim=1)  # (bs, dw * 2)
        # que_rep = self.q_proj(que_cat)  # (bs, nc)
        return que_rep

    def get_pc_probs(self, que_rep):
        # que_rep = self.get_question_rep(title_int, body_int)
        pc_score = torch.matmul(que_rep, self.c_embed.weight.transpose(1, 0))  # (bs, nc)
        pc_probs = F.softmax(pc_score, dim=1)  # (bs, nc)
        return pc_probs

    @staticmethod
    def mutual_cos(t1, t2):
        t1_e0 = t1.unsqueeze(dim=0)  # (1, bs, dw)
        t2_e1 = t2.unsqueeze(dim=1)  # (bs, 1, dw)
        mut_cos = F.cosine_similarity(t1_e0, t2_e1, dim=2)  # (bs, bs)
        return mut_cos

    def forward(self, title_int: List[List[int]], body_int: List[List[int]], user_int: List[int]):
        # user_tensor = torch.tensor(user_int)  # (bs, tn)
        # user_lkup = self.u_embed(user_tensor)  # (bs, dw)
        que_repre = self.get_question_rep(title_int, body_int)  # (bs, dw)
        pc_probs = self.get_pc_probs(que_repre)  # (bs, nc)
        que_recon = torch.matmul(pc_probs, self.c_embed.weight)  # (bs, dw)

        mut_cos = self.mutual_cos(que_repre, que_recon)
        ones = torch.ones(mut_cos.size(), dtype=torch.double).cuda(self.device)
        eye = torch.eye(mut_cos.size(0), dtype=torch.double).cuda(self.device)
        margin_point_pair = 1 + (ones - eye * 2) * mut_cos
        margin_loss = nn.ReLU()(margin_point_pair).sum()
        # pointwise_cos = F.cosine_similarity(que_repre, que_recon, dim=1)  # (bs,)
        # pointwise_loss = pointwise_cos.sum()
        return margin_loss

    """ runtime """

    def train_step(self, docarr: List[Document]):
        title_int = [doc.tseq for doc in docarr]
        body_int = [doc.bseq for doc in docarr]
        loss = self.forward(title_int, body_int, [])
        # loss.cuda(self.device)
        loss.backward()
        self.adam.step()
        self.zero_grad()
        return loss

    def predict(self, docarr: List[Document]):
        title_int = [doc.tseq for doc in docarr]
        body_int = [doc.bseq for doc in docarr]
        que_rep = self.get_question_rep(title_int, body_int)  # (bs, dw)
        pc_probs = self.get_pc_probs(que_rep)  # (bs, nc)
        pc_probs = pc_probs.cpu().detach().numpy()
        return np.argmax(pc_probs, axis=1)
