from torch.nn import Conv1d

from uclu.me.models.v1 import *


class ConvAttn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvAttn, self).__init__()
        self.conv_kernel = Conv1d(in_channels, out_channels, kernel_size)
        self.word_attn_q = nn.Linear(out_channels, 1)
        self.sent_attn_q = nn.Linear(out_channels, 1)

    def forward(self, rep):  # (bs, tn, dw) -> (bs, dx)
        rep = rep.transpose(2, 1)  # (bs, dw, tn)
        w_conv = self.conv_kernel(rep).transpose(2, 1)  # (bs, tn - pad, dx)
        w_score = self.word_attn_q(w_conv).transpose(2, 1)  # (bs, 1, tn - pad)
        w_probs = F.softmax(w_score, dim=2)  # (bs, 1, tn - pad)
        w_recon = w_probs.matmul(w_conv).squeeze(1)  # (bs, dx)
        return w_recon.mean(dim=0, keepdim=True)


class T1(V1):
    def __init__(self, device, args: UcluArgs):
        super(T1, self).__init__(device, args)

    def define_params(self):
        self.dim_x = self.dim_w // 2
        self.doc_attn = ConvAttn(self.dim_w, self.dim_x, 3)
        self.l_clu = nn.Linear(self.dim_w, self.dim_x)  # (dw, dx)

    def get_lkup(self, docarr: List[Document]):
        ttl_wint = self.assign_tensor([doc.tseq for doc in docarr])
        bdy_wint = self.assign_tensor([doc.bseq for doc in docarr])
        return self.w_embed(ttl_wint), self.w_embed(bdy_wint)

    def get_clu_proj(self):
        return self.l_clu(self.c_embed.weight)

    def get_pc_probs(self, rep, clu):
        pc_score = torch.matmul(rep, clu)  # (bs, nc)
        pc_probs = F.softmax(pc_score, dim=1)  # (bs, nc)
        return pc_probs

    """ runtime """

    def get_que_rep(self, docarr: List[Document], uint2docs: dict):
        uint2idx = dict()
        for doc in docarr:
            uint2idx.setdefault(doc.uint, len(uint2idx))
        idx2urep = [None] * len(uint2idx)
        for uint, idx in uint2idx.items():
            u_docs = uint2docs[uint]
            ut_lkup, ub_lkup = self.get_lkup(u_docs)
            ut_rep = self.doc_attn(ut_lkup)  # (bs, dx)
            ub_rep = self.doc_attn(ub_lkup)  # (bs, dx)
            u_rep = ut_rep + ub_rep
            idx2urep[idx] = u_rep
        for r in idx2urep:
            assert r is not None

        u_rep = [idx2urep[uint2idx[doc.uint]] for doc in docarr]
        u_rep = torch.cat(u_rep, dim=0)
        t_lkup, b_lkup = self.get_lkup(docarr)
        t_rep = self.doc_attn(t_lkup)  # (bs, dx)
        b_rep = self.doc_attn(b_lkup)  # (bs, dx)
        return u_rep, t_rep, b_rep

    def train_step(self, docarr: List[Document], uint2docs: dict, *args, **kwargs):
        u_rep, t_rep, b_rep = self.get_que_rep(docarr, uint2docs)  # (bs, dx)
        q_rep = u_rep + t_rep + b_rep
        c_embed = self.get_clu_proj()  # (nc, dx)
        c_probs = self.get_pc_probs(q_rep, c_embed.transpose(1, 0))  # (bs, nc)
        q_rec = torch.matmul(c_probs, c_embed)  # (bs, dx)

        rec_mut = self.mutual_cos(q_rep, q_rec)
        rec_loss = self.max_margin_loss(rec_mut)
        loss = rec_loss

        loss.backward()
        self.adam.step()
        self.zero_grad()
        return loss

    def predict(self, docarr: List[Document], uint2docs: dict):
        u_rep, t_rep, b_rep = self.get_que_rep(docarr, uint2docs)  # (bs, dx)
        q_rep = u_rep + t_rep + b_rep
        c_embed = self.get_clu_proj()  # (nc, dx)
        pc_probs = self.get_pc_probs(q_rep, c_embed.transpose(1, 0))  # (bs, nc)
        pc_probs = pc_probs.cpu().detach().numpy()
        return np.argmax(pc_probs, axis=1)
