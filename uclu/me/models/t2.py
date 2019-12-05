from uclu.me.models.t1 import *


class DocAttn(nn.Module):
    def __init__(self, out_channels):
        super(DocAttn, self).__init__()
        self.word_attn_q = nn.Linear(out_channels, 1)
        self.sent_attn_q = nn.Linear(out_channels, 1)

    def forward(self, rep, mask, merge_sent: bool = False):
        # rep = (bs, tn, dw), mask = (bs, tn)
        w_score = self.word_attn_q(rep) + (mask.unsqueeze(2) - 1) * 1e9  # (bs, tn, 1)
        w_probs = F.softmax(w_score, dim=1).transpose(2, 1)  # (bs, 1, tn)
        w_recon = w_probs.matmul(rep).squeeze(1)  # (bs, dw)
        if not merge_sent:
            return w_recon
        else:
            s_score = self.sent_attn_q(w_recon)  # (bs, 1)
            s_probs = F.softmax(s_score, dim=1).transpose(1, 0)  # (1, bs)
            s_recon = s_probs.matmul(w_recon)  # (1, dw)
            return s_recon


class T2(T1):
    def __init__(self, device, args: UcluArgs):
        super(T1, self).__init__(device, args)
        # self.addu: float = args.addu
        # self.addb: float = args.addb
        self.addpnt: float = args.addpnt

    def define_params(self):
        self.doc_attn = DocAttn(self.dim_w)

    """ runtime """

    def get_lkup_mask(self, docarr: List[Document]):
        t_wint = self.assign_tensor([doc.tseq for doc in docarr])
        b_wint = self.assign_tensor([doc.bseq for doc in docarr])
        t_mask = self.get_mask(t_wint, expand_last=True)
        b_mask = self.get_mask(b_wint, expand_last=True)
        t_lkup = self.w_embed(t_wint)
        b_lkup = self.w_embed(b_wint)
        return t_lkup, t_mask, b_lkup, b_mask

    def get_que_rep(self, docarr: List[Document], uint2docs: dict):
        uint2idx = dict()
        for doc in docarr:
            uint2idx.setdefault(doc.uint, len(uint2idx))
        idx2urep = [None] * len(uint2idx)
        for uint, idx in uint2idx.items():
            u_docs = uint2docs[uint]
            ut_lkup, ut_mask, ub_lkup, ub_mask = self.get_lkup_mask(u_docs)
            ut_rep = self.doc_attn(ut_lkup, ut_mask, merge_sent=True)  # (1, dw)
            ub_rep = self.doc_attn(ub_lkup, ub_mask, merge_sent=True)  # (1, dw)
            idx2urep[idx] = ut_rep + ub_rep  # (1, dw)
        for r in idx2urep:
            assert r is not None

        u_rep = [idx2urep[uint2idx[doc.uint]] for doc in docarr]
        u_rep = torch.cat(u_rep, dim=0)  # (bs, dw)
        t_lkup, t_mask, b_lkup, b_mask = self.get_lkup_mask(docarr)
        t_rep = self.doc_attn(t_lkup, t_mask)  # (bs, dw)
        b_rep = self.doc_attn(b_lkup, b_mask)  # (bs, dw)
        return u_rep, t_rep, b_rep

    def get_pc_probs(self, rep):
        pc_score = torch.matmul(rep, self.c_embed.weight.transpose(1, 0))  # (bs, nc)
        pc_probs = F.softmax(pc_score, dim=1)  # (bs, nc)
        return pc_probs

    def get_pc_recon(self, rep):
        pc_probs = self.get_pc_probs(rep)  # (bs, nc)
        pc_recon = torch.matmul(pc_probs, self.c_embed.weight)  # (bs, dw)
        return pc_recon

    def train_step(self, docarr: List[Document], uint2docs: dict, *args, **kwargs):
        u_rep, t_rep, b_rep = self.get_que_rep(docarr, uint2docs)  # (bs, dw)
        q_rep = u_rep + t_rep + b_rep
        q_rec = self.get_pc_recon(q_rep)
        rec_mut = self.mutual_cos(q_rep, q_rec)
        rec_loss = self.max_margin_loss(rec_mut)

        t_rec = self.get_pc_recon(t_rep)
        b_rec = self.get_pc_recon(b_rep)
        ut_mut = self.mutual_cos(u_rep, t_rec)
        ub_mut = self.mutual_cos(u_rep, b_rec)
        ut_loss = self.max_margin_loss(ut_mut)
        ub_loss = self.max_margin_loss(ub_mut)

        loss = rec_loss + self.addpnt * (ut_loss + ub_loss)
        loss.backward()
        self.adam.step()
        self.zero_grad()
        return loss

    def predict(self, docarr: List[Document], uint2docs: dict):
        u_rep, t_rep, b_rep = self.get_que_rep(docarr, uint2docs)  # (bs, dw)
        q_rep = u_rep + t_rep + b_rep
        pc_probs = self.get_pc_probs(q_rep)  # (bs, nc)
        pc_probs = pc_probs.cpu().detach().numpy()
        return np.argmax(pc_probs, axis=1)
