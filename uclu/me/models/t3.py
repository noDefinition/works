from uclu.me.models.t2 import *


class T3(T2):
    def __init__(self, device, args: UcluArgs):
        super(T3, self).__init__(device, args)
        self.addu: float = args.addu
        self.addb: float = args.addb
        self.addpnt: float = args.addpnt

    def define_params(self):
        self.doc_attn = DocAttn(self.dim_w)

    """ runtime """

    def get_lkup_mask(self, docarr: List[Document]):
        t_wint = self.assign_tensor([doc.tseq for doc in docarr])
        b_wint = self.assign_tensor([doc.bseq for doc in docarr])
        u_int = self.assign_tensor([doc.uint for doc in docarr])
        t_mask = self.get_mask(t_wint, expand_last=False)
        b_mask = self.get_mask(b_wint, expand_last=False)
        t_lkup = self.w_embed(t_wint)
        b_lkup = self.w_embed(b_wint)
        u_lkup = self.u_embed(u_int)
        return t_lkup, t_mask, b_lkup, b_mask, u_lkup

    def recon_attention_with_mask(self, rep1, rep2, mask2):
        # rep1 = (bs, dw),  rep2 = (bs, tn, dw), mask2 = (bs, tn)
        rep1 = rep1.unsqueeze(2)  # (bs, dw, 1)
        score = rep2.matmul(rep1).squeeze(2)  # (bs, tn)
        score += (mask2 - 1) * 1e9
        probs = F.softmax(score, dim=1).unsqueeze(1)  # (bs, 1, tn)
        recon = probs.matmul(rep2).squeeze(1)  # (bs, dw)
        return recon

    def get_que_rep(self, docarr: List[Document], uint2docs: dict):
        u_hist = list()
        for doc in docarr:
            u_docs = uint2docs[doc.uint]
            n_smp = int(np.clip(len(u_docs) * 0.3, 2, 32))
            smp_docs = np.random.choice(u_docs, n_smp)
            ut_lkup, ut_mask, ub_lkup, ub_mask, uu_lkup = self.get_lkup_mask(smp_docs)
            ut_recon = self.recon_attention_with_mask(uu_lkup, ut_lkup, ut_mask)  # (bs, dw)
            ub_recon = self.recon_attention_with_mask(uu_lkup, ub_lkup, ub_mask)  # (bs, dw)
            u_text = ut_recon.mean(dim=0) + self.addb * ub_recon.mean(dim=0)  # (dw, )
            u_hist.append(u_text.unsqueeze(dim=0))
        u_hist = torch.cat(u_hist, dim=0)

        t_lkup, t_mask, b_lkup, b_mask, u_lkup = self.get_lkup_mask(docarr)
        t_rep = self.doc_attn(t_lkup, t_mask)  # (bs, dw)
        b_rep = self.doc_attn(b_lkup, b_mask)  # (bs, dw)
        # print(u_lkup.size(), u_hist.size())
        # print(t_lkup.size(), t_mask.size(), b_lkup.size(), b_mask.size())
        # exit()
        return t_rep, b_rep, u_lkup, u_hist

    def train_step(self, docarr: List[Document], uint2docs: dict, *args, **kwargs):
        t_rep, b_rep, u_lkup, u_hist = self.get_que_rep(docarr, uint2docs)  # (bs, dw)
        q_rep = t_rep + self.addb * b_rep + self.addu * u_lkup

        _, q_rec = self.get_probs_recon(q_rep, self.c_embed.weight, do_recon=True)
        qr_mut = self.mutual_cos(q_rep, q_rec)
        qr_loss = self.max_margin_loss(qr_mut)

        uh_mut = self.mutual_cos(u_lkup, u_hist)
        uh_loss = self.max_margin_loss(uh_mut)

        loss = qr_loss + self.addpnt * uh_loss
        loss.backward()
        self.adam.step()
        self.zero_grad()
        return loss

    def predict(self, docarr: List[Document], uint2docs: dict):
        t_rep, b_rep, u_lkup, u_hist = self.get_que_rep(docarr, uint2docs)  # (bs, dw)
        q_rep = t_rep + self.addb * b_rep + self.addu * u_lkup
        probs = self.get_probs_recon(q_rep, self.c_embed.weight, do_recon=False)
        return probs.cpu().detach().numpy().argmax(axis=1)
