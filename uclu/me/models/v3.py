from uclu.me.models.v2 import *


class V3(V2):
    """ 加了用户-标题和用户-主体的额外pairwise，效果大涨 """
    def __init__(self, args: UcluArgs):
        super(V3, self).__init__(args)
        self.addpnt: float = args.addpnt

    def get_mean_from_wint(self, wint):
        mask = self.get_mask(wint)  # (bs, tn, 1)
        lkup = self.w_embed(wint)  # (bs, tn, dw)
        mean = self.mean_pooling(lkup, mask)  # (bs, dw)
        return mean

    def get_reps(self, title_int, body_int, user_int):
        u_rep = self.u_embed(user_int)  # (bs, dw)
        t_rep = self.get_mean_from_wint(title_int)  # (bs, dw)
        b_rep = self.get_mean_from_wint(body_int)  # (bs, dw)
        return u_rep, t_rep, b_rep

    def get_doc_rep(self, title_int, body_int, user_int):
        u_rep, t_rep, b_rep = self.get_reps(title_int, body_int, user_int)
        return t_rep + self.addb * b_rep + self.addu * u_rep

    def forward(self, title_int, body_int, user_int):
        u_rep, t_rep, b_rep = self.get_reps(title_int, body_int, user_int)
        qu_rep = t_rep + self.addb * b_rep + self.addu * u_rep

        pc_probs = self.get_pc_probs(qu_rep)  # (bs, nc)
        qu_rec = torch.matmul(pc_probs, self.c_embed.weight)  # (bs, dw)
        rec_mut = self.mutual_cos(qu_rep, qu_rec)
        rec_loss = self.max_margin_loss(rec_mut)

        ut_mut = self.mutual_cos(u_rep, t_rep)
        ut_loss = self.max_margin_loss(ut_mut)
        ub_mut = self.mutual_cos(u_rep, b_rep)
        ub_loss = self.max_margin_loss(ub_mut)

        if self.addpnt > 1e-6:
            loss = rec_loss + self.addpnt * (ut_loss + ub_loss)
        else:
            loss = rec_loss
        return loss

    def get_losses(self, docarr):
        title_int, body_int, user_int = self.get_tensors(docarr)
        u_rep, t_rep, b_rep = self.get_reps(title_int, body_int, user_int)
        qu_rep = t_rep + self.addb * b_rep + self.addu * u_rep
        pc_probs = self.get_pc_probs(qu_rep)  # (bs, nc)
        qu_rec = torch.matmul(pc_probs, self.c_embed.weight)  # (bs, dw)

        rec_mut = self.mutual_cos(qu_rep, qu_rec)
        rec_loss = self.max_margin_loss(rec_mut)
        ut_mut = self.mutual_cos(u_rep, t_rep)
        ut_loss = self.max_margin_loss(ut_mut)
        ub_mut = self.mutual_cos(u_rep, b_rep)
        ub_loss = self.max_margin_loss(ub_mut)

        def detach(t):
            return t.cpu().detach().numpy()

        return list(map(detach, [rec_loss, ut_loss, ub_loss]))
