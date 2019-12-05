from uclu.me.models.v2 import *


class V3(V2):
    """ 加了用户-标题和用户-主体的额外pairwise，效果大涨 """

    def __init__(self, device, args: UcluArgs):
        super(V3, self).__init__(device, args)
        self.addpnt: float = args.addpnt

    def get_mean_from_wint(self, wint):
        mask = self.get_mask(wint)  # (bs, tn, 1)
        lkup = self.w_embed(wint)  # (bs, tn, dw)
        mean = self.mean_pooling(lkup, mask)  # (bs, dw)
        return mean

    def get_reps(self, title_int, body_int, user_int):
        t_rep = self.get_mean_from_wint(title_int)  # (bs, dw)
        b_rep = self.get_mean_from_wint(body_int)  # (bs, dw)
        u_rep = self.u_embed(user_int)  # (bs, dw)
        return t_rep, b_rep, u_rep

    def get_doc_rep(self, title_int, body_int, user_int):
        t_rep, b_rep, u_rep = self.get_reps(title_int, body_int, user_int)
        return t_rep + self.addb * b_rep + self.addu * u_rep

    def forward(self, title_int: torch.tensor, body_int: torch.tensor, user_int: torch.tensor):
        t_rep, b_rep, u_rep = self.get_reps(title_int, body_int, user_int)
        q_rep = t_rep + self.addb * b_rep + self.addu * u_rep

        c_probs = self.get_pc_probs(q_rep)  # (bs, nc)
        qr_rec = torch.matmul(c_probs, self.c_embed.weight)  # (bs, dw)
        qr_mut = self.mutual_cos(q_rep, qr_rec)
        qr_loss = self.max_margin_loss(qr_mut)

        ut_mut = self.mutual_cos(u_rep, t_rep)
        ut_loss = self.max_margin_loss(ut_mut)
        ub_mut = self.mutual_cos(u_rep, b_rep)
        ub_loss = self.max_margin_loss(ub_mut)

        if self.addpnt > 1e-6:
            loss = qr_loss + self.addpnt * (ut_loss + self.addb * ub_loss)
        else:
            loss = qr_loss
        return loss
