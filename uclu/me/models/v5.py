from uclu.me.models.v4 import *


class V5(V4):
    """ 用用户表示重构文档并使其接近 """

    def __init__(self, device, args: UcluArgs):
        super(V5, self).__init__(device, args)

    def forward(self, title_int, body_int, user_int):
        t_rep, b_rep, u_rep = self.get_reps(title_int, body_int, user_int)
        q_rep = t_rep + self.addb * b_rep
        _, qc_rec = self.get_probs_recon(q_rep, self.c_embed.weight, do_recon=True)
        _, qu_rec = self.get_probs_recon(q_rep, self.u_embed.weight, do_recon=True)

        q_qc_mut = self.mutual_cos(q_rep, qc_rec)
        q_qc_loss = self.max_margin_loss(q_qc_mut)
        u_qu_mut = self.mutual_cos(u_rep, qu_rec)
        u_qu_loss = self.max_margin_loss(u_qu_mut)

        extra_loss = u_qu_loss
        return (q_qc_loss + self.addpnt * extra_loss) if self.addpnt > 1e-6 else q_qc_loss
