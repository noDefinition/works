from uclu.me.models.v3 import *


class V4(V3):
    """ 额外pairwise的时候考虑用重构做 """

    def __init__(self, device, args: UcluArgs):
        super(V4, self).__init__(device, args)

    def forward(self, title_int, body_int, user_int):
        t_rep, b_rep, u_rep = self.get_reps(title_int, body_int, user_int)
        q_rep = t_rep + self.addb * b_rep
        _, q_rec = self.get_probs_recon(q_rep, self.c_embed.weight, do_recon=True)

        qr_mut = self.mutual_cos(q_rep, q_rec)
        qr_loss = self.max_margin_loss(qr_mut)
        uq_mut = self.mutual_cos(u_rep, q_rec)
        uq_loss = self.max_margin_loss(uq_mut)

        extra_loss = uq_loss
        if self.addpnt > 1e-6:
            return qr_loss + self.addpnt * extra_loss
        else:
            return qr_loss

    def predict(self, docarr: List[Document]):
        t_rep, b_rep, u_rep = self.get_reps(*self.get_tensors(docarr))
        q_rep = t_rep + self.addb * b_rep
        probs = self.get_probs_recon(q_rep, self.c_embed.weight, do_recon=False)
        return torch.argmax(probs, dim=1).cpu().detach().numpy()
