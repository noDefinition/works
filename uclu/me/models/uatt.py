from uclu.me.models.v3 import *


# # noinspection PyUnresolvedReferences,PyAttributeOutsideInit
class UAtt(V3):
    # def __init__(self, device, args: UcluArgs):
    #     super(UAtt, self).__init__(device, args)
    #     self.addpnt: float = args.addpnt

    @staticmethod
    def attend_lkup_user(lkup, mask, urep):  # need tensor
        # (bs, tn, dw), (bs, tn, 1), (bs, dw)
        urep_et = urep.unsqueeze(dim=2)  # (bs, dw, 1)
        uw_score = lkup.matmul(urep_et)  # (bs, tn, 1)
        uw_score = uw_score.masked_fill(mask == 0, -1e32)  # (bs, tn, 1)
        uw_probs = F.softmax(uw_score, dim=1).transpose(1, 2)  # (bs, 1, tn)
        uw_recon = uw_probs.matmul(lkup).squeeze(1)  # (bs, dw)
        return uw_recon

    def get_user_recon(self, wint: torch.tensor, uint: torch.tensor):  # need tensor
        urep = self.u_embed(uint)  # (bs, dw)
        lkup = self.w_embed(wint)  # (bs, tn, dw)
        mask = self.get_mask(wint, expand_last=True)  # (bs, tn, 1)
        urec = self.attend_lkup_user(lkup, mask, urep)  # (bs, dw)
        return urec

    def get_reps(self, title_int, body_int, user_int):
        u_rep = self.u_embed(user_int)
        t_rep = self.get_user_recon(title_int, user_int)
        b_rep = self.get_user_recon(body_int, user_int)
        return u_rep, t_rep, b_rep

    # def get_doc_rep(self, title_int, body_int, user_int):
    #     u_rep, t_rep, b_rep = self.get_reps(title_int, body_int, user_int)
    #     return t_rep + self.addb * b_rep + self.addu * u_rep
    # user_tensor = self.assign_tensor(user_int)  # (bs, )
    # user_lkup = self.u_embed(user_tensor)  # (bs, dw)
    # title_wint = self.assign_tensor(title_int)  # (bs, tn)
    # title_mask = self.get_mask(title_wint)  # (bs, tn, 1)
    # title_lkup = self.w_embed(title_wint)  # (bs, tn, dw)
    # ut_recon = self.attend_lkup_user(title_lkup, title_mask, user_lkup)  # (bs, dw)
    # qu_rep = ut_recon
    # if self.addb > 1e-6:
    #     body_wint = self.assign_tensor(body_int)  # (bs, tn)
    #     body_mask = self.get_mask(body_wint)  # (bs, tn, 1)
    #     body_lkup = self.w_embed(body_wint)  # (bs, tn, dw)
    #     ub_recon = self.attend_lkup_user(body_lkup, body_mask, user_lkup)  # (bs, dw)
    #     qu_rep += self.addb * ub_recon
    # return qu_rep

    # def forward(self, title_int: torch.tensor, body_int: torch.tensor, user_int: torch.tensor):
    #     u_rep, t_rep, b_rep = self.get_reps(title_int, body_int, user_int)
    #     qu_rep = t_rep + self.addb * b_rep + self.addu * u_rep
    #
    #     pc_probs = self.get_pc_probs(qu_rep)  # (bs, nc)
    #     qr_rec = torch.matmul(pc_probs, self.c_embed.weight)  # (bs, dw)
    #     qr_mut = self.mutual_cos(qu_rep, qr_rec)
    #     qr_loss = self.max_margin_loss(qr_mut)
    #
    #     ut_mut = self.mutual_cos(u_rep, t_rep)
    #     ut_loss = self.max_margin_loss(ut_mut)
    #     ub_mut = self.mutual_cos(u_rep, b_rep)
    #     ub_loss = self.max_margin_loss(ub_mut)
    #
    #     if self.addpnt > 1e-6:
    #         loss = qr_loss + self.addpnt * (ut_loss + self.addb * ub_loss)
    #         # loss = qr_loss + self.addpnt * (ut_loss + ub_loss)
    #     else:
    #         loss = qr_loss
    #     return loss
