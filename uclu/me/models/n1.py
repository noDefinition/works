from uclu.me.models.v2 import *


class N1(V2):
    def __init__(self, device, args: UcluArgs):
        super(N1, self).__init__(device, args)
        self.addpnt: float = args.addpnt
        self.criterion = nn.CrossEntropyLoss()

    def get_que_rep(self, title_int, body_int):
        title_mask = self.get_mask(title_int)  # (bs, tn, 1)
        title_lkup = self.w_embed(title_int)  # (bs, tn, dw)
        title_mean = self.mean_pooling(title_lkup, title_mask)  # (bs, dw)
        if self.addb < 1e-6:
            return title_mean
        body_mask = self.get_mask(body_int)  # (bs, tn, 1)
        body_lkup = self.w_embed(body_int)  # (bs, tn, dw)
        body_mean = self.mean_pooling(body_lkup, body_mask)  # (bs, dw)
        return title_mean + self.addb * body_mean

    def get_doc_rep(self, title_int, body_int, user_int):
        return self.get_que_rep(title_int, body_int)

    def forward(self, title_int, body_int, user_int):
        q_rep = self.get_que_rep(title_int, body_int)  # (bs, dw)
        pc_probs = self.get_pc_probs(q_rep)  # (bs, nc)
        q_rec = pc_probs.matmul(self.c_embed.weight)  # (bs, dw)
        mut_cos = self.mutual_cos(q_rep, q_rec)  # (bs, bs)
        mut_loss = self.max_margin_loss(mut_cos)

        # u_rep = self.u_embed(user_int)  # (bs, dw)
        # uq_mut = self.mutual_cos(u_rep, q_rec)  # (bs, bs)
        # uq_loss = self.max_margin_loss(uq_mut)

        qu_score = q_rep.matmul(self.u_embed.weight.t())  # (bs, nu)
        isu_loss = self.criterion(qu_score, user_int)
        return mut_loss + self.addu * isu_loss  # + self.addpnt * uq_loss
