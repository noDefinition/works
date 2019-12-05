from uclu.me.models.v2 import *


class Doc2vec(V2):
    def __init__(self, device, args: UcluArgs):
        super(Doc2vec, self).__init__(device, args)
        self.addu: float = args.addu
        self.nll_loss = nn.NLLLoss()

    def build(self, w_init, u_init, c_init, d_init):
        self.define_embeds(w_init, u_init, c_init, d_init)
        self.define_params()
        self.define_optimizer()
        self.cuda(self.device)

    def define_embeds(self, w_init, c_init, u_init, d_init):
        super(Doc2vec, self).define_embeds(w_init, c_init, u_init)
        get_embed = nn.Embedding.from_pretrained
        self.d_embed = get_embed(torch.tensor(d_init), freeze=False).float()

    """ runtime """

    def get_reps(self, ctx_int, tgt_int, usr_int, doc_int):
        c_rep = self.w_embed(ctx_int)  # (bs, tn, dw)
        t_rep = self.w_embed(tgt_int)  # (bs, dw)
        u_rep = self.u_embed(usr_int)  # (bs, dw)
        d_rep = self.d_embed(doc_int)  # (bs, dw)
        return c_rep, t_rep, u_rep, d_rep

    def train_step(self, contexts, targets, uints, dints):
        c_rep, t_rep, u_rep, d_rep = self.get_reps(contexts, targets, uints, dints)
        if self.addu > 1e-6:
            rep_sum = c_rep.sum(axis=1) + d_rep + self.addu * u_rep  # (bs, dw)
            rep_mean = rep_sum / (contexts.size(1) + 2)
        else:
            rep_sum = c_rep.sum(axis=1) + d_rep  # (bs, dw)
            rep_mean = rep_sum / (contexts.size(1) + 1)
        w_score = rep_mean.matmul(self.w_embed.weight.t())  # (bs, nw)
        w_probs = F.softmax(w_score, dim=1)  # (bs, nw)
        loss = self.nll_loss(w_probs, targets)

        loss.backward()
        self.adam.step()
        self.zero_grad()
        return loss
