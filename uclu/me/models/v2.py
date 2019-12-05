from uclu.me.models.v1 import *


class V2(V1):
    def __init__(self, device, args: UcluArgs):
        super(V2, self).__init__(device, args)
        self.addu: float = args.addu
        self.addb: float = args.addb
        self.pair: int = args.pair

    def get_que_rep(self, title_int, body_int):
        title_mask = self.get_mask(title_int)  # (bs, tn, 1)
        title_lkup = self.w_embed(title_int)  # (bs, tn, dw)
        title_mean = self.mean_pooling(title_lkup, title_mask)  # (bs, dw)
        if not self.addb > 1e-6:
            que_rep = title_mean
        else:
            body_mask = self.get_mask(body_int)  # (bs, tn, 1)
            body_lkup = self.w_embed(body_int)  # (bs, tn, dw)
            body_mean = self.mean_pooling(body_lkup, body_mask)  # (bs, dw)
            que_rep = 0.3 * title_mean + 0.7 * body_mean
            # que_cat = torch.cat([title_mean, body_mean], dim=1)  # (bs, dw * 2)
            # que_rep = self.q_proj(que_cat)  # (bs, nc)
        return que_rep

    def get_doc_rep(self, title_int, body_int, user_int):
        q_rep = self.get_que_rep(title_int, body_int)
        if not self.addu > 1e-6:
            return q_rep
        user_lkup = self.u_embed(user_int)  # (bs, dw)
        # qu_rep = torch.cat([q_rep, user_lkup], dim=1)  # (bs, dw * 2)
        # qu_rep = self.q_proj(qu_rep)
        qu_rep = q_rep + self.addu * user_lkup
        return qu_rep

    def get_pc_probs(self, rep):
        pc_score = torch.matmul(rep, self.c_embed.weight.transpose(1, 0))  # (bs, nc)
        pc_probs = F.softmax(pc_score, dim=1)  # (bs, nc)
        return pc_probs

    def forward(self, title_int, body_int, user_int):
        qu_rep = self.get_doc_rep(title_int, body_int, user_int)  # (bs, dw)
        pc_probs = self.get_pc_probs(qu_rep)  # (bs, nc)
        qu_rec = torch.matmul(pc_probs, self.c_embed.weight)  # (bs, dw)
        if self.pair == 0:
            mut_val = self.mutual_cos(qu_rep, qu_rec)
        elif self.pair == 1:
            mut_val = torch.matmul(qu_rep, qu_rec.t()) / np.sqrt(self.dim_w)
        else:
            raise ValueError('No fucking pair value:', self.pair)
        loss = self.max_margin_loss(mut_val)
        return loss

    """ runtime """

    def get_tensors(self, docarr):
        title_int = self.assign_tensor([doc.tseq for doc in docarr])
        body_int = self.assign_tensor([doc.bseq for doc in docarr])
        user_int = self.assign_tensor([doc.uint for doc in docarr])
        return title_int, body_int, user_int

    def train_step(self, docarr: List[Document]):
        title_int, body_int, user_int = self.get_tensors(docarr)
        loss = self.forward(title_int, body_int, user_int)
        loss.backward()
        self.adam.step()
        self.zero_grad()
        return loss.item()

    def predict(self, docarr: List[Document]):
        title_int, body_int, user_int = self.get_tensors(docarr)
        q_rep = self.get_doc_rep(title_int, body_int, user_int)  # (bs, dw)
        probs = self.get_pc_probs(q_rep)  # (bs, nc)
        preds = torch.argmax(probs, dim=-1)
        preds = preds.cpu().detach().numpy()
        return preds
