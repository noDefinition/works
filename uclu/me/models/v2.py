from uclu.me.models.v1 import *


class V2(V1):
    def __init__(self, args: UcluArgs):
        super(V2, self).__init__(args)
        self.addu: float = args.addu
        self.addb: float = args.addb
        self.pair = args.pair

    def max_margin_loss(self, pairwise):
        n = pairwise.size(0)  # (n, n)
        ones = torch.ones((n, n), dtype=torch.double).cuda(self.device)  # (n, n)
        eye = torch.eye(n, dtype=torch.double).cuda(self.device)  # (n, n)
        pointwise = torch.diag(pairwise).reshape(-1, 1)  # (n, 1)
        margin_point_pair = (ones - eye) - pointwise + pairwise
        margin_max = nn.ReLU(inplace=True)(margin_point_pair)
        return margin_max.sum()

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
        # loss.cuda(self.device)
        loss.backward()
        self.adam.step()
        self.zero_grad()
        return loss

    def predict(self, docarr: List[Document]):
        title_int, body_int, user_int = self.get_tensors(docarr)
        qu_rep = self.get_doc_rep(title_int, body_int, user_int)  # (bs, dw)
        pc_probs = self.get_pc_probs(qu_rep)  # (bs, nc)
        pc_probs = torch.argmax(pc_probs, dim=-1)
        pc_probs = pc_probs.cpu().detach().numpy()
        return pc_probs
