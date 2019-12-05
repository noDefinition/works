from typing import List

from bert.pytorch.model.language_model import BERT
from uclu.bert import UcluBertArgs
from uclu.bert.uclu_bert_dataset import UcluBertDataset as UBD
from uclu.me.models.common import *


class B1(Common):
    def __init__(self, device, args: UcluBertArgs, bert: BERT):
        super(B1, self).__init__(device)
        self.lr = args.lr
        self.clu_num = args.cn
        self.bert = bert
        self.dim_w = bert.d_hidden
        self.bert.double()

    def define_params(self):
        self.c_embed = nn.Embedding(self.clu_num, self.dim_w).double()

    def define_optimizer(self):
        self.adam = opt.Adam(params=self.parameters(), lr=self.lr)

    def get_pc_probs(self, rep):
        # (bs, dw) & (nc, dw) -> (bs, nc)
        pc_score = torch.matmul(rep, self.c_embed.weight.t())  # (bs, nc)
        pc_probs = F.softmax(pc_score, dim=1)  # (bs, nc)
        return pc_probs

    def get_pooled(self, x_in, x_seg):
        x_out = self.bert(x_in, x_seg)  # (bs, tn, dw)
        x_pool = x_out[:, 0, :].squeeze(1)  # (bs, dw)
        return x_pool  # (bs, dw)

    def forward(self, x_in, x_seg):
        x_pool = self.get_pooled(x_in, x_seg)  # (bs, dw)
        pc_probs = self.get_pc_probs(x_pool)  # (bs, nc)
        return pc_probs

    def get_loss_pre(self, x_in, x_seg):
        x_emb = self.bert.embedding(x_in, x_seg)  # (bs, tn, dw)
        mask = (x_in > 0).unsqueeze(1).repeat(1, x_in.size(1), 1).unsqueeze(1)
        x_out = x_emb
        for transformer in self.bert.transformer_blocks:
            x_out = transformer.forward(x_out, mask)
        x_mean = self.mean_pooling(x_emb, (x_in > 0))  # (bs, dw)
        x_pool = x_out[:, 0, :].squeeze(1)  # (bs, dw)
        mut_val = self.mutual_cos(x_mean, x_pool)  # (bs, bs)
        loss = self.max_margin_loss(mut_val)
        return loss

    def get_loss(self, x_in, x_seg):
        x_pool = self.get_pooled(x_in, x_seg)
        pc_probs = self.get_pc_probs(x_pool)  # (bs, nc)
        x_rec = torch.matmul(pc_probs, self.c_embed.weight)  # (bs, dw)
        mut_val = self.mutual_cos(x_pool, x_rec)
        loss = self.max_margin_loss(mut_val)
        return loss

    """ runtime below """

    def get_tensors(self, docarr: List[Document]):
        wordints = []
        segments = []
        for doc in docarr:
            wints = [UBD.CLS] + doc.tseq + [UBD.EOS] + doc.bseq + [UBD.EOS, doc.uint, UBD.EOS]
            seg = [1] * (len(doc.tseq) + 2)
            seg += [2] * (len(wints) - len(seg))
            wordints.append(torch.tensor(wints))
            segments.append(torch.tensor(seg))
        wordints = pad_sequence(wordints, batch_first=True, padding_value=UBD.PAD)
        segments = pad_sequence(segments, batch_first=True, padding_value=UBD.PAD)
        wordints = wordints.to(self.device)
        segments = segments.to(self.device)
        return wordints, segments

    def train_step(self, docarr: List[Document], epoch: int, *args, **kwargs):
        x_in, x_seg = self.get_tensors(docarr)
        if epoch <= 4:
            loss = self.get_loss_pre(x_in, x_seg)
        else:
            loss = self.get_loss(x_in, x_seg)
        loss.backward()
        self.adam.step()
        self.zero_grad()
        return loss.item()

    def get_docarr_pooled(self, docarr: List[Document]):
        x_in, x_seg = self.get_tensors(docarr)
        ret = self.get_pooled(x_in, x_seg)
        return ret.cpu().detach().numpy()

    def predict(self, docarr: List[Document]):
        x_in, x_seg = self.get_tensors(docarr)
        pc_probs = self(x_in, x_seg)
        ret = torch.argmax(pc_probs, dim=-1)
        return ret.cpu().detach().numpy()
