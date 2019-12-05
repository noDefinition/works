from typing import List

from bert.pytorch.model.language_model import BERT
from uclu.bert import UcluBertArgs
from uclu.bert.uclu_bert_dataset import UcluBertDataset as UBD
from uclu.me.models.common import *


class SUV1(Common):
    def __init__(self, device, args: UcluBertArgs, bert: BERT):
        super(SUV1, self).__init__(device)
        self.lr = args.lr
        self.clu_num = args.cn
        self.bert = bert
        self.dim_w = bert.d_hidden
        self.relu = nn.ReLU()
        self.criterion = nn.CrossEntropyLoss()

    def define_params(self):
        # self.c_embed = nn.Embedding(self.clu_num, self.dim_w, UBD.PAD).double()
        self.cls_linear = nn.Linear(self.dim_w, self.clu_num).double()

    def define_optimizer(self):
        self.adam = opt.Adam(params=self.parameters(), lr=self.lr)

    # def get_pc_probs(self, rep):
    #     # (bs, dw) & (nc, dw) -> (bs, nc)
    #     pc_score = torch.matmul(rep, self.c_embed.weight.t())  # (bs, nc)
    #     pc_probs = F.softmax(pc_score, dim=1)  # (bs, nc)
    #     return pc_probs

    def forward(self, x_in, x_seg):
        x_out = self.bert(x_in, x_seg)  # (bs, tn, dw)
        x_pool = x_out[:, 0, :].squeeze(1)  # (bs, dw)
        x_pool = x_pool.double()
        pc_score = self.cls_linear(x_pool)
        pc_probs = F.softmax(pc_score, dim=1)  # (bs,)
        # pc_probs = self.get_pc_probs(x_pool)  # (bs, nc)
        return pc_probs

    def get_loss(self, x_in, x_seg, y_true):
        # mask = (x_in > 0).double()  # (bs, tn)
        # x_emb = self.bert.embedding(x_in, x_seg)  # (bs, tn, dw)
        # x_mean = self.mean_pooling(x_emb, mask)  # (bs, dw)
        # att_mask = (x_in > 0).unsqueeze(1).repeat(1, x_in.size(1), 1).unsqueeze(1)
        # x_out = x_emb
        # for transformer in self.bert.transformer_blocks:
        #     x_out = transformer.forward(x_out, att_mask)
        # x_pool = x_out[:, 0, :].squeeze(1)  # (bs, dw)

        # x_out = self.bert(x_in, x_seg)  # (bs, tn, dw)
        # x_pool = x_out[:, 0, :].squeeze(1)  # (bs, dw)
        # x_pool = x_pool.double()
        # pc_probs = self.get_pc_probs(x_pool)  # (bs, nc)
        # x_rec = torch.matmul(pc_probs, self.c_embed.weight)  # (bs, dw)
        # mut_val = self.mutual_cos(x_pool, x_rec)
        # loss = self.max_margin_loss(mut_val)
        # return loss

        x_out = self.bert(x_in, x_seg)  # (bs, tn, dw)
        x_pool = x_out[:, 0, :].squeeze(1)  # (bs, dw)
        x_pool = x_pool.double()
        y_score = self.cls_linear(x_pool)  # (bs, nc)
        loss = self.criterion(y_score, y_true)
        return loss

    """ runtime below """

    def get_tensors(self, docarr: List[Document]):
        wordints = []
        segments = []
        for doc in docarr:
            wints = [UBD.CLS] + doc.tseq + [UBD.EOS] + doc.bseq + [UBD.EOS, doc.uint, UBD.EOS]
            # wints = [UBD.CLS] + doc.tseq + [UBD.EOS, doc.uint, UBD.EOS]
            seg = [1] * (len(doc.tseq) + 2)
            seg += [2] * (len(wints) - len(seg))
            wordints.append(torch.tensor(wints))
            segments.append(torch.tensor(seg))
        wordints = pad_sequence(wordints, batch_first=True, padding_value=UBD.PAD)
        segments = pad_sequence(segments, batch_first=True, padding_value=UBD.PAD)
        wordints = wordints.to(self.device)
        segments = segments.to(self.device)
        return wordints, segments

    def train_step(self, docarr: List[Document]):
        x_in, x_seg = self.get_tensors(docarr)
        y_true = self.assign_tensor([doc.tag for doc in docarr])
        loss = self.get_loss(x_in, x_seg, y_true)
        loss.backward()
        self.adam.step()
        self.zero_grad()
        return loss.item()

    def predict(self, docarr: List[Document]):
        x_in, x_seg = self.get_tensors(docarr)
        x_out = self.bert(x_in, x_seg)  # (bs, tn, dw)
        x_pool = x_out[:, 0, :].squeeze(1)  # (bs, dw)
        x_pool = x_pool.double()
        pc_score = self.cls_linear(x_pool)
        return pc_score.cpu().detach().numpy()
        # pc_probs = self(x_in, x_seg)
        # clu_pred = torch.argmax(pc_probs, dim=-1)
        # clu_pred = clu_pred.cpu().detach().numpy()
        # return clu_pred
