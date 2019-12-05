import numpy as np
import torch
import torch.nn as nn
import tqdm

from bert.pytorch.model.language_model import BERT, BERTLM
from uclu.bert import UB, UcluBertArgs
from uclu.bert.uclu_bert_dataset import UcluBertDataset
from uclu.data.datasets import DataSo, Sampler, name2d_class
from utils import au, iu, lu
from utils.tune.base_runner import Runner


class BertPreTrain(Runner):
    def __init__(self, args: UcluBertArgs):
        super(BertPreTrain, self).__init__(args)
        self.learning_rate: float = args.lr
        self.max_seq_len: int = args.mxl
        self.n_layers: int = args.ly
        self.d_hidden: int = args.dh
        self.n_heads: int = args.nh
        self.dropout: float = args.do
        self.data_class = name2d_class[args.dn]
        self.sampler = Sampler(self.data_class)
        self.device = torch.device("cuda:%d" % self.gpu_id)

    def load(self):
        max_text_len = 50
        self.sampler.load(self.d_hidden, 0, 0, 1)
        for doc in self.sampler.docarr:
            doc.split_texts(max_text_len)
            doc.all_wints.append([doc.uint + self.sampler.vocab_size])
        self.dataset = UcluBertDataset(
            sampler=self.sampler,
            max_text_len=max_text_len,
        )
        self.data_loader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            collate_fn=UcluBertDataset.collate_fn,
            num_workers=4,
        )

    def build(self):
        vocab_user_size = self.sampler.vocab_size + self.sampler.user_size
        print('build: total vocab size', vocab_user_size)
        self.bert = BERT(
            vocab_size=vocab_user_size,
            d_hidden=self.d_hidden,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            dropout=self.dropout,
        ).to(self.device)
        self.bert_lm = BERTLM(
            bert=self.bert,
            vocab_size=vocab_user_size,
            learning_rate=self.learning_rate,
        ).to(self.device)

    def run(self):
        epoch = 0
        pbar = tqdm.tqdm(
            iterable=enumerate(self.data_loader),
            desc="EP_%d" % epoch,
            total=len(self.data_loader),
            ncols=60,
        )
        batch_cnt = loss_sum = correct_sum = n_sample = 0
        for bid, data in pbar:
            # print(bid, [d.size() for d in data])
            # if bid > 100: exit()
            # continue
            # for a, b, c, d in zip(*data):
            #     print(a)
            #     print(b)
            #     print(c)
            #     print(d)
            # torch.save(self.bert, './fuck')
            # exit()
            wints, labels, segment, is_next = [d.to(self.device) for d in data]
            correct_cnt, loss_val = self.bert_lm.train_step(wints, labels, segment, is_next)
            batch_cnt += 1
            loss_sum += loss_val
            correct_sum += correct_cnt
            n_sample += is_next.nelement()
            if bid % 300 == 0:
                self.ppp(iu.dumps(dict(
                    ep=epoch,
                    loss=round(loss_sum / batch_cnt, 4),
                    acc=round(correct_sum / n_sample, 4),
                )))
                # print('    ep:{}, avg loss:{:.4f}, acc:{:.2f}'.format(
                #     epoch, loss_sum / batch_cnt, correct_sum * 100.0 / n_sample
                # ))
                batch_cnt = loss_sum = correct_sum = n_sample = 0
        param_file = iu.join(self.log_path, self.log_name + '.torch')
        torch.save(self.bert, param_file)
        # torch.save(self.bert.state_dict(), param_file)


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    BertPreTrain(UcluBertArgs().parse_args()).main()
