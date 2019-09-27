import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from bert.pytorch.model import BERTLM, BERT
from bert.pytorch.trainer.optim_schedule import ScheduledOptim
from uclu.bert import UcluBertArgs
from uclu.bert.uclu_bert_dataset import UcluBertDataset
from uclu.data.datasets import Sampler, DataSo


class BertTrain:
    def __init__(self, args: UcluBertArgs):
        self.learning_rate: float = args.lr
        self.batch_size: int = args.bs
        self.max_seq_len: int = args.mxl
        self.n_epoch: int = args.ep
        self.n_heads: int = args.nh
        self.n_layers: int = args.ly
        self.d_hidden: int = args.dh

        self.epoch: int = 0
        self.sampler = Sampler(DataSo)
        self.device = torch.device("cuda:%d" % args.gi)

    def load(self):
        self.sampler.load(self.d_hidden, 0, 0, 1)
        self.dataset = UcluBertDataset(self.sampler, self.max_seq_len)
        self.data_loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            collate_fn=UcluBertDataset.collate_fn,
            num_workers=4,
        )

    def build(self):
        vocab_user_size = self.sampler.vocab_size + self.sampler.user_size
        bert = BERT(
            vocab_size=vocab_user_size,
            d_hidden=self.d_hidden,
            n_layers=self.n_layers,
            n_heads=self.n_heads
        )
        self.model = BERTLM(bert=bert, vocab_size=vocab_user_size).to(self.device)
        self.optim = Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        self.schedule = ScheduledOptim(self.optim, d_model=self.d_hidden, n_warmup_steps=10000)
        self.criterion = nn.NLLLoss(ignore_index=0)

    def main(self):
        self.load()
        self.build()
        for epoch in range(self.n_epoch):
            self.one_epoch(epoch)

    def one_epoch(self, epoch: int):
        loss_sum = correct_sum = n_sample = 0
        pbar = tqdm.tqdm(
            iterable=enumerate(self.data_loader),
            desc="EP_%d" % epoch,
            total=len(self.data_loader),
            ncols=60,
        )
        for bid, data in pbar:
            print([len(d) for d in data])
            print([np.array(d).shape for d in data])
            wints, labels, segment, is_next = [d.to(self.device) for d in data]
            # print()
            # print(wints)
            # print(labels)
            # print(segment)
            # print(is_next)
            # if bid > 5:
            exit()
            continue

            # 1. forward the next_sentence_prediction and masked_lm model
            next_sent_output, mask_lm_output = self.model.forward(wints, segment)
            # 2-1. NLL(negative log likelihood) loss of is_next classification result
            next_loss = self.criterion(next_sent_output, is_next)
            # 2-2. NLLLoss of predicting masked token word
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), labels)
            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            loss = next_loss + mask_loss
            # 3. backward and optimization only in train
            self.schedule.zero_grad()
            loss.backward()
            self.schedule.step_and_update_lr()
            # next sentence prediction accuracy
            correct = next_sent_output.argmax(dim=-1).eq(is_next).sum().item()
            correct_sum += correct
            loss_sum += loss.item()
            n_sample += data["is_next"].nelement()
            # post_fix = {
            #     "epoch": epoch, "iter": bid, "loss": loss.item(),
            #     "loss_sum": loss_sum / (epoch + 1), "avg_acc": correct / n_sample * 100,
            # }
            # pbar.write(str(post_fix))
        print('ep:{}, average loss:{:.4f}, acc:{:.2f}'.format(
            epoch, loss_sum / len(pbar), correct_sum * 100.0 / n_sample
        ))


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    BertTrain(UcluBertArgs().parse_args()).main()
