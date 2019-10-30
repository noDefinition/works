from typing import List

import numpy as np
import torch
import tqdm

from bert.pytorch.model.language_model import BERT
from uclu.bert import UB, UcluBertArgs
from uclu.bert.models import name2m_class
from uclu.data.datasets import Sampler, name2d_class
from uclu.data.document import Document
from utils import au, iu, tmu
from utils.tune.base_runner import Runner


class BertTrain(Runner):
    def __init__(self, args: UcluBertArgs):
        super(BertTrain, self).__init__(args)
        self.title_pad = args.tpad
        self.body_pad = args.bpad
        self.pre_train = args.ptn
        self.model_class = name2m_class[args.vs]
        self.data_class = name2d_class[args.dn]
        self.sampler = Sampler(self.data_class)
        self.device = torch.device("cuda:%d" % self.gpu_id)
        self.bert_param_file = './L2_H64_A8.torch'

    def build(self):
        self.bert: BERT = torch.load(self.bert_param_file)
        self.model = self.model_class(self.device, self.args, self.bert)
        self.model.build()
        self.sampler.load(self.bert.d_hidden, self.title_pad, self.body_pad, 1)
        tag_int_arr = au.reindex([doc.tag for doc in self.sampler.docarr])
        for doc, tag_int in zip(self.sampler.docarr, tag_int_arr):
            doc.uint += self.sampler.vocab_size
            doc.tag = tag_int

    def run(self):
        for e in range(self.num_epoch):
            self.epoch = e
            self.one_epoch()
            if e == 4:
                self.kmeans_pooled()

    def one_epoch(self):
        batches = list(self.sampler.generate(self.batch_size, shuffle=True))
        pbar = tqdm.tqdm(
            iterable=enumerate(batches),
            desc="epoch_%d" % self.epoch,
            total=len(batches),
            miniters=100,
            ncols=60,
        )
        for bid, docarr in pbar:
            self.one_batch(bid, docarr)
            if self.epoch > 4 and bid > 0 and bid % (len(batches) // 2) == 0:
                if self.should_early_stop():
                    exit()
                # self.ppp(self.evaluate_classify(), json=True)

    def one_batch(self, bid: int, docarr: List[Document]):
        return self.model.train_step(docarr, self.epoch)

    def kmeans_pooled(self):
        from clu.baselines.kmeans import fit_kmeans
        batches = list(self.sampler.generate(self.batch_size, shuffle=True))
        all_pooled = [self.model.get_docarr_pooled(docarr) for docarr in batches]
        all_pooled = np.concatenate(all_pooled, axis=0)
        print('all_pooled.shape', all_pooled.shape)
        kmeans = fit_kmeans(all_pooled, self.args.cn, max_iter=200, n_jobs=12)
        print('kmeans fit over')
        clu_pred = kmeans.predict(all_pooled)
        clu_true = [doc.tag for docarr in batches for doc in docarr]
        print(au.scores(clu_true, clu_pred))
        clu_init = torch.tensor(kmeans.cluster_centers_)
        self.model.c_embed.weight.data.copy_(clu_init)

    def evaluate_cluster(self) -> dict:
        y_true, y_pred = list(), list()
        for docarr in self.sampler.eval_batches:
            y_pred.extend(self.model.predict(docarr))
            y_true.extend(doc.tag for doc in docarr)
        return au.scores(y_true, y_pred)

    def should_early_stop(self) -> bool:
        scores = self.evaluate_cluster()
        self.ppp(scores, json=True)
        value = np.mean(list(scores.values()))
        h = self.history
        h.append(value)
        if len(h) > 20:
            early = 'early stop[epoch {}]:'.format(len(h))
            peak_idx = int(np.argmax(h))
            from_peak = np.array(h)[peak_idx:] - h[peak_idx]
            if (len(h) >= 20 and value <= 0.1) or (len(h) >= 50 and value <= 0.2):
                self.ppp(early + 'score too small-t.s.')
            elif len(from_peak) >= 20:
                self.ppp(early + 'no increase for too long-n.i.')
            elif sum(from_peak) <= -1.0:
                self.ppp(early + 'drastic decline-d.d.')
            else:
                return False
            return True
        return False

    # def evaluate_classify(self) -> dict:
    #     y_true, y_pred = list(), list()
    #     total_sample = correct_sum = 0
    #     for docarr in self.sampler.eval_batches:
    #         pc_score = self.model.predict(docarr)
    #         y_pred = pc_score.argmax(axis=1)
    #         y_true = np.array([doc.tag for doc in docarr])
    #         total_sample += len(docarr)
    #         correct_sum += np.sum(y_pred == y_true)
    #     acc = correct_sum / total_sample
    #     return {'ACC': round(acc, 4)}


if __name__ == '__main__':
    BertTrain(UcluBertArgs().parse_args()).main()
