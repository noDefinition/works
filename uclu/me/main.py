from typing import List

import numpy as np
import torch

from uclu.data.datasets import Sampler
from uclu.data.document import Document
from uclu.me import UcluArgs
from uclu.me.models import *
from utils import au
from utils.tune.base_runner import Runner


class UcluRunner(Runner):
    def __init__(self, args: UcluArgs):
        super(UcluRunner, self).__init__(args)
        self.d_embed = args.ed
        self.title_pad = args.tpad
        self.body_pad = args.bpad
        self.batch_size = args.bs
        self.sampler = Sampler(args.dn)
        self.device = torch.device("cuda:%d" % self.gpu_id)
        self.model = name2m_class[args.vs](self.device, args)

    def load(self):
        self.sampler.load(self.d_embed, self.title_pad, self.body_pad, topic_ratio=1)
        self.model.build(self.sampler.w_embed, self.sampler.c_embed, self.sampler.u_embed)

    def run(self):
        for e in range(self.num_epoch):
            self.epoch = e
            self.one_epoch()
            if self.should_early_stop():
                return

    def one_epoch(self):
        docarr_batches = list(self.sampler.generate(self.batch_size, True))
        for bid, docarr in enumerate(docarr_batches):
            self.one_batch(bid, docarr)

    def one_batch(self, bid: int, docarr: List[Document]):
        self.model.train_step(docarr)
        # if self.is_lucky(0.01):
        #     losses = self.model.get_losses(docarr)
        #     self.ppp(f'losses: {losses}')

    @staticmethod
    def is_lucky(prob: float):
        return np.random.random() < prob

    def evaluate(self):
        y_true, y_pred = list(), list()
        for docarr in self.sampler.eval_batches:
            y_pred.extend(self.model.predict(docarr))
            y_true.extend(doc.tag for doc in docarr)
        return au.scores(y_true, y_pred)

    def should_early_stop(self) -> bool:
        scores = self.evaluate()
        self.ppp(scores, json=True)
        value = np.mean(list(scores.values()))
        h = self.history
        h.append(value)
        # if self.enable_save_params and len(h) >= 1 and score > max(h):
        #     self.model.save(self.param_file)
        if len(h) <= 20:
            return False
        early = 'early stop[epoch {}]:'.format(len(h))
        if (len(h) >= 20 and value <= 0.1) or (len(h) >= 50 and value <= 0.2):
            self.ppp(early + 'score too small-t.s.')
            return True
        peak_idx = int(np.argmax(h))
        from_peak = np.array(h)[peak_idx:] - h[peak_idx]
        if len(from_peak) >= 20:
            self.ppp(early + 'no increase for too long-n.i.')
            return True
        if sum(from_peak) <= -1.0:
            self.ppp(early + 'drastic decline-d.d.')
            return True
        return False


class UcluTextUser(UcluRunner):
    def __init__(self, args: UcluArgs):
        super(UcluTextUser, self).__init__(args)

    def load(self):
        super(UcluTextUser, self).load()
        from collections import defaultdict
        self.uint2docs = defaultdict(list)
        for doc in self.sampler.docarr:
            self.uint2docs[doc.uint].append(doc)

    def one_batch(self, bid: int, docarr: List[Document]):
        self.model.train_step(docarr, self.uint2docs)

    def evaluate(self):
        y_true, y_pred = list(), list()
        for docarr in self.sampler.eval_batches:
            y_pred.extend(self.model.predict(docarr, self.uint2docs))
            y_true.extend(doc.tag for doc in docarr)
        return au.scores(y_true, y_pred)


def main():
    from uclu.me.main_d2v import D2vRunner
    args = UcluArgs().parse_args()
    m_cls = name2m_class[args.vs]
    if issubclass(m_cls, T1):
        run_cls = UcluTextUser
    elif issubclass(m_cls, Doc2vec):
        run_cls = D2vRunner
    else:
        run_cls = UcluRunner
    run_cls(args).main()


if __name__ == '__main__':
    main()
