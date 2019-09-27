from typing import List
import numpy as np
from utils import lu, iu, au, tmu
from uclu.me import UcluArgs, U
from uclu.me.models import *
from uclu.data.document import Document
from uclu.data.datasets import Sampler


class Runner:
    def __init__(self, args: UcluArgs):
        self.args = args
        self.n_epoch = args.ep
        self.d_embed = args.ed
        self.title_pad = args.tpad
        self.body_pad = args.bpad
        self.batch_size = args.bs
        self.sampler = Sampler(args.dn)
        self.model = name2m_class[args.vs](args)
        self.epoch: int = 0
        self.history = list()

        self.logger = None
        args_dict = {k: v for k, v in args.__dict__.items() if k != 'parser' and v is not None}
        if args.lg:
            log_name = au.entries2name(args_dict, exclude={U.gi, U.gp, U.lg}, postfix='.txt')
            self.logger = lu.get_logger(iu.join(args.lg, log_name))
        self.ppp(iu.dumps(args_dict))

    def ppp(self, info: str):
        print(info)
        if self.logger:
            self.logger.info(info)

    def main(self):
        self.sampler.load(self.d_embed, self.title_pad, self.body_pad, topic_ratio=1)
        self.model.build(self.sampler.w_embed, self.sampler.u_embed, self.sampler.c_embed)
        # self.model.cuda(self.model.device)
        for e in range(self.n_epoch):
            self.one_epoch()
            if self.should_early_stop():
                return

    @tmu.stat_time_elapse
    def one_epoch(self):
        docarr_batches = list(self.sampler.generate(self.batch_size, True))
        for bid, docarr in enumerate(docarr_batches):
            self.one_batch(bid, docarr)

    def one_batch(self, bid: int, docarr: List[Document]):
        self.model.train_step(docarr)
        if self.is_lucky(0.01):
            losses = self.model.get_losses(docarr)
            self.ppp(f'losses: {losses}')

    @staticmethod
    def is_lucky(prob: float):
        return np.random.random() < prob

    @tmu.stat_time_elapse
    def evaluate(self):
        y_true, y_pred = list(), list()
        for docarr in self.sampler.eval_batches:
            y_pred.extend(self.model.predict(docarr))
            y_true.extend(doc.tag for doc in docarr)
        return au.scores(y_true, y_pred)

    def should_early_stop(self) -> bool:
        scores = self.evaluate()
        self.ppp(iu.dumps(scores))
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


if __name__ == '__main__':
    Runner(UcluArgs().parse_args()).main()
