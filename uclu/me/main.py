from typing import List
import numpy as np
from utils import lu, iu, au
from uclu.me import UcluArgs, U
from uclu.me.models import *
from uclu.data.document import Document
from uclu.data.datasets import Sampler
from uclu.me.models.V1 import V1


class Runner:
    def __init__(self, args: UcluArgs):
        self.args = args
        self.epoch_num = args.ep
        self.embed_dim = args.ed
        self.title_pad = args.tpad
        self.body_pad = args.bpad
        self.sampler = Sampler(args.dn)
        self.model = name2m_class[V1.__name__](args)
        self.epoch: int = 0
        self.hist = list()

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
        self.sampler.load(self.embed_dim, self.title_pad, self.body_pad, 1, False)
        self.model.build(self.sampler.w_embed, self.sampler.u_embed, self.sampler.c_embed)
        for e in range(self.epoch_num):
            self.one_epoch()
            if self.should_early_stop():
                return

    def one_epoch(self):
        docarr_batches = list(self.sampler.generate(128, True))
        for bid, docarr in enumerate(docarr_batches):
            self.one_batch(bid, docarr)

    def one_batch(self, bid: int, docarr: List[Document]):
        self.model.train_step(docarr)

    def evaluate(self):
        y_true, y_pred = list(), list()
        for docarr in self.sampler.eval_batches:
            y_pred.extend(self.model.predict(docarr))
            y_true.extend(doc.tag for doc in docarr)
        return au.scores(y_true, y_pred)

    def should_early_stop(self) -> bool:
        scores = self.evaluate()
        value = np.mean(list(scores.values()))
        self.ppp(iu.dumps(scores))
        self.hist.append(value)
        return False


if __name__ == '__main__':
    Runner(UcluArgs().parse_args()).main()
