from collections import OrderedDict as Od

from tqdm import tqdm

from cqa.mee import *
from cqa.data.datasets import Sampler
from utils.deep.funcs import get_session
from utils import iu, lu, au


class Runner:
    def __init__(self, args: dict):
        self.args = args
        self.gpu_id = args[gi_]
        self.gpu_frac = args[gp_]
        self.data_name = args[dn_]
        self.model_name = args[vs_]
        self.is_full_data = args[fda_]
        self.epoch_num = args[ep_]
        self.early_stop = args[es_]

        self.log_path = args[lg_]
        entries = [(k, v) for k, v in args.items() if v is not None]
        log_name = au.entries2name(entries, exclude={gi_, gp_, lg_}, postfix='.txt')
        log_file = iu.join(self.log_path, log_name)
        self.logger = lu.get_logger(log_file)

        self.word_vec = self.user_vec = None
        self.sampler = self.model = self.sess = None
        self.brk_cnt = 0
        self.best_valid = None
        self.ppp(args)

    def ppp(self, info):
        print(info)
        self.logger.info(info)

    def get_model_class(self):
        from cqa.mee.models import V1, V2, V3
        return {v.__name__: v for v in [V1, V2, V3]}[self.model_name]

    def run(self):
        self.sample()
        self.build()
        self.iterate()

    def sample(self):
        self.sampler = Sampler(self.data_name)
        self.sampler.load(self.is_full_data)
        self.word_vec = self.sampler.d_obj.word_vec
        self.user_vec = self.sampler.d_obj.user_vec

    def build(self):
        self.model = self.get_model_class()(self.args)
        self.model.build(self.word_vec, self.user_vec)
        self.sess = get_session(self.gpu_id, self.gpu_frac, run_init=True, allow_growth=False)

    def iterate(self):
        # wid_range = set(range(len(self.word_vec) + 1))
        # uid_range = set(range(len(self.user_vec)))
        for e in range(self.epoch_num):
            self.ppp('\nepoch:{}'.format(e))
            train_data, train_size = self.sampler.get_train()
            update_pbar, close_pbar = start_pbar(ncols=50, desc='train')
            for bid, (ql, al, ul, vl) in enumerate(train_data):
                # if reach_partition(bid, train_size, 100):
                #     print(bid * 100 // train_size, '%')
                # assert set(np.reshape(al, (-1,))).issubset(wid_range)
                # assert set(ul).issubset(uid_range)
                loss = self.model.train_step(self.sess, ql, al, ul, vl)
                update_pbar(bid, train_size)
                if reach_partition(bid, train_size, 3) or bid == train_size - 1:
                    self.ppp({'kl_loss': loss})
                    if self.should_early_stop():
                        self.ppp('early stop')
                        return
            close_pbar()

    def should_early_stop(self):
        valid_data, valid_size = self.sampler.get_valid()
        valid_mrs = evaluate(self.sess, self.model, valid_data, valid_size, 'valid')
        scores = Od([('brk_cnt', self.brk_cnt), ('valid', valid_mrs.to_dict())])
        if valid_mrs.is_better_than(self.best_valid):
            self.brk_cnt = 0
            self.best_valid = valid_mrs
            test_data, test_size = self.sampler.get_test()
            test_mrs = evaluate(self.sess, self.model, test_data, test_size, 'test')
            scores['test'] = test_mrs.to_dict()
        else:
            self.brk_cnt += 1
        self.ppp(iu.dumps(scores))
        return self.brk_cnt >= self.early_stop


def evaluate(sess, model, data, dnum, desc):
    from cqa.mee.evaluate import MeanRankScores
    update_pbar, close_pbar = start_pbar(ncols=30, desc=desc)
    mrs = MeanRankScores()
    for bid, (ql, al, ul, vl) in enumerate(data):
        pd_ = model.predict(sess, ql, al, ul)
        mrs.append(vl, pd_)
        update_pbar(bid, dnum)
    close_pbar()
    return mrs


def reach_partition(i, i_max, part):
    return i > 0 and i % (i_max // part) == 0


def start_pbar(ncols, desc):
    def f1(i, i_max):
        if reach_partition(i, i_max, step):
            pbar.update(1)

    def f2():
        pbar.close()

    from sys import stdout
    step = 100
    pbar = tqdm(total=step, ncols=ncols, leave=False, desc=desc, file=stdout)
    return f1, f2


if __name__ == '__main__':
    from cqa.mee.grid import get_args

    Runner(get_args()).run()
