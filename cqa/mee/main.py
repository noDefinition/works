from sys import stdout
from typing import Union

import numpy as np
from tqdm import tqdm

from cqa.data.datasets import DataSo, DataZh, name2d_class
from cqa.mee import K
from cqa.mee.evaluate import MeanRankScores
from cqa.mee.models import B1, CqaBaseline, name2m_class
from utils import Nodes, au, iu, lu, tmu
from utils.deep.funcs import get_session, init_session


class Runner:
    def __init__(self, args: dict):
        full_args = args.copy()
        if args.get(K.lg, None) is not None:
            log_path = args.pop(K.lg)
            entries = [(k, v) for k, v in args.items() if v is not None]
            log_name = au.entries2name(entries, exclude={K.gi, K.gp, K.lg}, postfix='.txt')
            self.logger = lu.get_logger(str(iu.Path(log_path) / log_name))
            self.writer_path = str(iu.Path(log_path) / 'gid={}'.format(args.pop(K.gid)))
            self.param_file = str(iu.Path(self.writer_path) / 'model_param')
            iu.mkdir(self.writer_path)
        else:
            self.logger = self.writer_path = self.param_file = None

        gpu_id, gpu_frac = args.pop(K.gi), args.pop(K.gp)
        self.data_name, self.model_name = args.pop(K.dn), args.pop(K.vs)
        self.epoch_num, self.early_stop = args.pop(K.ep), args.pop(K.es)
        self.is_full_data = args.pop(K.fda)
        self.model_cls = name2m_class[self.model_name]
        self.model_args = args

        self.save_model_params = False
        self.save_model_graph = False

        self.data: Union[DataSo, DataZh] = None
        self.model = None
        self.train_size = self.valid_size = self.test_size = None
        self.brk_cnt = 0
        self.best_valid = None

        self.ppp(iu.dumps(full_args))
        self.ppp(iu.dumps({'writer_path': self.writer_path, 'param_file': self.param_file}))
        self.sess = get_session(gpu_id, gpu_frac, allow_growth=False)
        # self.sess = get_session(gpu_id, gpu_frac, Nodes.is_1702())

    def ppp(self, info):
        print(info)
        if self.logger is not None:
            self.logger.info(info)

    def get_writer(self):
        def do_summary(ql, al, ul, vl):
            nonlocal writer_step
            writer.add_summary(self.model.get_summary(ql, al, ul, vl), writer_step)
            writer_step += 1

        if not self.writer_path:
            return None
        import tensorflow.summary as su
        writer = su.FileWriter(self.writer_path, self.model.sess.graph)
        writer_step = 0
        return do_summary

    def run(self):
        if isinstance(self.model, B1):
            self.sample_data_bert()
            self.build_model_bert()
            self.iterate_data_bert()
        else:
            self.sample_data()
            self.build_model()
            self.iterate_data()

    """ bert """

    def sample_data_bert(self):
        self.data = name2d_class[self.data_name]()
        self.data.load_bert_full()

    def build_model_bert(self):
        self.model = self.model_cls(self.model_args)
        assert isinstance(self.model, B1)
        self.model.build(self.data.user_vec_bert)
        self.model.set_session(self.sess)
        init_session(self.sess)

    def iterate_data_bert(self):
        def eee(desc):
            def f():
                lut = {'valid': self.data.get_valid_qids, 'test': self.data.get_test_qids}
                return self.eval_bert(qids=lut[desc](), desc=desc)

            return f

        assert isinstance(self.model, B1)
        self.get_writer()
        for e in range(self.epoch_num):
            self.ppp('\nepoch:{}'.format(e))
            train_qids = au.shuffle(self.data.get_train_qids())
            train_size = len(train_qids)
            with my_pbar(desc='train', total=train_size, leave=True, ncols=50) as pbar:
                for bid, qid in enumerate(train_qids):
                    al, ul, vl = self.data.get_auv_bert(qid)
                    self.model.train_step(al, ul, vl)
                    pbar.update()
                    if reach_partition(bid, train_size, 3) or bid == train_size - 1:
                        # self.ppp(self.model.get_loss(al, ul, vl))
                        if self.should_early_stop(eval_valid=eee('valid'), eval_test=eee('test')):
                            self.ppp('early stop')
                            return

    def eval_bert(self, qids, desc):
        assert isinstance(self.model, B1)
        print('eval_bert', desc, len(qids))
        with my_pbar(desc=desc, total=len(qids), leave=True, ncols=30) as pbar:
            mrs = MeanRankScores()
            for bid, qid in enumerate(qids):
                al, ul, vl = self.data.get_auv_bert(qid)
                pl = self.model.predict(al, ul)
                mrs.append(vl, pl)
                pbar.update()
        return mrs

    """ normal """

    def sample_data(self):
        self.data = name2d_class[self.data_name]()
        self.data.load(self.is_full_data)
        self.train_size, self.valid_size, self.test_size = self.data.rvs_size()

    def build_model(self):
        word_vec, user_vec = self.data.word_vec, self.data.user_vec
        self.model = self.model_cls(self.model_args)
        if isinstance(self.model, CqaBaseline):
            qid = self.data.get_train_qids()[0]
            ql, al, _, _ = self.data.qid2qauv[qid]
            self.model.len_q = len(ql)
            self.model.len_a = len(al[0])
            print('len(q) %d, len(a) %d' % (self.model.len_q, self.model.len_a))
        self.model.build(word_embed=word_vec, user_embed=user_vec)
        self.model.set_session(self.sess)
        init_session(self.sess)

    def iterate_data(self):
        if self.save_model_graph:
            do_summary = self.get_writer()
        wid_range = set(range(len(self.data.word_vec) + 1))
        uid_range = set(range(len(self.data.user_vec)))
        tmu.check_time('iter_data')
        for e in range(self.epoch_num):
            self.ppp('\nepoch:{}'.format(e))
            train_data = self.data.gen_train(shuffle=True)
            update_pbar, close_pbar = start_pbar(ncols=50, desc='train')
            for bid, (ql, al, ul, vl) in enumerate(train_data):
                assert set(np.reshape(al, (-1,))).issubset(wid_range)
                assert set(ul).issubset(uid_range)
                self.model.train_step(ql, al, ul, vl, epoch=e)
                update_pbar(bid, self.train_size)
                if reach_partition(bid, self.train_size, 3) or bid == self.train_size - 1:
                    # if bid == 10 or \
                #         reach_partition(bid, self.train_size, 3) or bid == self.train_size - 1:
                #     dic['elapse'] = tmu.check_time('iter_start')
                    self.ppp(self.model.get_loss(ql, al, ul, vl))
                    self.ppp(tmu.check_time('iter_data'))
                    if self.should_early_stop(
                            eval_valid=lambda: self.evaluate_qauv(
                                self.data.gen_valid(), self.valid_size, 'valid'),
                            eval_test=lambda: self.evaluate_qauv(
                                self.data.gen_test(), self.test_size, 'test')
                    ):
                        self.ppp('early stop')
                        return
            close_pbar()
            exit() if not self.is_full_data else None

    def evaluate_qauv(self, qauv_list, dnum, desc):
        update_pbar, close_pbar = start_pbar(ncols=30, desc=desc)
        mrs = MeanRankScores()
        for idx, (ql, al, ul, vl) in enumerate(qauv_list):
            pl = self.model.predict(ql, al, ul)
            mrs.append(vl, pl)
            update_pbar(idx, dnum)
        close_pbar()
        return mrs

    """ eval func """

    def should_early_stop(self, eval_valid, eval_test):
        valid_mrs = eval_valid()
        scores = dict()
        scores.update({'v_' + k: v for k, v in valid_mrs.to_dict().items()})
        # for k, v in valid_mrs.to_dict().items():
        #     scores['v_' + k] = v
        if valid_mrs.is_better_than(self.best_valid):
            self.brk_cnt = 0
            self.best_valid = valid_mrs
            test_mrs = eval_test()
            scores.update({'t_' + k: v for k, v in test_mrs.to_dict().items()})
            # for k, v in test_mrs.to_dict().items():
            #     scores['t_' + k] = v
            if self.save_model_params:
                self.model.save(self.param_file)
        else:
            self.brk_cnt += 1
        scores['brk_cnt'] = self.brk_cnt
        self.ppp(iu.dumps(scores))
        return self.brk_cnt >= self.early_stop


def my_pbar(desc: str, total: int, leave: bool, ncols: int):
    return tqdm(desc=desc, total=total, leave=leave, file=stdout, ncols=ncols, mininterval=10)


def reach_partition(i, i_max, part):
    return i > 0 and i % (i_max // part) == 0


def start_pbar(ncols, desc):
    def update_pbar(i, i_max):
        nonlocal i_prev
        if reach_partition(i, i_max, step) and i != i_prev:
            i_prev = i
            pbar.update(1)

    def close_pbar():
        pbar.close()

    step = 100
    i_prev = None
    pbar = tqdm(total=step, ncols=ncols, leave=False, desc=desc, file=stdout)
    return update_pbar, close_pbar


if __name__ == '__main__':
    Runner(K.parse_args()).run()
