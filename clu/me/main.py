from typing import List
import numpy as np
import tensorflow.summary as su

from clu.me import C, CluArgs
from clu.me.models import *
from clu.data.sampler import Sampler

from utils import au, iu, lu
from utils.deep.layers import get_session, tf
from utils.doc_utils import Document


class Runner:
    def __init__(self, args: CluArgs):
        self.gid = args.gid
        self.num_epoch = args.ep
        self.embed_dim = args.ed
        self.gpu_id, self.gpu_frac = args.gi, args.gp
        self.batch_size, self.neg_num = args.bs, args.ns
        self.data_name, self.model_name = args.dn, args.vs
        self.pad_len: int = args.pad
        args_dict = {k: v for k, v in args.__dict__.items() if k != 'parser' and v is not None}

        # args = dict((k, v) for k, v in args.items() if v is not None)
        # log_path = args.get(C.lg, None)
        log_path = args.lg
        self.enable_log = (log_path is not None)
        self.enable_summary = (self.enable_log and True)
        self.enable_save_params = (self.enable_log and False)
        self.logger = self.writer = None
        self.writer_path = self.param_file = None
        self.merge_metric = self.score2ph = None
        if self.enable_log:
            log_name = au.entries2name(args_dict, exclude={C.gi, C.gp, C.lg}, postfix='.txt')
            self.logger = lu.get_logger(iu.join(log_path, log_name))
        if self.enable_summary or self.enable_save_params:
            self.writer_path = iu.join(log_path, 'gid={}'.format(self.gid))
            self.param_file = iu.join(self.writer_path, 'model.ckpt')
            iu.mkdir(self.writer_path, rm_prev=True)

        # args.clear_args()
        self.model = name2m_class[self.model_name](args)
        self.sampler = Sampler(self.data_name)
        self.history = list()
        self.epoch = 0
        self.ppp(iu.dumps(args_dict))
        # np.random.seed(5413 + self.gid)
        # tf.set_random_seed(17256 + self.gid)

    def ppp(self, info):
        print(info)
        if self.enable_log:
            self.logger.info(info)

    def summary_model(self, docarr: List[Document], merge_op, step: int):
        if self.enable_summary:
            merged = self.model.run_merge(docarr, merge=merge_op)
            self.writer.add_summary(merged, global_step=step)

    def summary_score(self, s2v: dict):
        if self.enable_summary:
            if self.merge_metric is None or self.score2ph is None:
                self.score2ph = {s: tf.placeholder(dtype=tf.float32) for s in au.eval_scores}
                s = [su.scalar(name=s, tensor=p, family='eval') for s, p in self.score2ph.items()]
                self.merge_metric = su.merge(s)
            p2v = {self.score2ph[s]: s2v[s] for s in self.score2ph.keys()}
            merged = self.model.sess.run(self.merge_metric, feed_dict=p2v)
            self.writer.add_summary(merged, global_step=self.epoch)

    def get_writer(self):
        if self.enable_summary:
            self.writer = su.FileWriter(self.writer_path)
            self.writer.add_graph(self.model.sess.graph)

    """ runtime """

    def run(self):
        self.load_data()
        self.build_model()
        self.iterate_data()

    def load_data(self):
        use_tfidf = isinstance(self.model, (VAE2, AeSpectral3, AeSoft))
        self.sampler.load(self.embed_dim, pad_len=self.pad_len, use_tfidf=use_tfidf)
        if self.pad_len:
            self.ppp('use padding length {}'.format(self.sampler.d_obj.seq_len))

    def build_model(self):
        self.model.build(self.sampler.word_embed_init, self.sampler.clu_embed_init)
        self.model.set_session(get_session(self.gpu_id, 0.8, allow_growth=True))

    def iterate_data(self):
        self.get_writer()
        for e in range(self.num_epoch):
            self.one_epoch()  # train
            s2v = self.evaluate()  # evaluate
            self.summary_score(s2v)
            if self.should_early_stop(s2v):
                return

    # def one_batch(self, bid: int, pos: List[Document]):
    #     def is_lucky(prob: float):
    #         return np.random.random() < prob
    # 
    #     self.model.train_step(pos, self.epoch, bid)
        # self.summary_model(pos, self.model.merge_batch, step=self.model.global_step)
        # if is_lucky(0.1):
        #     self.summary_model(pos, self.model.merge_some, step=self.epoch)

    def one_epoch(self):
        print(f'data:{self.data_name}, epoch:{self.epoch}')
        batches = list(self.sampler.generate(self.batch_size, self.neg_num, shuffle=True))
        for bid, (pos, negs) in enumerate(batches):
            self.model.train_step(pos, negs, self.epoch, bid)
            # self.one_batch(bid, pos)
        if isinstance(self.model, AeSoft):
            self.model.on_epoch_end(self.epoch, [b[0] for b in batches])
        self.epoch += 1

    def evaluate(self):
        y_pred, y_true = list(), list()
        for docarr in self.sampler.eval_batches:
            y_pred.extend(self.model.predict(docarr))
            y_true.extend(d.topic for d in docarr)
        return au.scores(y_true, y_pred, au.eval_scores)

    # def multi_evaluate(self):
    #     assert isinstance(self.model, AELstm)
    #     clusters, topics = list(), list()
    #     for docarr in self.sampler.eval_batches:
    #         pwc_probs = self.model.run_parts(docarr, self.model.pwc_probs)
    #         clusters.extend(np.argmax(pwc_probs, axis=1).reshape(-1))
    #         topics.extend(d.topic for d in docarr)
    #     name2score = au.scores(topics, clusters, au.eval_scores)
    #     print('    --> pwc_probs: ', iu.dumps(name2score))

    def should_early_stop(self, name2score: dict):
        self.ppp(iu.dumps(name2score))
        score = sum(name2score.values())
        h = self.history
        if self.enable_save_params and len(h) >= 1 and score > max(h):
            self.model.save(self.param_file)
        h.append(score)
        if len(h) <= 30:
            return False
        early = 'early stop[epoch {}]:'.format(len(h))
        if (len(h) >= 5 and score <= 0.3) or (len(h) >= 50 and score <= 0.7):
            self.ppp(early + 'score too small-t.s')
            return True
        peak_idx = int(np.argmax(h))
        from_peak = (np.array(h) - h[peak_idx])[peak_idx:]
        if sum(from_peak) <= -2.0:
            self.ppp(early + 'drastic decline-d.d')
            return True
        if len(from_peak) >= 40:
            self.ppp(early + 'no increase for too long-n.i')
            return True
        return False


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    Runner(CluArgs().parse_args()).run()
