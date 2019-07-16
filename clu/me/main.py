from typing import List
import numpy as np
import tensorflow.summary as su

from clu.me import C
from clu.me.models import *
from clu.data.datasets import Sampler

from utils import au, iu, lu
from utils.deep.layers import get_session, tf
from utils.doc_utils import Document


class Runner:
    def __init__(self, args: dict):
        self.gid = args[C.gid]
        self.epoch_max = args[C.ep]
        self.embed_dim = args[C.ed]
        self.gpu_id, self.gpu_frac = args[C.gi], args[C.gp]
        self.batch_size, self.neg_num = args[C.bs], args[C.ns]
        self.data_name, self.model_name = args[C.dn], args[C.vs]
        self.use_pad: bool = [False, True][args[C.pad]]
        args = dict((k, v) for k, v in args.items() if v is not None)

        log_path = args.get(C.lg, None)
        self.enable_log = (log_path is not None)
        self.enable_summary = (self.enable_log and True)
        self.enable_save_params = (self.enable_log and False)
        self.logger = self.writer = None
        self.writer_path = self.param_file = None
        self.merge_score = self.score2ph = None
        if self.enable_log:
            log_name = au.entries2name(args, exclude={C.gi, C.gp, C.lg}, postfix='.txt')
            self.logger = lu.get_logger(iu.join(log_path, log_name))
        if self.enable_summary or self.enable_save_params:
            self.writer_path = iu.join(log_path, 'gid={}'.format(self.gid))
            self.param_file = iu.join(self.writer_path, 'model.ckpt')
            iu.mkdir(self.writer_path, rm_prev=True)

        self.model = name2m_class[self.model_name](args)
        self.sampler = Sampler(self.data_name)
        self.history = list()
        self.epoch = 0
        self.ppp(iu.dumps(args))
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
            if self.merge_score is None or self.score2ph is None:
                self.score2ph = {s: tf.placeholder(dtype=tf.float32) for s in au.eval_scores}
                sus = [su.scalar(name=s, tensor=p, family='eval') for s, p in self.score2ph.items()]
                self.merge_score = su.merge(sus)
            p2v = {self.score2ph[s]: s2v[s] for s in self.score2ph.keys()}
            merged = self.model.sess.run(self.merge_score, feed_dict=p2v)
            self.writer.add_summary(merged, global_step=self.epoch)

    def get_writer(self):
        if self.enable_summary:
            self.writer = su.FileWriter(self.writer_path)
            self.writer.add_graph(self.model.sess.graph)

    def run(self):
        self.load_data()
        self.build_model()
        self.iterate_data()

    """ runtime """

    def load_data(self):
        use_tfidf = isinstance(self.model, VAE2)
        self.sampler.load(self.embed_dim, use_pad=self.use_pad, use_tfidf=use_tfidf)
        if self.use_pad:
            self.ppp('use padding length {}'.format(self.sampler.d_obj.seq_len))

    def build_model(self):
        self.model.build(self.sampler.word_embed_init, self.sampler.clu_embed_init)
        self.model.set_session(get_session(self.gpu_id, 0.8, allow_growth=True))

    def one_batch(self, bid: int, docarr: List[Document]):
        def is_lucky(prob: float):
            return np.random.random() < prob

        # assert isinstance(self.model, SBX)
        # for p in self.model.test_parts:
        #     print(p.name)
        #     self.model.run_parts(docarr, p)
        # if bid > 10:
        #     exit()
        # return
        # parts = self.model.run_parts(docarr, self.model.debug_parts)
        # for v, p in zip(self.model.debug_parts, parts):
        #     print(v)
        #     print(p)
        # print('\n\n')
        # if bid > 2:
        #     exit()
        self.model.train_step(docarr, self.epoch, bid)
        self.summary_model(docarr, self.model.merge_loss, step=self.model.global_step)
        if is_lucky(0.1):
            self.summary_model(docarr, self.model.merge_mid, step=self.epoch)

    def one_epoch(self):
        print(f'data:{self.data_name}, epoch:{self.epoch}')
        batches = self.sampler.generate(self.batch_size, self.neg_num, shuffle=True)
        for bid, (pos, negs) in enumerate(batches):
            self.one_batch(bid, pos)
        self.epoch += 1

    def iterate_data(self):
        self.get_writer()
        for e in range(self.epoch_max):
            self.one_epoch()  # train
            s2v = self.evaluate()  # evaluate
            self.summary_score(s2v)
            if self.should_early_stop(s2v):
                return

    def evaluate(self):
        clusters, topics = list(), list()
        for docarr in self.sampler.eval_batches:
            clusters.extend(self.model.predict(docarr))
            topics.extend(d.topic for d in docarr)
        return au.scores(topics, clusters, au.eval_scores)

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
    Runner(C.parse_args()).run()
