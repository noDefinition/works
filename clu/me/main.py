import numpy as np
import tensorflow.summary as su
from clu.me import C
from clu.me.gen1 import name2m_class, N1, VAE1
from clu.data.datasets import Sampler
from utils import au, iu, lu, Nodes
from utils.deep.layers import get_session, tf
from typing import List
from utils.doc_utils import Document


class Runner:
    def __init__(self, args: dict):
        # self.args = args
        self.gid = args[C.gid]
        self.epoch_max = args[C.ep]
        self.embed_dim = args[C.ed]
        self.gpu_id, self.gpu_frac = args[C.gi], args[C.gp]
        self.batch_size, self.neg_num = args[C.bs], args[C.ns]
        self.data_name, self.model_name = args[C.dn], args[C.vs]
        self.use_pad: bool = [False, True][args.get(C.pad)]

        log_path = args.get(C.lg, None)
        self.enable_log = (log_path is not None)
        self.enable_summary = (self.enable_log and True)
        self.enable_save_params = (self.enable_log and False)
        self.logger = self.writer = None
        self.writer_path = self.param_file = None
        self.merge_score = self.score2ph = None
        if self.enable_log:
            entries = [(k, v) for k, v in args.items() if v is not None]
            log_name = au.entries2name(entries, exclude={C.gi, C.gp, C.lg}, postfix='.txt')
            self.logger = lu.get_logger(iu.join(log_path, log_name))
            if self.enable_summary or self.enable_save_params:
                self.writer_path = iu.join(log_path, 'gid={}'.format(self.gid))
                self.param_file = iu.join(self.writer_path, 'model.ckpt')
                iu.mkdir(self.writer_path, rm_prev=True)

        for k, v in list(args.items()):
            if v is None:
                args.pop(k)
        self.model = name2m_class[self.model_name](args)
        self.sampler = Sampler(self.data_name)
        self.history = list()
        self.epoch = 0
        self.global_step = 0
        self.ppp(iu.dumps(args))
        np.random.seed(5413 + self.gid)
        tf.set_random_seed(17256 + self.gid)

    def ppp(self, info):
        print(info)
        if self.enable_log:
            self.logger.info(info)

    def summary_model(self, p_seq: List[List[int]], merge_op, step: int):
        if self.enable_summary:
            merged = self.model.run_merge(p_seq, merge=merge_op)
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

    def load_data(self):
        self.sampler.load(self.embed_dim, use_pad=self.use_pad)
        if self.use_pad:
            self.ppp('use padding length {}'.format(self.sampler.d_obj.seq_len))

    def build_model(self):
        self.model.build(self.sampler.word_embed_init, self.sampler.clu_embed_init)
        self.model.set_session(get_session(self.gpu_id, 0.6, allow_growth=False))

    def one_batch(self, bid: int, docarr: List[Document]):
        def is_lucky(prob: float):
            return np.random.random() < prob

        self.global_step += 1
        p_seq = self.get_docarr_feature(docarr)
        self.model.train_step(p_seq, self.epoch, bid)
        self.summary_model(p_seq, self.model.merge_loss, step=self.global_step)
        if is_lucky(0.05):
            self.summary_model(p_seq, self.model.merge_mid, step=self.epoch)

    def one_epoch(self):
        self.epoch += 1
        print('data:{}, epoch:{}'.format(self.data_name, self.epoch))
        batches = self.sampler.generate(self.batch_size, self.neg_num, shuffle=True)
        for bid, (pos, negs) in enumerate(batches):
            self.one_batch(bid, pos)

    def iterate_data(self):
        assert isinstance(self.model, N1) or isinstance(self.model, VAE1)
        self.get_writer()
        for e in range(self.epoch_max):
            self.one_epoch()  # train
            s2v = self.evaluate()  # evaluate
            self.summary_score(s2v)
            if self.should_early_stop(s2v):
                return
            # print('data:{}, epoch:{}'.format(self.data_name, self.epoch))
            # """ train """
            # batches = list(self.sampler.shuffle_generate(self.batch_size, self.neg_size))
            # data_size = len(batches)
            # for b, (pos, negs) in enumerate(batches):
            #     # fd = self.model.get_fd_by_batch(pos, negs)
            #     p_seq = [doc.tokenids for doc in pos]
            #     self.model.train_step(p_seq, self.epoch, b)
            #     self.summary_details(p_seq, self.model.merge_loss, step=self.global_step)
            #     if is_lucky(0.05):
            #         # if b % (data_size // 8) == 0:
            #         self.summary_details(p_seq, self.model.merge_mid, step=self.epoch)

    def evaluate(self):
        assert isinstance(self.model, N1) or isinstance(self.model, VAE1)
        clusters, topics = list(), list()
        for docarr in self.sampler.eval_batches:
            p_seq = self.get_docarr_feature(docarr)
            clusters.extend(self.model.predict(p_seq))
            topics.extend(d.topic for d in docarr)
        return au.scores(topics, clusters, au.eval_scores)

    def should_early_stop(self, name2score: dict):
        self.ppp(iu.dumps(name2score))
        score = sum(name2score.values())
        h = self.history
        if len(h) >= 1 and score > max(h) and self.enable_save_params:
            self.model.save(self.param_file)
        h.append(score)
        early = 'early stop[epoch {}]:'.format(len(h))
        if (len(h) >= 5 and score <= 0.1) or (len(h) >= 50 and score <= 0.3):
            self.ppp(early + 'score too small-t.s')
            return True
        if len(h) <= 10:
            return False
        peak_idx = int(np.argmax(h))
        from_peak = (np.array(h) - h[peak_idx])[peak_idx:]
        if sum(from_peak) <= -0.5:
            self.ppp(early + 'drastic decline-d.d')
            return True
        if len(from_peak) >= 15:
            self.ppp(early + 'no increase for too long-n.i')
            return True
        return False

    @staticmethod
    def get_docarr_feature(docarr: List[Document]) -> List[List[int]]:
        return [doc.tokenids for doc in docarr]


if __name__ == '__main__':
    Runner(C.parse_args()).run()
