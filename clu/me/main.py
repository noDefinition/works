from clu.data.datasets import Sampler
from clu.me import C
from clu.me.gen1 import name2m_class, N1
from utils import au, iu, lu
from utils.deep.layers import get_session, np, tf


class Runner:
    def __init__(self, args: dict):
        self.args = args
        self.gid = args[C.gid]
        self.epoch_num = args[C.ep]
        self.gpu_id, self.gpu_frac = args[C.gi], args[C.gp]
        self.batch_size, self.neg_size = args[C.bs], args[C.ns]
        self.data_name, self.model_name = args[C.dn], args[C.vs]

        self.w_init, self.c_init = args[C.wini], args[C.cini]
        self.scale = args[C.sc]
        self.c_num = args[C.cn]

        self.log_path = args[C.lg]
        entries = [(k, v) for k, v in args.items() if v is not None]
        log_name = au.entries2name(entries, exclude={C.gi, C.gp, C.lg}, postfix='.txt')
        self.logger = lu.get_logger(iu.join(self.log_path, log_name))

        self.use_record = False
        self.save_model_params = False
        if self.use_record:
            self.writer_path = iu.join(self.log_path, 'gid={}'.format(self.gid))
            self.param_file = iu.join(self.writer_path, 'model.ckpt')
            iu.mkdir(self.writer_path, rm_prev=True)

        self.model = None
        self.model_class = name2m_class[self.model_name]
        self.history = list()
        self.writer_step = 0
        self.ppp(iu.dumps(args))

    def ppp(self, info):
        print(info)
        self.logger.info(info)

    def get_writer(self, sess):
        def summary_details(fd):
            writer.add_summary(sess.run(model_merge, feed_dict=fd), self.writer_step)
            self.writer_step += 1

        def summary_scores(s2v):
            p2v = {s2p[s]: s2v[s] for s in s2p.keys()}
            writer.add_summary(sess.run(score_merge, feed_dict=p2v), len(self.history))

        import tensorflow.summary as su
        writer = su.FileWriter(self.writer_path, sess.graph)
        model_merge = su.merge_all()
        s2p = {t: tf.placeholder(dtype=tf.float32) for t in au.eval_scores}
        score_merge = su.merge([su.scalar(s, p, family='eval') for s, p in s2p.items()])
        return summary_details, summary_scores

    def run(self):
        np.random.seed(5413 + self.gid)
        tf.set_random_seed(17256 + self.gid)
        sampler = Sampler(self.data_name)
        sampler.load()
        w_embed, c_embed = sampler.word_embed_init, sampler.clu_embed_init
        # if self.e_init == 0:
        #     self.ppp('random w & c embed')
        #     w_embed = np.random.normal(0., self.scale, size=w_embed.shape)
        #     c_embed = np.random.normal(0., self.scale, size=c_embed.shape)
        model = self.model = self.model_class(self.args)
        assert isinstance(model, N1)
        model.build(w_embed, c_embed)
        model.set_session(get_session(self.gpu_id, self.gpu_frac, allow_growth=True))
        # summary_details, summary_scores = self.get_writer(sess)
        for e in range(self.epoch_num):
            print('\nepoch:{}'.format(e))
            """ train & predict """
            train_batches = list(sampler.shuffle_generate(self.batch_size, self.neg_size))
            data_size = len(train_batches)
            for bid, (pos, negs) in enumerate(train_batches):
                fd = model.get_fd_by_batch(pos, negs)
                model.train_step(fd, e, self.epoch_num, bid, data_size)
                if bid % (data_size // 8) == 0:
                    self.ppp('b:{} L:{}'.format(bid, model.get_loss(fd)))
                    # summary_details(fd)
            """ evaluate """
            s2v = model.evaluate(sampler.eval_batches)
            # summary_scores(s2v)
            # if self.is_record and (len(self.history) <= 1 or score > max(self.history[:-1])):
            if self.should_early_stop(s2v):
                # model.manual_assign_adv(train_batches, sampler.eval_batches, self.ppp)
                return

    def should_early_stop(self, name2score: dict):
        self.ppp(iu.dumps(name2score))
        score = sum(name2score.values())
        h = self.history
        if len(h) >= 1 and score > max(h) and self.save_model_params:
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


if __name__ == '__main__':
    Runner(C.parse_args()).run()
