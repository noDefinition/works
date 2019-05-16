from clu.data.datasets import Sampler
from clu.me import C
from clu.me.gen1 import *
from utils import au, iu, lu
from utils.deep.layers import get_session, np, tf


class Runner:
    def __init__(self, args: dict):
        self.args = args
        self.gid = args[C.gid]
        self.gpu_id = args[C.gi]
        self.gpu_frac = args[C.gp]
        self.epoch_num = args[C.ep]
        self.batch_size = args[C.bs]
        self.neg_size = args[C.ns]
        self.data_name = args[C.dn]
        self.model_name = args[C.vs]

        self.w_init = args[C.wini]
        self.c_init = args[C.cini]
        self.scale = args[C.sc]
        self.c_num = args[C.cn]

        self.log_path = args[C.lg]
        entries = [(k, v) for k, v in args.items() if v is not None]
        log_name = au.entries2name(entries, exclude={C.gi, C.gp, C.lg}, postfix='.txt')
        self.log_file = iu.join(self.log_path, log_name)
        self.logger = lu.get_logger(self.log_file)

        # self.is_record = Nodes.is_1702()
        self.is_record = False
        if self.is_record:
            self.writer_path = iu.join(self.log_path, 'gid={}'.format(self.gid))
            self.param_file = iu.join(self.writer_path, 'model.ckpt')
            self.hyper_file = iu.join(self.writer_path, 'hyper')
            iu.mkdir(self.writer_path)
            iu.dump_json(self.hyper_file, args)

        self.history = list()
        self.writer_step = 0
        self.ppp(args)

    def ppp(self, info):
        print(info)
        self.logger.info(info)

    def get_model_class(self):
        return {v.__name__: v for v in [N6]}[self.model_name]

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
        w_embed, c_embed = sampler.d_obj.load_word_cluster_embed()
        # if self.c_num != sampler.d_obj.topic_num:
        #     from baselinee.kmeans import fit_kmeans
        #     avg_embeds, _ = sampler.d_obj.get_avg_embeds_and_topics()
        #     c_embed = fit_kmeans(avg_embeds, self.c_num).cluster_centers_
        #     self.ppp('c_embed.shape: {}'.format(c_embed.shape))
        # if self.e_init == 0:
        #     self.ppp('random w & c embed')
        #     w_embed = np.random.normal(0., self.scale, size=w_embed.shape)
        #     c_embed = np.random.normal(0., self.scale, size=c_embed.shape)
        model = self.get_model_class()(self.args)
        model.build(w_embed, c_embed)
        sess = get_session(self.gpu_id, self.gpu_frac, allow_growth=True, run_init=True)
        model.set_session(sess)
        # summary_details, summary_scores = self.get_writer(sess)
        for e in range(self.epoch_num):
            print('\nepoch:{}'.format(e))
            """ train & predict """
            train_batches = list(sampler.shuffle_generate(self.batch_size, self.neg_size))
            data_size = len(train_batches)
            for bid, pos, negs in train_batches:
                fd = model.get_fd_by_batch(pos, negs)
                model.train_step(fd, e, self.epoch_num, bid, data_size)
                if bid % (data_size // 8) == 0:
                    self.ppp('b:{} L:{}'.format(bid, model.get_loss(fd)))
                    # summary_details(fd)
            """ evaluate """
            s2v = model.evaluate(sampler.eval_batches)
            self.ppp(iu.dumps(s2v))
            # summary_scores(s2v)
            score = s2v['acc']
            self.history.append(score)
            # if self.is_record and (len(self.history) <= 1 or score > max(self.history[:-1])):
            #     tf.train.Saver(tf.trainable_variables()).save(sess, self.param_file)
            if self.should_early_stop(score):
                # model.manual_assign_adv(train_batches, sampler.eval_batches, self.ppp)
                return

    def should_early_stop(self, score):
        h = self.history
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
