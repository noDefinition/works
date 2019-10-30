from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from cqa.data.datasets import Data, DataSo, DataZh
from cqa.mee.evaluate import MeanRankScores
from cqa.mee.main import my_pbar
from cqa.mee.models.funcs import instant_denses, pairwise_sub
from utils import iu, lu, mu, tmu, tu
from utils.deep.layers import *


@tmu.stat_time_elapse
def get_x_train(d_class):
    data = d_class()
    data.load_cdong_full()
    x_train = list()
    idx = 0
    for qid, (ql, al, _, _) in data.qid2qauv.items():
        for wids in [ql] + al:
            idx += 1
            td = TaggedDocument(list(map(str, wids)), [idx])
            x_train.append(td)
    return x_train


def train_model(d_class):
    x_train = get_x_train(d_class)
    model = Doc2Vec(
        x_train, dm=1, size=64, window=10, alpha=1e-3, min_alpha=1e-4,
        min_count=0, workers=20, iter=20, negative=10,
    )
    data = d_class()
    model.save(data.fill('_mid', 'doc2vec_model'))
    model.load()


def doc2vec_mlp_multi():
    od_list = tu.LY((
        ('d_class', [DataSo, DataZh]),
        ('gpu_id', [0, 1]),
        # ('local_id', [0, 1]),
    )).eval()
    for idx, od in enumerate(od_list):
        od['gid'] = idx
        print(od)
    print(len(od_list))
    mu.multi_process(doc2vec_mlp, kwargs_list=od_list)


# so mean len 81.61
# zh mean len 87.33
def doc2vec_mlp(gid, d_class, gpu_id):
    # print(d_class)
    # return
    d2vm = Doc2VecMLP(gid, d_class, gpu_id)
    d2vm.fit()
    # d = Doc2VecMLP()
    # data = d_class()
    # data.load_cdong_full()
    # lenarr = list()
    # for qid, (_, al, _, _) in data.qid2qauv.items():
    #     for wids in al:
    #         cnt = 0
    #         for wid in wids:
    #             if wid > 0:
    #                 cnt += 1
    #         lenarr.append(cnt)
    # print(np.mean(lenarr))


class Doc2VecMLP(object):
    def __init__(self, gid, d_class, gpu_id):
        self.lr = 1e-4
        self.gid = gid
        self.gpu_id = gpu_id
        self.sess: tf.Session = None
        self.train_op = None

        assert issubclass(d_class, Data)
        self.data = d_class()
        self.data.rvs2qids = iu.load_pickle(self.data.rvs2qids_file)
        self.qid2vecs_vl = iu.load_pickle(self.data.fill('_mid', 'qid2vecs_vl.pkl'))

        self.logger = lu.get_logger('./d2v/gid={}.txt'.format(self.gid))
        self.best_mrs = None
        self.brk_cnt = 0

        # graph
        self.vecs = tf.placeholder(f32, (None, 64))
        self.vl = tf.placeholder(i32, (None,))
        self.is_train = tf.placeholder_with_default(False, ())
        self.true_scores = tf.cast(self.vl, dtype=f32)

        uas = [(64, relu), (64, relu), (1, None), ]
        outputs = instant_denses(self.vecs, uas, name='output')  # (bs, 1)
        self.pred_scores = tf.squeeze(outputs, axis=1)  # (bs,)

        true_pw = pairwise_sub(self.true_scores, name='true_pw')
        ones_upper = tf.matrix_band_part(tf.ones_like(true_pw), 0, -1, name='ones_upper')
        true_sign = tf.sign(true_pw * ones_upper, name='true_sign')
        pred_pw = pairwise_sub(self.pred_scores, name='pred_pw')
        margin_pw = tf.maximum(1. - pred_pw * true_sign, 0., name='margin_pw')
        margin_pw = tf.layers.dropout(margin_pw, rate=0.5, training=self.is_train)
        margin_loss = tf.reduce_sum(margin_pw, name='margin_loss')
        opt = tf.train.AdamOptimizer(learning_rate=self.lr, name='adam_opt')
        self.train_op = opt.minimize(margin_loss, name='train_op')

    def train_step(self, vecs, vl):
        fd = {self.vecs: vecs, self.vl: vl, self.is_train: True}
        self.sess.run(self.train_op, feed_dict=fd)

    def predict(self, vecs):
        fd = {self.vecs: vecs}
        return self.sess.run(self.pred_scores, feed_dict=fd)

    # @tmu.stat_time_elapse
    # def load(self):
    #     self.qid2vecs = iu.load_pickle(d_class().fill('_mid', 'qid2vecs_vl.pkl'))
    #     self.data.load_cdong_full()
    #     for qid, (_, al, _, vl) in self.data.qid2qauv.items():
    #         if qid not in self.qid2vecs:
    #             vecs = list()
    #             for wids in al:
    #                 v = self.doc2vec.infer_vector(list(map(str, wids)))
    #                 vecs.append(v)
    #             self.qid2vecs[qid] = (vecs, vl)
    #         else:
    #             raise ValueError('qid duplicated')

    def fit(self):
        self.sess = get_session(self.gpu_id, 0.3, allow_growth=False, run_init=True)
        train_qids = self.data.get_train_qids()
        print(self.gid, 'start train')
        for e in range(10):
            # with my_pbar('train', len(train_qids), leave=True, ncols=60) as pbar:
            for bid, qid in enumerate(train_qids):
                vecs, vl = self.qid2vecs_vl[qid]
                self.train_step(vecs, vl)
                # pbar.update()
                if bid > 0 and bid % (len(train_qids) // 3.2) == 0:
                    if self.should_stop():
                        return

    def should_stop(self):
        v_mrs = self.evaluate(self.data.get_valid_qids())
        if v_mrs.is_better_than(self.best_mrs):
            self.brk_cnt = 0
            self.best_mrs = v_mrs
            t_mrs = self.evaluate(self.data.get_test_qids())
            print(self.gid, 'test:', t_mrs.to_dict())
            self.logger.info(iu.dumps(t_mrs.to_dict()))
        else:
            print(self.gid, 'break count:', self.brk_cnt)
            self.brk_cnt += 1
        return self.brk_cnt >= 5

    def evaluate(self, qids) -> MeanRankScores:
        mrs = MeanRankScores()
        for qid in qids:
            vecs, vl = self.qid2vecs_vl[qid]
            pred = self.predict(vecs)
            mrs.append(vl, pred)
        return mrs


if __name__ == '__main__':
    # mu.multi_process(train_model, [(DataSo,), (DataZh,), ])
    # mu.multi_process(doc2vec_mlp, [(0, DataSo, 0), ])
    # mu.multi_process(doc2vec_mlp, [(DataZh, 0), ])
    doc2vec_mlp_multi()
