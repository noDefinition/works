import json
import pickle
import time

import numpy as np
from scipy import sparse

import utils


class Metric:
    eps = 3
    
    def __init__(self, es, cmps):
        self.es = es
        self.cmps = cmps
    
    def is_better_than(self, other):
        if other is None:
            return True
        # return sum(self.es * np.array(self.cmps)) > sum(other.es * np.array(other.cmps))
        return list(self.es * np.array(self.cmps)) > list(other.es * np.array(other.cmps))
    
    @staticmethod
    def mean(ms):
        es = [np.mean([m.es[i] for m in ms]) for i in range(len(ms[0].es))]
        return Metric(es, ms[0].cmps)
    
    @staticmethod
    def merge(a, b):
        return Metric(a.es + b.es, a.cmps + b.cmps)
    
    def __str__(self):
        # self.es = [1,2,3]
        s = '{{:.{}f}}'.format(self.eps)
        s = ','.join([s] * len(self.es))
        s = '[' + s + ']'
        return s.format(*self.es)
    
    def prt(self):
        return '\t'.join(['{}'] * len(self.es)).format(*self.es)


class ErrorMetric(Metric):
    metrics = 'MSE, MAE'
    
    def __init__(self, y_true, y_pred):
        self.es = [self.mse(y_true, y_pred), self.mae(y_true, y_pred)]
        self.cmps = [-1, -1]
    
    @staticmethod
    def mse(y_true, y_pred):
        e = np.array(y_true) - y_pred
        return np.mean(e ** 2)
    
    @staticmethod
    def rmse(y_true, y_pred):
        e = np.array(y_true) - y_pred
        return np.sqrt(np.mean(e ** 2))
    
    @staticmethod
    def mae(y_true, y_pred):
        e = np.array(y_true) - y_pred
        return np.mean(np.abs(e))


class RankMetric(Metric):
    metrics = 'ndcg@5, MAP, MRR'
    
    def __init__(self, tp_list):
        self.tpr = sorted(tp_list, reverse=True)
        self.tpr[0] = self.tpr[0] + (1,)
        for i in range(1, len(self.tpr)):
            if self.tpr[i][0] == self.tpr[i - 1][0]:
                self.tpr[i] = self.tpr[i] + (self.tpr[i - 1][2],)
            else:
                self.tpr[i] = self.tpr[i] + (i + 1,)
        
        self.sorted_tpr = sorted(self.tpr, key=lambda x: (-x[1], x[0]))
        # print(self.tpr)
        # print(self.sorted_tpr)
        # input()
        self.es = [self.ndcg(5), self.MAP(), self.MRR()]
        self.cmps = [1, 1, 1]
    
    def ndcg(self, k=5):
        def dcg_score(tp):
            score = np.array([v[0] for v in tp])
            score = score[:k]
            score[score >= 0] = score[score >= 0] + 1
            score[score < 0] = score[score < 0] - 1
            score = np.array(score, dtype='float32')
            score[score < 0] = -1 / score[score < 0]
            gain = score
            discounts = np.log2(np.arange(min(k, len(score))) + 2)
            return np.sum(gain / discounts)
        
        dcg = dcg_score(self.sorted_tpr)
        idcg = dcg_score(self.tpr)
        if idcg < 1e-7:
            return 1
        return dcg / idcg
    
    def MAP(self, k=0):
        ps = []
        rank = [tpr[2] for tpr in self.sorted_tpr]
        if k:
            rank = rank[:k]
        n = len(rank)
        for i in range(n):
            a = i + 1
            b = 0
            for j in range(a):
                if rank[k] <= a:
                    b += 1
            ps.append(b / a)
        return np.mean(ps)
    
    def MRR(self):
        for i, tpr in enumerate(self.sorted_tpr):
            if tpr[2] == 1:
                return 1 / (i + 1)
        return 1 / 0


class tf_data:
    def __init__(self, dataset):
        home = '/home/wwang/Projects/QAR_data/{}_data'.format(dataset)
        self.home = home
        numbers = json.load(open('{}/raw/numbers.json'.format(home), 'r'))
        self.__dict__.update(numbers)
        # utils.logger.log(numbers)
        self.aid2vote = pickle.load(open('{}/raw/aid2vote_dict.pkl'.format(home), 'rb'))
        self.aid2wids = pickle.load(open('{}/features/aid2wids_dict.pkl'.format(home), 'rb'))
        self.aid2qu = pickle.load(open('{}/data/aid2qu_dict.pkl'.format(home), 'rb'))
        
        self.qid2wids = pickle.load(open('{}/features/qid2wids_dict.pkl'.format(home), 'rb'))
        
        self.uid2int = pickle.load(open('{}/raw/uid2int_dict.pkl'.format(home), 'rb'))
        
        train_aids = pickle.load(open('{}/data/train_aids_list.pkl'.format(home), 'rb'))
        vali_aids = pickle.load(open('{}/data/vali_aids_list.pkl'.format(home), 'rb'))
        test_aids = pickle.load(open('{}/data/test_aids_list.pkl'.format(home), 'rb'))
        self.data = {'train': train_aids, 'vali': vali_aids, 'test': test_aids}
        self.nb_users = len(self.uid2int)
    
    def set_args(self, pair_mode='all', rank=True, wordVec_size=64, add_features=False):
        self.pair_mode = pair_mode
        self.rank = rank
        self.wordVec_size = wordVec_size
        self.add_features = add_features
        self.load_data()
    
    def load_data(self):
        home = self.home
        if self.rank:
            if self.pair_mode == 'question':
                self.train_qids = pickle.load(open('{}/raw/train_qids_list.pkl'.format(home), 'rb'))
                self.qid2vau = pickle.load(
                    open('{}/raw/qid2vote_aid_uid_dict.pkl'.format(home), 'rb'))
                self.nb_train_data = len(self.train_qids)
            elif self.pair_mode == 'answer':
                self.train_qids = pickle.load(open('{}/raw/train_qids_list.pkl'.format(home), 'rb'))
                self.qid2vau = pickle.load(
                    open('{}/raw/qid2vote_aid_uid_dict.pkl'.format(home), 'rb'))
                self.nb_train_data = len(self.data['train']) - len(self.train_qids)
            else:
                self.train_aid_pairs = pickle.load(
                    open('{}/data/train_aid_pairs_{}_list.pkl'.format(home, self.pair_mode), 'rb'))
                self.nb_train_data = len(self.train_aid_pairs)
        else:
            self.nb_train_data = len(self.data['train'])
        if self.wordVec_size:
            self.wordVec = pickle.load(
                open('{}/features/wordVec_{}d.pkl'.format(home, self.wordVec_size), 'rb'))
    
    def scale_y(self, y):
        if self.rank:
            return y
        return np.sign(y) * np.log2(np.abs(y) + 1)
    
    def get_data(self, aid):
        q, u = self.aid2qu[aid]
        q = self.qid2wids[q]
        u = self.uid2int[u]
        a = self.aid2wids[aid]
        v = self.aid2vote[aid]
        v = self.scale_y(v)
        return [q, a, u], v
    
    def get_y(self, dataname):
        return self.scale_y([self.aid2vote[aid] for aid in self.data[dataname]])
    
    def gen_data_by_batch(self, dataname, batch_size):
        x = None
        for aid in self.data[dataname]:
            _x, _y = self.get_data(aid)
            if x is None:
                x = [[] for _ in range(len(_x))]
                y = []
                aids = []
            for i in range(len(_x)):
                x[i].append(_x[i])
            y.append(_y)
            aids.append(aid)
            if len(y) == batch_size:
                yield x, y, aids
                x = None
        if x is not None:
            yield x, y, aids
    
    def gen_data(self, dataname, batch_size):
        while True:
            yield from self.gen_data_by_batch(dataname, batch_size)
    
    def gen_train_aid_pairs(self, shuffle=False):
        if self.pair_mode == 'question':
            if shuffle:
                np.random.shuffle(self.train_qids)
            for qid in self.train_qids:
                vaus = sorted(self.qid2vau[qid], reverse=True)
                n = len(vaus)
                if n > 1 and vaus[0][0] > vaus[-1][0]:
                    while True:
                        i = np.random.randint(n)
                        j = np.random.randint(n)
                        if i > j:
                            i, j = j, i
                        if vaus[i][0] > vaus[j][0]:
                            yield vaus[i][1], vaus[j][1]
                            break
        elif self.pair_mode == 'answer':
            if shuffle:
                np.random.shuffle(self.train_qids)
            for qid in self.train_qids:
                vaus = sorted(self.qid2vau[qid], reverse=True)
                n = len(vaus)
                j = 0
                for i in range(n):
                    while j < n and vaus[i][0] <= vaus[j][0]:
                        j += 1
                    if j < n:
                        k = np.random.randint(j, n)
                        yield vaus[i][1], vaus[k][1]
                    else:
                        break
        else:
            if shuffle:
                np.random.shuffle(self.train_aid_pairs)
            yield from self.train_aid_pairs
    
    def gen_train_pair_by_batch(self, batch_size, shuffle=True):
        # data: [x_pos; x_neg]
        data = None
        # for aid_pos, aid_neg in self.train_aid_pairs:
        for aid_pos, aid_neg in self.gen_train_aid_pairs(shuffle):
            x_pos, y_pos = self.get_data(aid_pos)
            x_neg, y_neg = self.get_data(aid_neg)
            x = x_pos + x_neg[1:]
            if data is None:
                data = [[] for i in range(len(x))]
            for _ in range(len(data)):
                data[_].append(x[_])
            if len(data[0]) == batch_size:
                yield data
                data = None
        if data is not None:
            yield data
    
    def gen_train_pair(self, batch_size):
        while True:
            yield from self.gen_train_pair_by_batch(batch_size)
    
    def evaluate(self, dataname, pred_f, batch_size=256):
        n = np.ceil(len(self.data[dataname]) / batch_size)
        progress_bar = utils.ProgressBar(n, msg='{}'.format(dataname))
        qid2tp = {}
        yt = []
        yp = []
        for x, y, aids in self.gen_data_by_batch(dataname, batch_size):
            pred_y = pred_f(x)
            if self.rank:
                for i in range(len(aids)):
                    qid = self.aid2qu[aids[i]][0]
                    qid2tp.setdefault(qid, [])
                    qid2tp[qid].append((y[i], pred_y[i]))
            else:
                yt.extend(y)
                yp.extend(pred_y)
            progress_bar.make_a_step()
        t = progress_bar.stop()
        if self.rank:
            metric = Metric.mean([RankMetric(v) for v in qid2tp.values()])
        else:
            metric = ErrorMetric(yt, yp)
        return metric, t


class sk_data(tf_data):
    def __init__(self, dataset):
        # so or zhihu
        self.dataset = dataset
        home = '/home/wwang/Projects/QAR_data/{}_data'.format(dataset)
        self.home = '{}/data'.format(home, dataset)
        numbers = json.load(open('{}/numbers.json'.format(self.home), 'r'))
        self.__dict__.update(numbers)
        utils.logger.log(numbers)
        # utils.logger.log('metrics: {}'.format(Evaluation.metrics))
        q_text = 'title'
        # q_text = 'desc'
        # q_text = 'all'
        print('q_text:', q_text)
        
        if q_text == 'title' or q_text == 'all':
            qid, qid_title_tfidf = pickle.load(
                open('{}/make_text/qid_title_tfidf_tuple.pkl'.format(home), 'rb'))
        if q_text == 'desc' or q_text == 'all':
            qid, qid_body_tfidf = pickle.load(
                open('{}/make_text/qid_body_tfidf_tuple.pkl'.format(home), 'rb'))
        if q_text == 'all':
            self.qid_tfidf = qid_title_tfidf + qid_body_tfidf
        elif q_text == 'title':
            self.qid_tfidf = qid_title_tfidf
        else:
            self.qid_tfidf = qid_body_tfidf
        self.qid2int = dict(zip(qid, range(len(qid))))
        aid, self.aid_tfidf = pickle.load(
            open('{}/make_text/aid_tfidf_tuple.pkl'.format(home), 'rb'))
        self.aid2int = dict(zip(aid, range(len(aid))))
        self.uid_eye = sparse.eye(self.nb_users, format='csr')
        
        faid, self.features = pickle.load(
            open('{}/quality_features/aid_features_tuple.pkl'.format(home), 'rb'))
        self.features = self.features - self.features.min(0)
        self.features = self.features / self.features.max(0)
        self.faid2int = dict(zip(faid, range(len(faid))))
        
        self.cache = {}
    
    def parse(self, line):
        # line: qid-0, vote-1, rank-2, aid-3, uid-4, q_wids-5, a_wids-6, u_int-7
        # data: qid, aid, u_int, vote, raw_aid
        line = json.loads(line)
        data = [self.qid2int[line[0]], self.aid2int[line[3]], line[7], line[1],
                self.faid2int[line[3]]]
        # data = [self.qid2int[line[0]], self.aid2int[line[3]], line[7], -line[2]]
        info = line[:3]
        return data, info
    
    def get_xy_info(self, dataname, using_data, using_features):
        # batch: qid-0, vote-1, rank-2, aid-3, uid-4, q_wids-5, a_wids-6, u_int-7
        cache_name = 'sk_data_{}_{}_cache.pkl'.format(self.dataset, dataname)
        try:
            data, info = pickle.load(open(cache_name, 'rb'))
            data[4]
        except Exception as e:
            print(e)
            data = [[], [], [], [], []]
            info = [[], [], []]
            with open(self.get_fn(dataname), 'r') as f:
                for line_number, line in enumerate(f):
                    d, i = self.parse(line)
                    for _ in range(len(data)):
                        data[_].append(d[_])
                    for _ in range(len(info)):
                        info[_].append(i[_])
            pickle.dump((data, info), open(cache_name, 'wb'))
        dx = []
        dy = np.array(data[3])
        dy = self.scale_y(dy)
        # dy = dy - dy.min()
        # dy = dy / dy.max()
        if 'q' in using_data:
            dx.append(self.qid_tfidf[data[0]])
        if 'a' in using_data:
            dx.append(self.aid_tfidf[data[1]])
        if 'u' in using_data:
            dx.append(self.uid_eye[data[2]])
        if 'f' in using_data:
            # dx.append(np.array([self.aid2features[aid] for aid in data[4]]))
            dx.append(sparse.csr_matrix(self.features[:, using_features][data[4]]))
        if 'm' in using_data:
            dx.append(self.qid_tfidf[data[0]].multiply(self.aid_tfidf[data[1]]).sum(1))
        dx = sparse.hstack(dx) if len(dx) > 1 else dx[0]
        return (dx, dy), info
    
    def evaluate(self, dataname, pred_f, using_data, using_features):
        data, info = self.get_xy_info(dataname, using_data, using_features)
        x, y = data
        qids, votes, ranks = info
        pred_y = pred_f(x)
        qid2prv = {}
        y_true = y
        y_pred = pred_y
        for i in range(len(qids)):
            qid = qids[i]
            qid2prv.setdefault(qid, [])
            qid2prv[qid].append((pred_y[i], ranks[i], votes[i]))
        if rank_metric:
            metric = Metric.mean([RankMetric(v) for v in qid2prv.values()])
        else:
            metric = ErrorMetric(y_true, y_pred)
        return metric


def main():
    print('hello world, uclu_bert_dataset.py')
    bt = time.time()
    if 0:
        data = sk_data('so')
        xy, info = data.get_xy_info('train', 'f', [])
        x, y = xy
        avg = np.mean(y)
        
        xy, info = data.get_xy_info('vali', 'f', [])
        x, y = xy
        # print(x)
        # print(info)
        print(avg, np.mean(y))
        print(ErrorMetric(y, np.ones(y.shape) * np.mean(y)))
        print(ErrorMetric(y, np.ones(y.shape) * avg))
        print(ErrorMetric(y, np.ones(y.shape) * 0))
    if 1:
        def pred(x):
            return [m for i in range(len(x[0]))]
            # return [np.random.randint(100) for i in range(len(x[0]))]
        
        data = tf_data('so')
        data.set_args(rank=False)
        m = np.mean(data.get_y('train'))
        print(m)
        m, t = data.evaluate('test', pred)
        print(m, t)
        print(data.nb_train_data)
        return
        for p in data.gen_data_by_batch('train', 3):
            print(p)
            input()


if __name__ == '__main__':
    main()
