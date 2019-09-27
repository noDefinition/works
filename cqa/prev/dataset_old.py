import json
import pickle
import time

import math
import numpy as np
from scipy import sparse

import utils

rank_metric = False
add_features = False
word2vec_size = 64


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
        self.es = [self.rmse(y_true, y_pred), self.mae(y_true, y_pred)]
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
    
    def __init__(self, prv_list):
        self.prv = sorted(prv_list, reverse=True)
        prvr = [self.prv[i] + (i + 1,) for i in range(len(self.prv))]
        self.iprv = sorted(prvr, key=lambda prv: prv[1])
        # print(self.prv_list)
        # print()
        # for prv in self.prv_list:
        # print(prv)
        # input()
        self.es = [self.ndcg(5), self.MAP(), self.MRR()]
        self.cmps = [1, 1, 1]
    
    def ndcg(self, k=5):
        def dcg_score(prv_list):
            # score = np.array([v[2] for v in prv_list[:k]])
            # score[score < 0] = -1
            # gain = score + 1
            score = np.array([v[2] for v in prv_list])
            score = score - score.min()
            score = score / (score.max() + 1e-7)
            score = score[:k]
            gain = score
            discounts = np.log2(np.arange(k) + 2)
            return np.sum(gain / discounts)
        
        dcg = dcg_score(self.prv)
        idcg = dcg_score(self.iprv)
        if idcg < 1e-7:
            return 1
        return dcg / idcg
    
    def MAP(self, k=0):
        ps = []
        rank = [prv[1] for prv in self.prv]
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
        return 1 / self.iprv[0][-1]


class tf_data:
    def __init__(self, dataset):
        # so or zhihu
        self.dataset = dataset
        # uid, qt, at
        home = '/home/wwang/Projects/QAR_data/{}_data'.format(dataset)
        # self.word_idf = pickle.load(open('{}/make_text/word_idf_list.pkl'.format(home), 'rb'))
        # self.word_idf = np.array(self.word_idf) / np.mean(self.word_idf)
        
        if add_features:
            faid, self.features = pickle.load(
                open('{}/quality_features/aid_features_tuple.pkl'.format(home), 'rb'))
            self.features = self.features - self.features.min(0)
            self.features = self.features / self.features.max(0)
            self.faid2int = dict(zip(faid, range(len(faid))))
        
        self.home = '{}/data'.format(home, dataset)
        numbers = json.load(open('{}/numbers.json'.format(self.home), 'r'))
        self.__dict__.update(numbers)
        utils.logger.log(numbers)
        # print('dataset size: ', word2vec_size); input()
        try:
            self.word2vec = pickle.load(open('{}/word2vec_{}d.pkl'.format(self.home, word2vec_size), 'rb'))
            self.user2vec = pickle.load(open('{}/user2vec_{}d.pkl'.format(self.home, word2vec_size), 'rb'))
            print('word dict:', self.word2vec.shape, self.nb_words)
            print('user vec:', self.user2vec.shape, self.nb_users)
        except Exception as e:
            utils.logger.log(e, red=True)
            utils.logger.log('not found word2vec', red=True)
        # utils.logger.log('metrics: {}'.format(Evaluation.metrics))
        self.cache_data = {}
        self.cache_size = int(4e5)
    
    def scale_y(self, y):
        # return np.array(y)
        return np.sign(y) * np.log2(np.abs(y) + 1)
    
    def get_fn(self, dataname):
        return '{}/{}.json'.format(self.home, dataname)
    
    def parse(self, line):
        # line: qid-0, vote-1, rank-2, aid-3, uid-4, q_wids-5, a_wids-6, u_int-7
        line = json.loads(line)
        # data = [line[5], line[6], line[7], line[1]]
        data = line[-3:]
        if add_features:
            data.append(self.features[self.faid2int[line[3]]].tolist())
        info = line[:3]
        return data, info
    
    def gen_data_by_batch(self, dataname, batch_size, max_batch=0):
        if dataname not in self.cache_data:
            self.cache_data[dataname] = [None] * self.cache_size
        with open(self.get_fn(dataname), 'r') as f:
            data = [[], [], []]
            if add_features:
                data.append([])
            info = [[], [], []]
            for line_number, line in enumerate(f):
                if len(data[0]) == batch_size:
                    yield data, info
                    data = [[], [], []]
                    if add_features:
                        data.append([])
                    info = [[], [], []]
                if line_number < self.cache_size:
                    if self.cache_data[dataname][line_number] is None:
                        self.cache_data[dataname][line_number] = self.parse(line)
                    d, i = self.cache_data[dataname][line_number]
                else:
                    d, i = self.parse(line)
                for _ in range(len(data)):
                    data[_].append(d[_])
                for _ in range(len(info)):
                    info[_].append(i[_])
            yield data, info
    
    def evaluate(self, dataname, pred_f, batch_size=256):
        N = self.__dict__['nb_{}'.format(dataname)]
        n = math.ceil(N / batch_size)
        qid2prv = {}
        progress_bar = utils.ProgressBar(n, msg='{}'.format(dataname))
        y_true = []
        y_pred = []
        for batch, info in self.gen_data_by_batch(dataname, batch_size):
            qids, votes, ranks = info
            pred_y = pred_f(batch)
            y_pred.extend(pred_y)
            y_true.extend(self.scale_y(votes))
            for i in range(len(qids)):
                qid = qids[i]
                qid2prv.setdefault(qid, [])
                qid2prv[qid].append((pred_y[i], ranks[i], votes[i]))
            if 0:
                print(pred_y[:10])
                print(self.scale_y(votes)[:10])
                input()
            else:
                progress_bar.make_a_step()
        t = progress_bar.stop()
        if rank_metric:
            metric = Metric.mean([RankMetric(v) for v in qid2prv.values()])
        else:
            metric = ErrorMetric(y_true, y_pred)
        return metric, t
    
    def gen_data(self, dataname='train', batch_size=2, max_batch=0):
        while True:
            for batch, info in self.gen_data_by_batch(dataname, batch_size, max_batch):
                yield batch + [self.scale_y(info[1])]
    
    def gen_train_pair(self, batch_size, max_batch=0):
        if add_features:
            n = 7
        else:
            n = 5
        nb_batch = 0
        while True:
            with open(self.get_fn('train_pair'), 'r') as f:
                data = [[] for i in range(n)]
                for line_number, line in enumerate(f):
                    if len(data[0]) == batch_size:
                        yield data
                        nb_batch += 1
                        if nb_batch == max_batch:
                            return
                        data = [[] for i in range(n)]
                    d = json.loads(line)
                    if add_features:
                        rge = [0, 1, 2, 3, 4, 5, 6]
                    else:
                        rge = [0, 1, 2, 4, 5]
                    for i in range(n):
                        data[i].append(d[rge[i]])
                yield data


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
        aid, self.aid_tfidf = pickle.load(open('{}/make_text/aid_tfidf_tuple.pkl'.format(home), 'rb'))
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
        data = [self.qid2int[line[0]], self.aid2int[line[3]], line[7], line[1], self.faid2int[line[3]]]
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
    
    def gen_train_pair(self, batch_size, max_batch=0):
        nb_batch = 0
        # while True:
        #     with open(self.get_fn('train_pair'), 'r') as f:


def check(ds='test'):
    home = '/home/wwang/Projects/QAR_data/{}_data'.format(ds)
    data = tf_data(ds)
    word_dict = json.load(open('{}/make_text/word_dict.json'.format(home), 'r'))
    n = len(word_dict)
    print('nb_words:', n)
    int2word = {}
    for k in word_dict:
        int2word[word_dict[k]] = k
    int2word[n + 1] = '#empty'
    batch_size = 10
    # print(word_dict)
    for batch, info in data.gen_data_by_batch('test', batch_size, max_batch=1):
        bs = len(batch[0])
        for i in range(bs):
            q = batch[0][i]
            a = batch[1][i]
            u = batch[2][i]
            qt = ' '.join([int2word[w] for w in q if w > 0])
            at = ' '.join([int2word[w] for w in a if w > 0])
            print('\nqid:', info[0][i])
            print(qt)
            print('\n')
            print(at)
            print('user:', u)
            input()


def sim(ds='test'):
    print('ds:', ds)
    np.random.seed(123456)
    home = '/home/wwang/Projects/QAR_data/{}_data'.format(ds)
    fn = '{}/data/{}.json'.format(home, 'train')
    qid, qid_tfidf = pickle.load(open('{}/make_text/qid_title_tfidf_tuple.pkl'.format(home), 'rb'))
    np.random.shuffle(qid)
    qid2int = dict(zip(qid, range(len(qid))))
    aid, aid_tfidf = pickle.load(open('{}/make_text/aid_tfidf_tuple.pkl'.format(home), 'rb'))
    aid2int = dict(zip(aid, range(len(aid))))
    qid2aids = {}
    for line in open(fn, 'r'):
        # data: qid-0, vote-1, rank-2, aid-3, uid-4, q_wids-5, a_wids-6, u_int-7
        data = json.loads(line)
        qid = data[0]
        aid = data[3]
        rank = data[2]
        qid2aids.setdefault(qid, [])
        qid2aids[qid].append((rank, aid))
        # print(data); input()
    n = 10
    s = [[] for i in range(n)]
    for qid, rank_aids in qid2aids.items():
        rank_aids = sorted(rank_aids)
        for i in range(min(n, len(rank_aids))):
            # print(i)
            aid = rank_aids[i][1]
            # print(i, rank_aids[i])
            # print(qid2int[qid])
            s[i].append((qid_tfidf[qid2int[qid]] * aid_tfidf[aid2int[aid]].T).toarray()[0, 0])
            # input()
    for i in range(n):
        print('rank{}: {}'.format(i + 1, np.mean(s[i])))


def main():
    print('hello world, uclu_bert_dataset.py')
    bt = time.time()
    # sim('so')
    # print('time:', time.time() - bt)
    # return
    # check('zhihu'); return
    # import utils
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
        data = tf_data('test')
        # for p in data.gen_train_pair(1):
        for p in data.gen_data_by_batch('train', 1):
            for pp in p:
                print(pp);
                input()


if __name__ == '__main__':
    main()
