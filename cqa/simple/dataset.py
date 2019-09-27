import json
import pickle
import time

import numpy as np
# from tqdm import tqdm


class Metric:
    eps = 3
    
    def __init__(self, es, cmps):
        self.es = es
        self.cmps = cmps
    
    def is_better_than(self, other):
        if other is None:
            return True
        return sum(self.es * np.array(self.cmps)) > sum(other.es * np.array(other.cmps))
    
    @staticmethod
    def mean(ms):
        es = [np.mean([m.es[i] for m in ms]) for i in range(len(ms[0].es))]
        return Metric(es, ms[0].cmps)
    
    @staticmethod
    def merge(a, b):
        return Metric(a.es + b.es, a.cmps + b.cmps)
    
    def __str__(self):
        s = '{{:.{}f}}'.format(self.eps)
        s = ','.join([s] * len(self.es))
        s = '[' + s + ']'
        return s.format(*self.es)
    
    def prt(self):
        return '\t'.join(['{}'] * len(self.es)).format(*self.es)


class RankMetric(Metric):
    # metrics = 'ndcg@5, MAP, MRR'
    
    def __init__(self, tp_list):
        self.tpr = sorted(tp_list, reverse=True)
        self.tpr[0] = self.tpr[0] + (1,)
        for i in range(1, len(self.tpr)):
            if self.tpr[i][0] == self.tpr[i - 1][0]:
                self.tpr[i] = self.tpr[i] + (self.tpr[i - 1][2],)
            else:
                self.tpr[i] = self.tpr[i] + (i + 1,)
        self.sorted_tpr = sorted(self.tpr, key=lambda x: (-x[1], x[0]))
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
    nb_features = 9
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.home = '/home/wwang/Projects/QAR_data/{}_data'.format(dataset)
        self.load_data()
    
    def load_data(self):
        # load features data
        data_home = '/home/yhzhang/Projects/get_feature/{}'.format(self.dataset)
        self.uid2features = pickle.load(open('{}/uid2features.pkl'.format(data_home), 'rb'))
        self.qid2features = pickle.load(open('{}/qid2features.pkl'.format(data_home), 'rb'))
        self.aid2features = pickle.load(open('{}/aid2features.pkl'.format(data_home), 'rb'))
        # self.aid2features = pickle.load(open('{}/aid2len.pkl'.format(data_home), 'rb'))
        # self.qid2features = pickle.load(open('{}/qid2len.pkl'.format(data_home), 'rb'))
        # print(type(list(self.uid2features.values())[0]))
        # print(self.nb_users)
        # print(self.nb_answers)
        # print(len(self.uid2features))
        # print(len(self.qid2features))
        # print(len(self.aid2features))
        # input()
        
        home = self.home
        self.train_aid_pairs = pickle.load(open('{}/data/train_aid_pairs_sample_list.pkl'.format(home), 'rb'))
        self.nb_train_data = len(self.train_aid_pairs)
        self.wordVec = pickle.load(open('{}/features/wordVec_64d.pkl'.format(home), 'rb'))
        
        numbers = json.load(open('{}/raw/numbers.json'.format(home), 'r'))
        self.__dict__.update(numbers)
        
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
    
    def scale_y(self, y):
        return y
        # return np.sign(y) * np.log2(np.abs(y) + 1)
    
    def get_data(self, aid):
        qid, uid = self.aid2qu[aid]
        q = self.qid2wids[qid]
        u = self.uid2int[uid]
        a = self.aid2wids[aid]
        v = self.aid2vote[aid]
        v = self.scale_y(v)
        # f = [0, 1, 2]
        f = self.qid2features[qid] + self.aid2features[aid] + self.uid2features[uid]
        # print(len(f))
        # input()
        return [q, a, u, f], v
    
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
            for _ in range(len(_x)):
                x[_].append(_x[_])
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
            x = x_pos + x_neg
            if data is None:
                data = [[] for _ in range(len(x))]
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
        qid2tp = {}
        pbar = tqdm(total=n, leave=False, ncols=100, desc='Evaluating')
        for x, y, aids in self.gen_data_by_batch(dataname, batch_size):
            pred_y = pred_f(x)
            for i in range(len(aids)):
                qid = self.aid2qu[aids[i]][0]
                qid2tp.setdefault(qid, list()).append((y[i], pred_y[i]))
            pbar.update(1)
        metric = Metric.mean([RankMetric(v) for v in qid2tp.values()])
        pbar.close()
        return metric


# class sk_data:
#     def __init__(self, dataset):
#         self.dataset = dataset
#         self.home = '/home/wwang/Projects/QAR_data/{}_data'.format(dataset)
#         self.load_data()
#
#     def load_data(self):
#         home = self.home
#         self.train_aid_pairs = pickle.load(open('{}/data/train_aid_pairs_sample_list.pkl'.format(home), 'rb'))
#         numbers = json.load(open('{}/raw/numbers.json'.format(home), 'r'))
#         self.__dict__.update(numbers)
#
#         self.aid2vote = pickle.load(open('{}/raw/aid2vote_dict.pkl'.format(home), 'rb'))
#         self.aid2qu = pickle.load(open('{}/data/aid2qu_dict.pkl'.format(home), 'rb'))
#
#         self.uid2int = pickle.load(open('{}/raw/uid2int_dict.pkl'.format(home), 'rb'))
#
#         train_aids = pickle.load(open('{}/data/train_aids_list.pkl'.format(home), 'rb'))
#         vali_aids = pickle.load(open('{}/data/vali_aids_list.pkl'.format(home), 'rb'))
#         test_aids = pickle.load(open('{}/data/test_aids_list.pkl'.format(home), 'rb'))
#         self.data = {'train': train_aids, 'vali': vali_aids, 'test': test_aids}
#         self.nb_users = len(self.uid2int)
#
#     def get_data(self, aid):
#         q, u = self.aid2qu[aid]
#         v = self.aid2vote[aid]
#         v = self.scale_y(v)
#         return [q, u], v
#
#     def get_xy_aids(self, dataname):
#         data, y, aids = list(self.gen_data_by_batch(dataname, -1))[0]
#         qids = data[0]
#         uids = data[1]
#         # q =
#         return x, y, aids
#
#     def evaluate(self, dataname, pred_f):
#         x, y, aids = self.get_xy_aids(dataname)
#         pred_y = pred_f(x)
#         qid2tp = {}
#         for i in range(len(aids)):
#             qid = self.aid2qu[aids[i]][0]
#             qid2tp.setdefault(qid, [])
#             qid2tp[qid].append((y[i], pred_y[i]))
#         metric = Metric.mean([RankMetric(v) for v in qid2prv.values()])
#         return metric


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
            # time.sleep(1)
            return [m for i in range(len(x[0]))]
            # return [np.random.randint(100) for i in range(len(x[0]))]
        
        data = tf_data('so')
        m = np.mean(data.get_y('train'))
        print(m)
        m = data.evaluate('test', pred)
        print(m)
        print(data.nb_train_data)
        for p in data.gen_data_by_batch('train', 3):
            print(p)
            input()


if __name__ == '__main__':
    main()
