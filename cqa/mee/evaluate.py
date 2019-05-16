import numpy as np


class RankScores:
    def __init__(self, y_true, y_pred):
        # tp_list 是同一个问题下的预测-实际分数对
        tp_list = list(zip(y_true, y_pred))
        self.tpr = sorted(tp_list, reverse=True)
        self.tpr[0] = self.tpr[0] + (1,)
        for i in range(1, len(self.tpr)):
            if self.tpr[i][0] == self.tpr[i - 1][0]:
                self.tpr[i] = self.tpr[i] + (self.tpr[i - 1][2],)
            else:
                self.tpr[i] = self.tpr[i] + (i + 1,)
        self.sorted_tpr = sorted(self.tpr, key=lambda x: (-x[1], x[0]))
        # self.es = [self.NDCG(5), self.MAP(5), self.MRR(k)]

    def NDCG(self, k):
        def dcg_score(ranking):
            gain = np.array([v[0] for v in ranking], dtype='float32')[:k]
            gain[gain >= 0] = gain[gain >= 0] + 1
            gain[gain < 0] = -1 / (gain[gain < 0] - 1)
            discounts = np.log2(np.arange(min(k, len(gain))) + 2)
            return np.sum(gain / discounts)

        dcg = dcg_score(self.sorted_tpr)
        idcg = dcg_score(self.tpr)
        return 1 if idcg < 1e-7 else dcg / idcg

    def MAP(self, k):
        ps = []
        rank = [tpr[2] for tpr in self.sorted_tpr]
        if k:
            rank = rank[:k]
        n = len(rank)
        for i in range(n):
            a = i + 1
            b = 0
            for j in range(a):
                if rank[j] <= a:
                    b += 1
            ps.append(b / a)
        return np.mean(ps)

    def MRR(self, k):
        for i, tpr in enumerate(self.sorted_tpr):
            if i > k:
                return 0
            if tpr[2] == 1:
                return 1 / (i + 1)
        raise ValueError('what MRR?')


class MeanRankScores:
    s_names = ['NDCG', 'MAP', 'MRR']
    k_values = [1, 5, 10, 20]

    def __init__(self):
        # self.es_list = list()  # (None, 3)
        self.name2scores = dict()
        self.name2mean: dict = None

    def append(self, y_true, y_pred):
        rs = RankScores(y_true, y_pred)
        for s, f in zip(self.s_names, [rs.NDCG, rs.MAP, rs.MRR]):
            for k in self.k_values:
                name = '{}@{}'.format(s, k)
                score = f(k)
                self.name2scores.setdefault(name, list()).append(score)

    def get_mean(self):
        if self.name2mean is None:
            self.name2mean = {name: np.mean(scores) for name, scores in self.name2scores.items()}
        assert len(set(len(v) for v in self.name2scores.values())) == 1
        return self.name2mean
        # return np.mean(self.es_list, axis=0)

    def to_dict(self):
        return {name: round(mean, 4) for name, mean in self.get_mean().items()}
        # return dict(zip(['NDCG', 'MAP', 'MRR'], [round(v, 4) for v in self.get_mean()]))

    def is_better_than(self, other):
        if other is None:
            return True
        assert isinstance(other, MeanRankScores)
        return sum(self.get_mean().values()) > sum(other.get_mean().values())
        # mask = np.array([1, 1, 1])
        # return sum(self.get_mean() * mask) > sum(other.get_mean() * mask)
