import numpy as np


class MeanRankScores:
    def __init__(self):
        self.es_list = list()  # (None, 3)

    def append(self, y_true, y_pred):
        self.es_list.append(RankScores(y_true, y_pred).es)

    def get_mean(self):
        return np.mean(self.es_list, axis=0)

    def to_dict(self):
        return dict(zip(['NDCG', 'MAP', 'MRR'], [round(v, 4) for v in self.get_mean()]))

    def is_better_than(self, other):
        if other is None:
            return True
        mask = np.array([1, 1, 1])
        return sum(self.get_mean() * mask) > sum(other.get_mean() * mask)


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
        self.es = [self.NDCG(5), self.MAP(5), self.MRR()]

    def NDCG(self, k=5):
        def dcg_score(ranking):
            gain = np.array([v[0] for v in ranking], dtype='float32')[:k]
            gain[gain >= 0] = gain[gain >= 0] + 1
            gain[gain < 0] = -1 / (gain[gain < 0] - 1)
            discounts = np.log2(np.arange(min(k, len(gain))) + 2)
            return np.sum(gain / discounts)

        dcg = dcg_score(self.sorted_tpr)
        idcg = dcg_score(self.tpr)
        return 1 if idcg < 1e-7 else dcg / idcg

    def MAP(self, k=5):
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

    def MRR(self):
        for i, tpr in enumerate(self.sorted_tpr):
            if tpr[2] == 1:
                return 1 / (i + 1)
        return 1 / 0
