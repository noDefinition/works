from typing import Union
import numpy as np
# from sklearn import ensemble
from scipy.sparse import csr_matrix, vstack
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor

from utils import au, mu, tu, iu
from cqa.mee import Args
from cqa.data.datasets import DataSo, DataZh, name2d_class
from cqa.mee.evaluate import MeanRankScores


class RegressArgs(Args):
    def __init__(self):
        super(RegressArgs, self).__init__()
        self.ne = self.add_arg('ne')
        self.lr = self.add_arg('lr')
        self.mss = self.add_arg('mss')
        self.msl = self.add_arg('msl')
        self.mf = self.add_arg('mf')
        self.md = self.add_arg('md')


R = RegressArgs()


class RegressCqa:
    def __init__(self, kwargs):
        self.dname = kwargs[R.dn]
        self.data: Union[DataSo, DataZh] = name2d_class[self.dname]()
        self.model = None

        self.ne = kwargs[R.ne]
        self.lr = kwargs[R.lr]
        self.mss = kwargs[R.mss]
        self.msl = kwargs[R.msl]
        self.md = kwargs[R.md]
        # self.mf = kwargs[R.mf]

        self.hist = list()
        self.best_mrs = None
        self.brk_cnt = 0

    def load_data(self):
        self.data.load_cdong_full()
        self.data.load_wid2idf()

    def set_model_gbrt(self):
        self.model = GradientBoostingRegressor(
            loss='ls', learning_rate=self.lr, n_estimators=self.ne,
            min_samples_split=self.mss, min_samples_leaf=self.msl,
            max_features='sqrt', max_depth=self.md, warm_start=True,
            n_iter_no_change=5, validation_fraction=0.1,
        )
        # self.model = GradientBoostingRegressor(
        #     loss='ls', learning_rate=1e-3, n_estimators=100, min_samples_split=8,
        #     min_samples_leaf=4, min_impurity_split=1e-6, max_features='sqrt',
        #     max_depth=8, warm_start=True, n_iter_no_change=5, validation_fraction=0.1,
        # )

    def get_x(self, al) -> csr_matrix:
        wid2idf = self.data.wid2idf
        X = list()
        for wids in al:
            idf = np.zeros(len(wid2idf))
            tf = np.zeros(len(wid2idf))
            for wid in wids:
                idf[wid] = wid2idf[wid]
                tf[wid] += 1
            tfidf = tf * idf
            X.append(tfidf)
        X = csr_matrix(X)
        return X

    def fit(self):
        # print('fuck')
        # import numpy as np
        # p = [9, 1, 1, 1]
        # X = np.random.choice([0, 1, 2, 3], p=np.array(p) / np.sum(p), size=[100, 64])
        # y = np.random.choice([0, 1, 2], size=[100])
        # X = csr_matrix(X)
        # self.model.fit(X, y)
        # for y1, y2 in zip(y, self.model.predict(X)):
        #     print(y1, y2)
        # return
        assert isinstance(self.model, GradientBoostingRegressor)
        # qid = self.data.get_train_qids()
        train_data = self.data.gen_train(shuffle=True)
        X = None
        Y = list()
        for bid, (ql, al, ul, vl) in enumerate(train_data):
            # if bid > 10000:
            #     print(self.evaluate(self.data.gen_valid()).to_dict())
            #     print(self.evaluate(self.data.gen_test()).to_dict())
                # break
            if bid > 0 and bid % 100 == 0:
                print('bid', bid)
            X_a = self.get_x(al)
            X = X_a if X is None else vstack([X, X_a])
            Y.extend(vl)
            if bid > 0 and bid % 10000 == 0:
                print(X.shape)
            # if bid > 0 and bid % 100 == 0:
            #     print('bid', bid)
        self.model.fit(X, Y)
        print('len(self.model.estimators_)', len(self.model.estimators_))

    def should_stop(self):
        v_mrs = self.evaluate(self.data.gen_valid())
        if v_mrs.is_better_than(self.best_mrs):
            self.brk_cnt = 0
            self.best_mrs = v_mrs
            t_mrs = self.evaluate(self.data.gen_test())
            self.hist.append(t_mrs)
        else:
            self.brk_cnt += 1
        return self.brk_cnt >= 5

    def evaluate(self, qauv_gen) -> MeanRankScores:
        mrs = MeanRankScores()
        for bid, (ql, al, ul, vl) in qauv_gen:
            X = self.get_x(ql)
            y = self.model.predict(X)
            mrs.append(vl, y)
        return mrs


def fit_one_model(kwargs):
    regress = RegressCqa(kwargs)
    regress.set_model_gbrt()
    regress.load_data()
    regress.fit()


def fit_models_multi():
    od_list = tu.LY((
        (R.dn, [d.name for d in [DataSo]]),
        (R.lr, [1e-3]),
        (R.ne, [100]),
        (R.mss, [8]),
        (R.msl, [4]),
        (R.md, [4]),
    )).eval()
    print('len(od_list)', len(od_list))
    fit_one_model(od_list[0])


if __name__ == '__main__':
    fit_models_multi()
