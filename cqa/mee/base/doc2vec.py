from utils import tmu, mu
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from cqa.data.datasets import DataSo, DataZh


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


if __name__ == '__main__':
    mu.multi_process(train_model, [(DataSo,), (DataZh,), ])
