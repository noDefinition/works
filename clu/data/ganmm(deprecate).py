from collections import OrderedDict as Od
import numpy as np
from sklearn import metrics
from sklearn.utils.linear_assignment_ import linear_assignment
import pickle


def get_score(y_true, y_pred, using_score):
    func = {
        'acc': acc,
        'ari': metrics.adjusted_rand_score,
        'nmi': metrics.normalized_mutual_info_score,
    }[using_score.lower()]
    return round(float(func(y_true, y_pred)), 4)


def acc(y_true, y_pred):
    def reindex(array):
        item2idx = dict((item, idx) for idx, item in enumerate(sorted(set(array))))
        return [item2idx[item] for item in array]

    y_true, y_pred = reindex(y_true), reindex(y_pred)
    assert len(y_true) == len(y_pred)
    d = max(max(y_true), max(y_pred)) + 1
    w = np.zeros((d, d), dtype=np.int64)
    for y_t, y_p in zip(y_true, y_pred):
        w[y_t][y_p] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i][j] for i, j in ind]) / len(y_pred)


def main():
    # print出来的结果最好也输出到文件里
    base = '/home/cdong/works/research/clu/data/'
    files = [
        '20ng_tf.pkl', '20ng_tfidf.pkl', 'Event_tf.pkl', 'Event_tfidf.pkl',
        'Google_tf.pkl', 'Google_tfidf.pkl', 'Reuters_tf.pkl', 'Reuters_tfidf.pkl',
        'TREC_tf.pkl', 'TREC_tfidf.pkl',
    ]
    for file in files:
        file = base + file
        print(file)
        scores = list()
        for i in range(3):
            features, y_true = pickle.load(open(file, 'rb'))
            y_pred = GANMM(features)
            score = Od((s, get_score(y_true, y_pred, s)) for s in ('acc', 'ari', 'nmi'))
            scores.append(score)
            print(score)
        name2scores = Od()
        for score in scores:
            for score_name, score_value in score.items():
                name2scores.setdefault(score_name, list()).append(score_value)
            for score_name, score_values in name2scores.items():
                print('mean {}: {}\n'.format(score_name, np.mean(score_values)))


if __name__ == '__main__':
    main()
