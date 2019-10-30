from collections import OrderedDict as Od

import pandas as pd
from scipy.sparse import vstack
from sklearn.decomposition import LatentDirichletAllocation as LDA

from uclu.data.datasets import *
from utils import iu, mu, tu


def run_lda(out_file, tf_matrix, topics, lda_kwargs: dict):
    print(lda_kwargs)
    ret = lda_kwargs.copy()
    x_new = LDA(**lda_kwargs, n_jobs=1).fit_transform(tf_matrix)
    clusters = np.argmax(x_new, axis=1)
    ret.update(au.scores(topics, clusters))
    iu.dump_array(out_file, [ret], mode='a')


def main(add_body: bool):
    d_class = DataAu
    smp = Sampler(d_class)
    smp.load(0, 0, 0, 0)
    smp.fit_sparse(add_body=add_body, tfidf=False, p_num=20)
    tf_mtx = vstack([d.tf for d in smp.docarr])
    topics = [d.tag for d in smp.docarr]

    log_name = 'lda_{}_{}.json'.format(d_class.name, 'add_body' if add_body else 'no_body')
    out_file = iu.join(d_class.name, log_name)
    iu.remove(out_file)

    ratios = np.array([1])
    iter_num = 100
    rerun_num = 1
    kwargs = tu.LY({
        'max_iter': [iter_num],
        'learning_method': ['batch'],
        'random_state': [i + np.random.randint(0, 1000) for i in range(rerun_num)],
        'doc_topic_prior': [0.1, 1, 0.01],
        'topic_word_prior': [10, 1, 0.1, 0.01],
        'n_components': list(map(int, (ratios * d_class.topic_num))),
    }).eval()
    args = [(out_file, tf_mtx, topics, kwarg) for kwarg in kwargs]
    mu.multi_process_batch(run_lda, 6, args)


if __name__ == '__main__':
    for addb in [True, False]:
        main(addb)
