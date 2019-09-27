from collections import Counter, defaultdict
from typing import List

import numpy as np
from gensim.corpora import Dictionary
from gensim.matutils import dirichlet_expectation
from gensim.models import AuthorTopicModel as ATM

from uclu.data.datasets import DataSo, Sampler
from uclu.data.document import Document
from utils import au, iu, mu, tu


def fit_atm(out_file: str, docarr: List[Document], add_body: bool, kwargs: dict):
    print(add_body, kwargs)
    # out_file = './atm_{}.json'.format('add_body' if add_body else 'no_body')
    for doc in docarr:
        body_gen = (wint for wints in doc.body for wint in wints)
        doc.all_texts = doc.title + list(body_gen) if add_body else doc.title

    corpus = [list(map(str, doc.all_texts)) for doc in docarr]
    dic = Dictionary(corpus)
    corpus = [dic.doc2bow(bow) for bow in corpus]
    author2doc = defaultdict(list)
    for i, doc in enumerate(docarr):
        author2doc[str(doc.uint)].append(i)

    atm = ATM(corpus=corpus, author2doc=author2doc, **kwargs)  # with training

    expElogtheta = np.exp(dirichlet_expectation(atm.state.gamma))  # author * topic
    expElogbeta = np.exp(dirichlet_expectation(atm.state.get_lambda()))  # topic * word
    topics = [doc.tag for doc in docarr]
    clusters = []
    for doc, bow in zip(docarr, corpus):
        uid = atm.author2id[str(doc.uint)]
        wids = np.array([wid for wid, _ in bow])
        atd = expElogtheta[uid].reshape(-1, 1)
        twd = expElogbeta[:, wids]
        atw = np.prod(atd * twd, axis=1)
        clu = np.argmax(atw)
        # clu_probs = atm.get_new_author_topics([bow])
        # if len(clu_probs) == 0:
        #     clu = 0
        # else:
        #     clus, probs = np.array(clu_probs).T
        #     amax = np.argmax(probs)
        #     clu = clus[amax]
        clusters.append(clu)

    ret = kwargs
    ret.update(au.scores(topics, clusters))
    ret['addb'] = int(add_body)
    iu.dump_array(out_file, [ret], mode='a')
    print('----over---->', ret)


def main():
    out_file = './atm.json'
    iu.remove(out_file)

    smp = Sampler(DataSo)
    smp.load_basic()
    # docarr = smp.docarr[:10000]
    docarr = smp.docarr

    ratios = np.array([1])
    rerun_num = 1
    kwargs = tu.LY({
        'random_state':
        [i + np.random.randint(0, 1000) for i in range(rerun_num)],
        'passes': [3],
        'iterations': [30],
        # 'passes': [1], 'iterations': [1],
        'alpha': [1, 0.1, 0.01],
        'eta': [1, 0.1, 0.01],
        'num_topics':
        list(map(int, (ratios * DataSo.topic_num))),
    }).eval()

    args = [(out_file, docarr, add_body, kwarg) for kwarg in kwargs
            for add_body in [True, False]]
    mu.multi_process(fit_atm, args)


if __name__ == '__main__':
    main()
