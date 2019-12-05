from collections import defaultdict

from gensim.corpora import Dictionary
from gensim.matutils import dirichlet_expectation
from gensim.models import AuthorTopicModel as ATM

from uclu.data.datasets import *
from utils import au, iu, tmu, tu


def fit_atm(out_file: str, d_class, add_body: bool, kwargs: dict, **_):
    print(add_body, kwargs)
    smp = Sampler(d_class)
    smp.load_basic()
    docarr = smp.docarr
    for doc in docarr:
        doc.all_wints = list(doc.title)
        if add_body:
            doc.all_wints += list(doc.flatten_body())

    corpus = [list(map(str, doc.all_wints)) for doc in docarr]
    dic = Dictionary(corpus)
    corpus = [dic.doc2bow(bow) for bow in corpus]
    author2doc = defaultdict(list)
    for i, doc in enumerate(docarr):
        author2doc[str(doc.uint)].append(i)

    atm = ATM(corpus=corpus, author2doc=author2doc, num_topics=d_class.topic_num, **kwargs)

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

    kwargs.update(au.scores(topics, clusters))
    extra_kwargs = {'addb': int(add_body), 'dn': d_class.name}
    # kwargs['addb'] = int(add_body)
    # kwargs['dn'] = d_class.name
    kwargs.update(extra_kwargs)
    print('----over---->', kwargs)
    iu.dump_array(out_file, [kwargs], mode='a')


def main():
    out_file = 'atm_{}.json'.format(tmu.format_date()[2:])
    iu.remove(out_file)
    n_rerun = 1
    od_list = tu.LY({
        'random_state': [i + np.random.randint(0, 1000) for i in range(n_rerun)],
        'alpha': [1, 0.1, 0.01],
        'eta': [1, 0.1, 0.01],
        'iterations': [100],
        'passes': [10],
    }).eval()

    args = [
        (out_file, dcls, addb, od)
        for od in od_list
        # for addb in [True, False]
        # for dcls in [DataAu, DataSf]
        for addb in [True]
        for dcls in [DataZh]
    ]
    # mu.multi_process_batch(fit_atm, 9, args)
    from utils.tune.tune_utils import auto_gpu
    auto_gpu(fit_atm, args, {0: 9})


if __name__ == '__main__':
    main()
