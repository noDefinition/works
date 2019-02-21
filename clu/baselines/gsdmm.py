import pandas as pd

from clu.baselines.lda import run_multi_and_output
from clu.data.datasets import *
from utils.id_freq_dict import IdFreqDict


class GSDMM:
    def __init__(self):
        self.alpha = self.alpha0 = self.beta = self.beta0 = self.K = self.iter_num = None
        self.cludict = self.twharr = None
        self.nmi_list, self.ari_list, self.acc_list = list(), list(), list()

    def set_hyperparams(self, alpha, beta, k, iter_num, seed):
        self.alpha0 = alpha * k
        self.alpha, self.beta, self.K, self.iter_num = alpha, beta, k, iter_num
        self.cludict = dict([(i, GSDMM.ClusterHolder(i)) for i in range(k)])
        np.random.seed(seed)

    def input_docarr(self, docarr):
        self.twharr = [GSDMM.TweetHolder(d) for d in docarr]
        vocab_size = len(set(au.merge([d.tokenids for d in docarr])))
        # print('vocab_size:{}'.format(vocab_size))
        self.beta0 = self.beta * vocab_size

    def sample(self, twh, doc_num, using_max=False):
        alpha, beta, alpha0, beta0 = self.alpha, self.beta, self.alpha0, self.beta0
        cluids = list()
        probs = list()
        tw_word_freq = twh.ifd.word_freq_enumerate(newest=False)
        for cluid, cluster in self.cludict.items():
            clu_tokens = cluster.tokens
            n_zk = clu_tokens.get_freq_sum() + beta0
            clu_prob_prefix = (cluster.twnum + alpha) / (doc_num - 1 + alpha0)
            prob_delta = 1.0
            ii = 0
            for word, freq in tw_word_freq:
                n_zwk = clu_tokens.freq_of_word(word) if clu_tokens.has_word(word) else 0
                if freq == 1:
                    prob_delta *= (n_zwk + beta) / (n_zk + ii)
                    ii += 1
                elif freq > 1:
                    for jj in range(freq):
                        prob_delta *= (n_zwk + beta + jj) / (n_zk + ii)
                        ii += 1
            cluids.append(cluid)
            probs.append(clu_prob_prefix * prob_delta)
        # import warnings
        # warnings.filterwarnings("error")
        # try:
        #     int(np.random.choice(a=cluids, p=np.array(probs) / np.sum(probs)))
        # except:
        #     print('fuck')
        #     print(probs)
        if using_max:
            return cluids[np.argmax(probs)]
        else:
            if np.sum(probs) == 0.0:
                choice = int(np.random.choice(a=cluids))
            else:
                choice = int(np.random.choice(a=cluids, p=np.array(probs) / np.sum(probs)))
            return choice

    def fit(self):
        iter_num = self.iter_num
        twharr, cludict = self.twharr, self.cludict
        D = len(twharr)
        """ start iteration """
        for i in range(iter_num):
            for twh in twharr:
                twh.update_cluster(None)
                cluid = self.sample(twh, D, using_max=(i >= iter_num - 5))
                twh.update_cluster(cludict[cluid])
            # print('  {} th epoch'.format(i))
            # print(au.score(topics, clusters, score_type='nmi'))
            topics, clusters = self.get_topics_and_clusters()
            self.nmi_list.append(au.score(topics, clusters, 'nmi'))
            self.ari_list.append(au.score(topics, clusters, 'ari'))
            self.acc_list.append(au.score(topics, clusters, 'acc'))

    def get_topics_and_clusters(self):
        topic_cluster = [(twh.topic, twh.get_cluid()) for twh in self.twharr]
        topics, clusters = list(zip(*topic_cluster))
        return topics, clusters

    def get_score_lists(self):
        return self.nmi_list, self.ari_list, self.acc_list

    def print_top_words(self, ifd):
        from collections import OrderedDict as Od
        cluid2counter = Od((twh.cluster.cluid, Counter()) for twh in self.twharr)
        for twh in self.twharr:
            cluid2counter[twh.cluster.cluid][twh.topic] += 1
        print('total cluster number: {}'.format(len(self.cludict)))
        clu2distrib = Od()
        for cluid in sorted(cluid2counter.keys()):
            topic_distrib = cluid2counter[cluid].most_common()
            topic = topic_distrib[0][0]
            print('cluid: {}, topic: {}, t distrib: {}'.format(cluid, topic, topic_distrib[:10]))
            word_distrib = self.cludict[cluid].tokens.most_common()[:60]
            valid_words = [ifd.id2word(wid) for wid, cnt in word_distrib]
            print(' '.join(valid_words))
            clu2distrib[cluid] = {'topic_distrib': topic_distrib, 'word_distrib': valid_words}
        return clu2distrib

    class ClusterHolder:
        def __init__(self, cluid):
            self.cluid = cluid
            self.tokens = IdFreqDict()
            self.twnum = 0

        def update_by_twh(self, twh, factor):
            twh_tokens = twh.ifd
            if factor > 0:
                self.tokens.merge_freq_from(twh_tokens, newest=False)
                self.twnum += 1
            else:
                self.tokens.drop_freq_from(twh_tokens, newest=False)
                self.twnum -= 1

    class TweetHolder:
        def __init__(self, doc):
            self.cluster = None
            self.text = doc.text
            self.topic = doc.topic
            self.tokenids = doc.tokenids
            self.ifd = IdFreqDict()
            for t in self.tokenids:
                self.ifd.count_word(t)

        def get_cluid(self):
            return self.cluster.cluid

        def update_cluster(self, cluster):
            if self.cluster is not None:
                self.cluster.update_by_twh(self, factor=-1)
            self.cluster = cluster
            if cluster is not None:
                cluster.update_by_twh(self, factor=1)


def run_gsdmm(d_class, gsdmm_kwargs):
    print(gsdmm_kwargs)
    docarr = d_class().load_docarr()
    g = GSDMM()
    g.set_hyperparams(**gsdmm_kwargs)
    g.input_docarr(docarr)
    g.fit()
    nmi_list, ari_list, acc_list = g.get_score_lists()
    gsdmm_kwargs.pop('seed')
    return gsdmm_kwargs, nmi_list, ari_list, acc_list


def run_gsdmm_using_kwargs(d_class, result_file):
    nv_list = {
        DataTREC: [('alpha', [0.01]), ('beta', [0.01])],
        DataGoogle: [('alpha', [1, 0.1]), ('beta', [0.01])],
        DataEvent: [('alpha', [1, 0.1, 0.01]), ('beta', [1, 0.1, 0.01])],
        DataReuters: [('alpha', [1, 0.1, 0.01]), ('beta', [0.1])],
        Data20ng: [('alpha', [1, 0.1, 0.01]), ('beta', [0.1])],
    }[d_class]
    common = [('iter_num', [100]), ('k', [d_class().topic_num]),
              ('seed', [(i + 34) for i in range(3)])]
    args_list = [(d_class, g) for g in au.grid_params(nv_list + common)]
    print(len(args_list))
    run_multi_and_output(run_gsdmm, Nodes.max_cpu_num(), args_list, result_file)


def run_one_d_class(d_cls):
    print('run one : ', d_cls.name)
    ifd, docarr = d_cls().load_ifd_and_docarr()
    g = GSDMM()
    gsdmm_kwargs = dict(alpha=0.1, beta=0.01, k=d_cls.topic_num, iter_num=100, seed=764)
    g.set_hyperparams(**gsdmm_kwargs)
    g.input_docarr(docarr)
    g.fit()
    clu2distrib = g.print_top_words(ifd)
    iu.dump_json('{}_gsdmm_clu_distrib.txt'.format(d_cls.name), clu2distrib)
    print('run over : ', d_cls.name)


def one_run_for_word_distribution():
    from utils import mu
    # run_one_d_class(DataTREC)
    mu.multi_process(run_one_d_class, [(d,) for d in [DataTREC, DataGoogle, DataEvent]])


if __name__ == '__main__':
    from utils.node_utils import Nodes
    # one_run_for_word_distribution()
    # exit()

    for _d_class in [DataEvent]:
        print('Using data:', _d_class.name)
        _score_file = 'GSDMM_{}_score.txt'.format(_d_class.name)
        run_gsdmm_using_kwargs(_d_class, _score_file)

        df = pd.DataFrame(
            columns=['k', 'alpha', 'beta', 'nmi', 'ari', 'acc'],
            data=[
                (kwargs['k'], kwargs['alpha'], kwargs['beta'], nmi_list[-1], ari_list[-1],
                 acc_list[-1]) for kwargs, nmi_list, ari_list, acc_list in
                iu.load_array(_score_file)
            ]
        )
        df.sort_values(by='nmi', ascending=False, inplace=True)
        nmi_list, ari_list, acc_list = list(), list(), list()
        for _, df_ in au.group_data_frame(df, column='k'):
            nmi_list.append(au.mean_std(df_['nmi'].values[:3]))
            ari_list.append(au.mean_std(df_['ari'].values[:3]))
            acc_list.append(au.mean_std(df_['acc'].values[:3]))
            print(df_)

        # print(_d_class.name)
        print('nmi: {:.4f}±{:.4f}'.format(*nmi_list[0]), end=', ', flush=True)
        print('ari: {:.4f}±{:.4f}'.format(*ari_list[0]), end=', ', flush=True)
        print('acc: {:.4f}±{:.4f}'.format(*acc_list[0]))
