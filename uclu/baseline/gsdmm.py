from utils.doc_utils import Document

from uclu.data.datasets import *
from clu.baselines.gsdmm import GSDMM
from utils import iu, mu, tu


def run_gsdmm(out_file, docarr: List[Document], kwargs: dict):
    print(kwargs)
    g = GSDMM()
    g.set_hyperparams(**kwargs)
    g.input_docarr(docarr)
    g.fit()
    nmi_list, ari_list, acc_list = g.get_score_lists()

    def mmm(arr):
        return np.mean(sorted(arr)[-10:])

    scores = {au.s_acc: mmm(acc_list), au.s_ari: mmm(ari_list), au.s_nmi: mmm(nmi_list), }
    kwargs.update(scores)
    iu.dump_array(out_file, [kwargs], mode='a')


def main():
    add_body = True
    out_file = 'gsdmm_{}.json'.format('add_body' if add_body else 'no_body')

    iu.remove(out_file)
    smp = Sampler(DataSo)
    smp.load_basic()

    docarr = []
    len_sum = 0
    for doc in smp.docarr:
        dd = Document()
        dd.tokenids = doc.title + list(doc.flat_body())[:200]
        dd.topic = doc.tag
        docarr.append(dd)
        len_sum += len(dd.tokenids)
    print(':.2f'.format(len_sum / len(docarr)))
    print('docarr built')

    rerun_num = 2
    iter_num = 100
    ratios = np.array([1])
    # ratios = np.concatenate([np.arange(0.2, 1, 0.2), np.arange(1, 5.1, 0.5)])
    od_list = tu.LY({
        'iter_num': [iter_num],
        'seed': [(i + np.random.randint(0, 10000)) for i in range(rerun_num)],
        'alpha': [1, 0.1, 0.01],
        'beta': [1, 0.1, 0.01],
        'k': list(map(int, (ratios * DataSo.topic_num))),
    }).eval()
    args = [(out_file, docarr, od) for od in od_list]
    mu.multi_process_batch(run_gsdmm, 20, args)


if __name__ == '__main__':
    main()
    
