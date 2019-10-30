from clu.baselines.gsdmm import GSDMM
from uclu.data.datasets import *
from utils import iu, mu, tu
from utils.doc_utils import Document


def run_gsdmm(out_file, d_class: Data, add_body: bool, kwargs: dict):
    print(kwargs)
    smp = Sampler(d_class)
    smp.load_basic()
    docarr = []
    len_sum = 0
    for doc in smp.docarr:
        dd = Document()
        dd.tokenids = doc.title
        if add_body:
            dd.tokenids += list(doc.flatten_body())[:200]
        dd.topic = doc.tag
        docarr.append(dd)
        len_sum += len(dd.tokenids)
    print('average doc len: {:.2f}'.format(len_sum / len(docarr)))

    g = GSDMM()
    g.set_hyperparams(k=d_class.topic_num, **kwargs)
    g.input_docarr(docarr)
    g.fit()
    nmi_list, ari_list, acc_list = g.get_score_lists()

    def mmm(arr):
        return np.mean(sorted(arr)[-10:])

    scores = {au.s_acc: mmm(acc_list), au.s_ari: mmm(ari_list), au.s_nmi: mmm(nmi_list)}
    kwargs.update(scores)
    kwargs['addb'] = int(add_body)
    kwargs['dn'] = d_class.name
    print('----over---->', kwargs)
    iu.dump_array(out_file, [kwargs], mode='a')


def main():
    out_file = './gsdmm.json'
    iu.remove(out_file)

    rerun_num = 1
    # ratios = np.concatenate([np.arange(0.2, 1, 0.2), np.arange(1, 5.1, 0.5)])
    od_list = tu.LY({
        'iter_num': [100],
        'seed': [(i + np.random.randint(0, 10000)) for i in range(rerun_num)],
        'alpha': [1, 0.1, 0.01],
        'beta': [1, 0.1, 0.01],
    }).eval()
    args = [
        (out_file, _d_class, add_body, od)
        for od in od_list
        for add_body in [True, False]
        for _d_class in [DataAu, DataSf]
    ]
    mu.multi_process_batch(run_gsdmm, 20, args)


if __name__ == '__main__':
    main()
