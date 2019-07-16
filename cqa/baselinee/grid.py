from cqa.baselinee import J
from cqa.baselinee.b1 import B1
from cqa.baselinee.counterpart import AAAI15, AAAI17, IJCAI15
from cqa.mee.grid import DataSo, DataZh
from utils import Nodes, au, tu, iu


def get_args():
    parser = tu.get_common_parser()
    _ = parser.add_argument
    _(J.bs, type=int, help='runtime: batch size')
    _(J.es, type=int, help='runtime: step before early stop')
    _(J.fda, type=int, default=0, help='if use full data, 0=no')
    _(J.sc, type=float, help='model: scale for normal init')
    _(J.lr, type=float, default=1e-3, help='train: learning rate')

    _(J.dmu, type=int, help='data multiply for number of pairs')
    _(J.reg, type=float, default=0., help='coeff of regularization')
    _(J.drp, type=float, default=0., help='probability of dropout')
    return tu.parse_args(parser)


def grid_od_list(datas, vers, use_fda):
    common = [
        # (J.lid_, [0, 1]),
        (J.dn, datas),
        (J.vs, vers),
        (J.ep, [20]),
        (J.es, [8]),

        (J.fda, [use_fda]),
        (J.sc, [1.]),
        (J.bs, [512]),
        (J.lr, [3e-4]),

        (J.dmu_, [5]),
        (J.reg, [0, 1e-4]),
    ]
    return au.merge(au.grid_params(kv_pairs) for kv_pairs in [common])


def main():
    datas, gpu_ids, gpu_max, vers = Nodes.select(
        # n1702=([DataZh, DataSo], [0, 1], 1, [AAAI15, B1]),
        # n1702=([DataSo, DataZh], [0, 1], 1, [AAAI17]),
        ngpu=([DataZh, DataSo], [2, 3], 1, [IJCAI15]),
    )
    datas, vnames, memo = [d.name for d in datas], [v.__name__ for v in vers], ['full']
    log_path = tu.get_log_path(datas + vnames + memo, make_new=True)
    iu.copy(__file__, log_path)
    use_fda = 0 if len(gpu_ids) * gpu_max <= 1 else 1
    od_list = grid_od_list(datas, vnames, use_fda)
    od_list = tu.update_od_list(od_list, log_path, shuffle=True)
    tu.run_od_list('python3.6 main.py ', od_list, gpu_ids, gpu_max)


if __name__ == '__main__':
    main()
