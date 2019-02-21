from cqa.baselinee import *
from cqa.baselinee.b1 import B1
from cqa.mee.grid import DataSo
from utils import Nodes, au, tu


def get_args():
    parser = tu.get_common_parser()
    _ = parser.add_argument
    _(bs_, type=int, help='runtime: batch size')
    _(es_, type=int, help='runtime: step before early stop')
    _(fda_, type=int, default=0, help='if use full data, 0=no')

    _(md_, type=int, help='model: model inner dim')
    _(sc_, type=float, help='model: scale for normal init')
    _(lr_, type=float, default=1e-3, help='train: learning rate')

    _(wit_, type=int, help='model: word vec init type')
    _(wtn_, type=int, help='model: word vec train type')
    _(tmd_, type=int, help='model: text mode')

    _(dmu_, type=int, help='data multiply for number of pairs')
    _(itp_, type=int, help='type of initializer')
    return tu.parse_args(parser)


def grid_od_list(datas, vers, use_fda):
    common = [
        (dn_, datas),
        (vs_, vers),
        (ep_, [50]),
        (es_, [10]),

        (fda_, [use_fda]),
        (sc_, [1.]),
        (md_, [64]),
        (bs_, [1024]),

        (dmu_, [8]),
        (itp_, [1, 0]),

        (wit_, [1]),
        (wtn_, [1]),
        (tmd_, [1]),
    ]
    return au.merge(au.grid_params(kv_pairs) for kv_pairs in [common])


def main():
    datas, gpu_ids, gpu_max, vers = Nodes.select(
        # ngpu=([DataSo], [0, ], 2, [B1]),
        n1702=([DataSo], [0, 1], 2, [B1]),
    )
    datas, vers = [d.name for d in datas], [v.__name__ for v in vers]
    use_fda = 1 if len(gpu_ids) * gpu_max <= 1 else 1
    od_list = grid_od_list(datas, vers, use_fda)
    log_path = tu.get_log_path(vers, make_new=True)
    od_list = tu.update_od_list(od_list, log_path, shuffle=True)
    tu.run_od_list('python3.6 main.py ', od_list[0:], gpu_ids, gpu_max)


if __name__ == '__main__':
    main()
