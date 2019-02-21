from cqa.mee import *
from cqa.mee.models import V1, V2, V3
from cqa.data.datasets import DataSo, DataZh
from utils import Nodes, tu, iu


def get_args():
    parser = tu.get_common_parser()
    _ = parser.add_argument
    _(fda_, type=int, help='if use full data, 0=no, 1=full', required=True)
    _(es_, type=int, help='runtime: step before early stop')
    _(md_, type=int, help='model: model inner dim')
    _(sc_, type=float, help='model: scale for normal init')
    _(lr_, type=float, help='train: learning rate')

    _(winit_, type=int, default=None, help='initialize of word embedding')
    _(wtrn_, type=int, default=None, help='training of word embedding')
    # _(tmd_, type=int, default=None, help='text mode')

    _(ptp_, type=int, help='distribution type')
    _(mtp_, type=int, help='mut_att type: if transpose')
    _(atp_, type=int, help='final au_rep type')
    _(otp_, type=int, help='optimizer type')
    # _(aqm_, type=int, help='how a & q merge')
    return tu.parse_args(parser)


def grid_od_list(datas, vers, use_fda):
    LY = tu.LY
    common_ly = LY((
        (ep_, [100]),
        (es_, [10]),
        # (md_, [64]),
        (lr_, [5e-4]),
        (dn_, datas),
        (fda_, [use_fda]),
        (lid_, list(range(1))),
    ))
    vers_ly = LY()
    for v in vers:
        tmp_ly = LY(((vs_, [v]),))
        if v == V1.__name__:
            tmp_ly *= LY((
                (wtrn_, [1]),
                (winit_, [1]),
                # (otp_, [1]),
            ))
        if v == V2.__name__:
            p1 = LY((
                (wtrn_, [1]),
                (winit_, [1]),
                # (otp_, [1, ]),
            ))
            p2 = LY((
                (ptp_, [0]),
            ), (
                (ptp_, [1]),
                (atp_, [0]),
            ), (
                (ptp_, [1]),
                (mtp_, [0, 1]),
                (atp_, [1, 2]),
            ))
            tmp_ly *= (p1 * p2)
        if v == V3.__name__:
            tmp_ly *= LY((
                (winit_, [1]),
                (otp_, [1]),
            ))
        vers_ly += tmp_ly
    return (common_ly * vers_ly).eval()


def main():
    datas, gpu_ids, gpu_max, vers_ = Nodes.select(
        n1702=([DataZh, DataSo], [0, 1], 1, [V1]),
        # ngpu=([DataSo], [0, ], 1, [V3]),
    )
    datas, vers, comm = [d.name for d in datas], [v.__name__ for v in vers_], ['compare']
    log_path = tu.get_log_path(vers + datas + comm, make_new=True)
    iu.copy(__file__, log_path)
    for v in vers_:
        iu.copy(v.file, log_path)
    use_fda = 1 if len(gpu_ids) * gpu_max <= 1 else 1
    od_list = grid_od_list(datas, vers, use_fda)
    od_list = tu.update_od_list(od_list, log_path, shuffle=True)
    tu.run_od_list('python3.6 ./main.py ', od_list[0:], gpu_ids, gpu_max)


if __name__ == '__main__':
    main()
