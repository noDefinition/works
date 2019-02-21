from clu.data.datasets import *
from clu.me import *
from clu.me.gen1 import *
from utils import Nodes, iu, tu


def get_args():
    parser = tu.get_common_parser()
    _ = parser.add_argument
    _(cn_, type=int, help='model: cluster num')
    _(sc_, type=float, help='model: scale for normal init')
    _(md_, type=int, default=None, help='model: model inner dim')
    _(bs_, type=int, help='runtime: pos batch size')
    _(ns_, type=int, help='runtime: neg batch num')

    _(l1_, type=float, default=None, help='model: lambda for negative samples')
    _(l2_, type=float, default=None, help='model: lambda for cluster similarity')
    _(l3_, type=float, default=None, help='model: lambda for point-wise loss')
    _(l4_, type=float, default=None, help='model: lambda for regularization loss')

    _(mgn_, type=float, default=1., help='model: margin in hinge loss')
    _(bn_, type=int, default=None, help='model: if use batch normalization')
    _(smt_, type=float, default=None, help='model: smoothness of cross entropy')

    _(wini_, type=int, default=None, help='word embed 0:random, 1:pre-trained')
    _(cini_, type=int, default=None, help='cluster embed 0:random, 1:pre-trained')
    _(wtrn_, type=int, default=None, help='if train word embedding')
    _(ctrn_, type=int, default=None, help='if train cluster embedding')

    _(ptn_, type=int, default=None, help='epoch num for pre-training')
    _(worc_, type=int, default=None, help='0:word noise, 1: embed noise')
    _(eps_, type=float, default=None, help='epsilon for adv grad')
    _(tpk_, type=int, default=None, help='top k for word feature')
    return tu.parse_args(parser)


def grid_od_list(datas, vers):
    def diff_c_num(clu_num):
        return [int(i * clu_num / 10) for i in range(2, 10, 2)] + \
               [int(i * clu_num / 2) for i in range(2, 11)]
        # return (np.array([int(i * clu_num / 2) for i in range(1, 11)])).astype(int).tolist()

    LY = tu.LY
    com_ly = LY(((lid_, list(range(5))), (ep_, [100]), (bs_, [64]), (ns_, [16]),))
    # dat_ly = LY(*[((dn_, [d.name]), (cn_, [d.topic_num]),) for d in datas])
    dat_ly = LY(*[((dn_, [d.name]), (cn_, diff_c_num(d.topic_num)),) for d in datas])
    ver_ly = LY()
    for v in vers:
        tmp_ly = LY(((vs_, [v.__name__]),))
        # if v == N6:
        #     part1 = LY((
        #         (l3_, [0]),
        #     ), (
        #         # (eps_, [1, 2, 5, 8, 10, 20, 50, 80, 100, 200, 500]),
        #         (l3_, [1]), (worc_, [1]), (eps_, [50]),
        #     )) * LY((
        #         (l1_, [0]), (l2_, [1]),  # ARL w/o pairwise
        #     ), (
        #         (l1_, [1]), (l2_, [0]),  # ARL w/o pointwise
        #     ), (
        #         (l1_, [1]), (l2_, [1]),  # ARL-Adv
        #     ))
        #     part2 = LY((  # ARL
        #         (l1_, [1]), (l2_, [1]), (l3_, [0]),
        #     ))
        #     part3 = LY((  # ARL-Adv(word)
        #         (l1_, [1]), (l2_, [1]), (l3_, [1]), (worc_, [0]), (eps_, [50]),
        #     ))
        #     tmp_ly *= (part1 + part2 + part3)
        # if v == N7:
        #     tmp_ly *= LY((  # ARL-Random
        #         (l1_, [1]), (l2_, [1]), (l3_, [1]), (worc_, [1]), (eps_, [50]),
        #     ))
        # if v == N6:
        #     part1 = LY((
        #         (l3_, [0]),
        #     ), (
        #         (l3_, [1]), (worc_, [1]), (eps_, [10, 20, 50]),
        #         # (eps_, [20, 30, 50]),
        #         # (eps_, [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
        #         # ), (
        #         #     (eps_, [20, 30, 50]), (worc_, [1]),
        #         #     (l3_, [0.1, 0.2, 0.5, 0.8, 1, 2, 5, 8, 10, 20, 50]),
        #     )) * LY((
        #         # (l1_, [0, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]), (l2_, [1]),
        #     # ), (
        #     #     (l1_, [1]), (l2_, [0, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]),
        #     # ), (
        #         (l1_, [1]), (l2_, [1]),
        #     )) * LY((
        #         (wtrn_, [0]), (ctrn_, [1]),
        #     ), (
        #         (wtrn_, [1]), (ctrn_, [0]),
        #     ), (
        #         (wtrn_, [1]), (ctrn_, [1]),
        #     ))
        #     tmp_ly *= part1
        if v == N6:
            part1 = LY((  # different cluster num
                (l3_, [0]),
            ), (
                (l3_, [1]), (worc_, [1]), (eps_, [50]),
            )) * LY((
                (l1_, [1]), (l2_, [1]),
            ))
            tmp_ly *= part1
        ver_ly += tmp_ly
    return (com_ly * dat_ly * ver_ly).eval()


def main():
    datas, gpu_ids, gpu_max, vers = Nodes.select(
        # n1702=([DataEvent, DataTREC, DataGoogle], [0, 1], 3, [N6, N7]),
        ngpu=([DataEvent, DataTREC, DataGoogle], [0, 1, 2, 3], 2, [N6]),
    )
    names = [v.__name__ for v in vers] + [d.name for d in datas] + ['c_num_3']
    iu.rmtree('./__pycache__')
    log_path = tu.get_log_path(names, make_new=True)
    iu.copy(__file__, log_path)
    for v in vers:
        iu.copy(v.file, log_path)
    od_list = grid_od_list(datas, vers)
    od_list = tu.update_od_list(od_list, log_path, shuffle=True)
    print('log_path: {}, arg num: {}'.format(log_path, len(od_list)))
    offset = 0
    tu.run_od_list('python3.6 main.py ', od_list[offset:], gpu_ids, gpu_max)
    iu.rename(log_path, log_path + '_over')


if __name__ == '__main__':
    main()
