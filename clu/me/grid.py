from clu.me import C
from clu.me.gen1 import *
from clu.data.datasets import *
from utils import Nodes
from utils.tune.base_grid import BaseGrid
from utils.tune.tune_utils import LY


class GridClu(BaseGrid):
    def __init__(self):
        super(GridClu, self).__init__()
        import clu.me.main as m
        self.datas: List[Data]
        self.datas, self.gpu_ids, self.gpu_max, self.vers = Nodes.select(
            # ngpu=([DataEvent, DataTREC, DataGoogle], [0, 1, 2, 3], 1, [VAE1]),
            # ngpu=(class_list, [1, 2, 3], 1, [VAE3]),
            nnew=(class_list, [0, 1], 1, [VAE4]),
        )
        self.comment = self.get_comment(default='neg_sample_tune_ed')
        self.modules_to_copy = [m] + self.vers

    def grid_od_list(self):
        com_ly = LY({
            C.lid: [1],
            C.ep: [80],
            C.lr: [5e-4],
            C.pad: [1],
        })
        dat_ly = LY({
                        C.dn: [d.name],
                        C.cn: [d.topic_num],
                        # C.pad: [d.seq_len],
                        C.bs: [128] if d in {DataTREC, DataGoogle, DataEvent} else [32],
                    } for d in self.datas)
        ver_ly = LY()
        for v in self.vers:
            tmp_ly = LY({
                C.vs: [v.__name__],
            })
            if v in {VAE1, VAE2, VAE3}:
                tmp_ly *= LY({
                    C.ptp: [0],
                    C.smt: [1, 0.5, 0.1],
                    # C.ckl: [0, 0.1, 0.5],
                    C.cini: [1],
                    C.wini: [1],
                    # C.cini: [0, 1],
                    # C.wini: [0, 1],
                    # }) * LY({
                    # }, {
                    #     C.ptp: [1],
                    #     C.kalp: [1, 2],
                }) * LY({
                            C.ed: [ed],
                            C.hd: [ed],
                        } for ed in [64, 128])
            if v in {VAE4}:
                tmp_ly *= LY({
                       C.smt: [1, 0.5, 0.1],
                       C.ckl: [0, 0.1, 0.5, 1],
                       C.psmp: [0, 1],
                       C.cini: [1],
                       C.wini: [1],
                       C.ed: [ed],
                       C.hd: [ed],
                   } for ed in [64, 128])
            ver_ly += tmp_ly
        # com_ly = LY((
        #     (C.lid, list(range(1))), (C.ep, [80]), (C.lr, [5e-4]),
        #     # (C.bs, [64]), (C.ns, [16]),
        # ))
        # dat_ly = LY((
        #                 (C.dn, [d.name]),
        #                 (C.cn, [d.topic_num]),
        #                 (C.bs, [128] if d in {DataTREC, DataGoogle, DataEvent} else [32]),
        #             ) for d in self.datas)
        # ver_ly = LY()
        # for v in self.vers:
        #     tmp_ly = LY((
        #         (C.vs, [v.__name__]),
        #     ))
        #     # if v == N6:
        #     #     part1 = LY((
        #     #         (l3_, [0]),
        #     #     ), (
        #     #         # (eps_, [1, 2, 5, 8, 10, 20, 50, 80, 100, 200, 500]),
        #     #         (l3_, [1]), (worc_, [1]), (eps_, [50]),
        #     #     )) * LY((
        #     #         (l1_, [0]), (l2_, [1]),  # ARL w/o pairwise
        #     #     ), (
        #     #         (l1_, [1]), (l2_, [0]),  # ARL w/o pointwise
        #     #     ), (
        #     #         (l1_, [1]), (l2_, [1]),  # ARL-Adv
        #     #     ))
        #     #     part2 = LY((  # ARL
        #     #         (l1_, [1]), (l2_, [1]), (l3_, [0]),
        #     #     ))
        #     #     part3 = LY((  # ARL-Adv(word)
        #     #         (l1_, [1]), (l2_, [1]), (l3_, [1]), (worc_, [0]), (eps_, [50]),
        #     #     ))
        #     #     tmp_ly *= (part1 + part2 + part3)
        #     # if v == N6:
        #     #     part1 = LY((
        #     #         (l3_, [0]),
        #     #     ), (
        #     #         (l3_, [1]), (worc_, [1]), (eps_, [10, 20, 50]),
        #     #         # (eps_, [20, 30, 50]),
        #     #         # (eps_, [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
        #     #         # ), (
        #     #         #     (eps_, [20, 30, 50]), (worc_, [1]),
        #     #         #     (l3_, [0.1, 0.2, 0.5, 0.8, 1, 2, 5, 8, 10, 20, 50]),
        #     #     )) * LY((
        #     #         # (l1_, [0, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]), (l2_, [1]),
        #     #     # ), (
        #     #     #     (l1_, [1]), (l2_, [0, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]),
        #     #     # ), (
        #     #         (l1_, [1]), (l2_, [1]),
        #     #     )) * LY((
        #     #         (wtrn_, [0]), (ctrn_, [1]),
        #     #     ), (
        #     #         (wtrn_, [1]), (ctrn_, [0]),
        #     #     ), (
        #     #         (wtrn_, [1]), (ctrn_, [1]),
        #     #     ))
        #     #     tmp_ly *= part1
        #     # if v == N6:
        #     #     part1 = LY((  # different cluster num
        #     #         (C.l3, [0]),
        #     #     ), (
        #     #         (C.l3, [1]), (C.worc, [1]), (C.eps, [50]),
        #     #     )) * LY((
        #     #         (C.l1, [1]), (C.l2, [1]),
        #     #     ))
        #     #     tmp_ly *= part1
        #     if v in {VAE1, VAE2, VAE3}:
        #         tmp_ly *= LY((
        #             (C.pad, [1]),
        #             (C.smt, [1]),
        #             (C.ckl, [0]),
        #             # (C.smt, [0.1, 0.5, 1]),
        #             # (C.ckl, [0, 0.1, 0.5]),
        #         )) * LY((
        #             (C.ptp, [0]),
        #         ), (
        #             (C.ptp, [1]),
        #             (C.kalp, [1, 2]),
        #         )) * LY((
        #                     (C.ed, [ed]),
        #                     (C.hd, [ed]),
        #                 ) for ed in [64, 128, 256])
        #     ver_ly += tmp_ly
        # od_list = (com_ly * dat_ly).eval()
        # print((com_ly * dat_ly).eval())
        # return od_list
        return (com_ly * dat_ly * ver_ly).eval()


if __name__ == '__main__':
    GridClu().main()
