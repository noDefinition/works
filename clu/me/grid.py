import clu.me.main as m
from clu.me import C
from clu.me.models import *
from clu.data.datasets import *
from utils import Nodes
from utils.tune.base_grid import BaseGrid
from utils.tune.tune_utils import LY


class GridClu(BaseGrid):
    def __init__(self):
        super(GridClu, self).__init__()
        self.datas, self.vers, self.gpu_ids, self.gpu_max = Nodes.select(
            nnew=(class_list, [AELstm], [0, 1], 3),
        )
        if self.is_debugging():
            self.datas, self.gpu_ids, self.gpu_max = Nodes.select(
                nnew=([DataTrec], [1], 1),
            )
        self.comment = self.get_comment(default='try_ae_lstm')
        self.copy_modules = [m] + self.vers

    def grid_od_list(self):
        dat_ly = LY({
            C.lid: list(range(1)), C.ep: [100], C.lr: [1e-4], C.pad: [1],
        }) * LY({
                    C.dn: [d.name],
                    C.cn: [d.topic_num],
                    C.bs: {
                        DataTrec: [64],
                        DataGoogle: [64],
                        DataEvent: [30],
                        DataReuters: [10],
                        Data20ng: [10],
                    }.get(d),
                    # C.bs: [64] if d in {DataTREC, DataGoogle, DataEvent} else [16],
                } for d in self.datas)
        ver_ly = LY()
        for v in self.vers:
            tmp_ly = LY({
                C.vs: [v.__name__],
            })
            # if v in {VAE1, VAE3}:
            #     tmp_ly *= LY({
            #                      C.ptp: [0],
            #                      C.smt: [1, 0.5, 0.1],
            #                      # C.ckl: [0, 0.1, 0.5],
            #                      C.cini: [1],
            #                      C.wini: [1],
            #                      # C.cini: [0, 1],
            #                      # C.wini: [0, 1],
            #                      # }) * LY({
            #                      # }, {
            #                      #     C.ptp: [1],
            #                      #     C.kalp: [1, 2],
            #                      C.ed: [ed],
            #                      C.hd: [ed],
            #                  } for ed in [64, 128])
            # if v in {VAE2}:
            #     tmp_ly *= LY({
            #         #     C.ckl: [0],
            #         #     C.regmu: [0],
            #         # }, {
            #         C.ckl: [0, 0.001, 0.01, 0.1, 1, 10],
            #         C.regmu: [0],
            #     }) * LY({
            #                 C.cini: [1],
            #                 C.wini: [1],
            #                 C.smt: [1],
            #                 C.ed: [ed],
            #                 C.hd: [ed],
            #             } for ed in [32, 128])
            # if v in {VAE4}:
            #     tmp_ly *= LY({
            #                      C.smt: [1, 0.5, 0.1],
            #                      C.ckl: [0, 0.1, 0.5, 1],
            #                      C.psmp: [0, 1],
            #                      C.cini: [1],
            #                      C.wini: [1],
            #                      C.ed: [ed],
            #                      C.hd: [ed],
            #                  } for ed in [64, 128])
            # if v in {SBX}:
            #     tmp_ly *= LY({
            #         C.cini: [1], C.wini: [1],
            #     }, {
            #         C.cini: [0], C.wini: [0],
            #     }) * LY({
            #                 C.ed: [ed],
            #                 C.hd: [ed],
            #                 C.useb: [0, 1],
            #                 C.creg: [0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            #                 C.span: [5, 10, 20],
            #             } for ed in [16, 32, 64, 128, 256])
            if isinstance(v, VAE1):
                tmp_ly *= LY({
                    C.cini: [1],
                    C.wini: [1],
                })
            if v in {AeSpectral}:
                tmp_ly *= LY({
                    C.cini: [1],
                    C.wini: [1],
                    C.wtrn: [1],
                    C.ptn: [10, 30],
                    C.ptncmb: [0, 1, 2],
                }) * LY({
                    #     C.cspec: [0], C.cdecd: [0.1, 1], C.sigma: [0],
                    # }, {
                    #     C.cspec: [0.1, 1], C.cdecd: [0],
                    #     C.sigma: [0.1, 1],
                    # }, {
                    C.cspec: [0.1, 1], C.cdecd: [0.1, 1],
                    C.sigma: [0.1, 1],
                }) * LY({
                    C.ed: [64], C.hd: [64],
                })
            if v in {AeSpectral2}:
                tmp_ly *= LY({
                    C.sigma: [1],
                    C.cspec: [1],
                    C.cdecd: [1],
                    C.cpurt: [1],
                    C.ptn: [10],
                    C.ed: [64],
                    C.hd: [64],
                })
            if v in {AELstm}:
                tmp_ly *= LY({
                    # C.sigma: [1],
                    # C.cspec: [1],
                    # C.cdecd: [1],
                    C.lbsmt: [1, 0.5],
                    C.ptn: [10, 20],
                    C.ed: [64], C.hd: [64],
                })
            ver_ly += tmp_ly
        # com_ly = LY((
        #     (C.lid, list(range(1))), (C.ep, [80]), (C.lr, [5e-4]), (C.bs, [64]), (C.ns, [16]),
        # ))
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
        # if v == N6:
        #     part1 = LY((  # different cluster num
        #         (C.l3, [0]),
        #     ), (
        #         (C.l3, [1]), (C.worc, [1]), (C.eps, [50]),
        #     )) * LY((
        #         (C.l1, [1]), (C.l2, [1]),
        #     ))
        #     tmp_ly *= part1
        ret_ly = dat_ly * ver_ly
        return ret_ly.eval()


if __name__ == '__main__':
    GridClu().main()
