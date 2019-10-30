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
            # nnew=(class_list, [N5], [0, 1], 2),
            n1702=([DataTrec, DataGoogle, DataEvent], [N5], [0, 1], 3),
        )
        if self.is_debug:
            self.datas, self.gpu_ids, self.gpu_max = [DataTrec], [1], 1
        self.comment = self.get_comment(default='aaai_rebuttal')
        self.copy_modules = [m] + self.vers

    def grid_od_list(self):
        data_ly = LY({
            # C.bs: {
            #     DataTrec: [64],
            #     DataGoogle: [64],
            #     DataEvent: [30],
            #     DataReuters: [10],
            #     Data20ng: [10],
            # }.get(d),
            C.bs: [64], C.dn: [d.name], C.cn: [d.topic_num], 
            # C.pad: [d.seq_len],
            C.pad: [0],
        } for d in self.datas)
        model_ly = LY()
        for v in self.vers:
            tmp_ly = LY({
                C.lid: [7, 23], C.ep: [100], C.lr: [1e-4], C.vs: [v.__name__],
            })
            if v in {N5}:
                tmp_ly *= LY({
                    C.trep: [1, 0],
                    C.clin: [1, 0],
                    C.l1: [1, 2],
                    C.bs: [64],
                    C.ns: [16],
                    C.ed: [300],
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
            if v in {AeSpectral3}:
                tmp_ly *= LY({
                    C.ptn: [0, 10, 20],
                    C.alpha: [0.1, 1],
                    C.beta: [0.1, 1],
                    C.gamma: [0.1, 0.5, 0.9],
                    C.ed: [64], C.hd: [64],
                })
            if v in {AELstm}:
                tmp_ly *= LY({
                    C.ptn: [20],
                    C.lbsmt: [1, 0.5, 0.1],
                    C.alpha: [0.1, 0.5, 0.9],
                    C.ed: [64], C.hd: [64],
                })
            if v in {AeSoft}:
                tmp_ly *= LY({
                    C.ptncmb: [0, 1],
                }) * LY({
                    C.ed: [16], C.hd: [16],
                }, {
                    C.ed: [64], C.hd: [64],
                }, {
                    C.ed: [128], C.hd: [128],
                }, {
                    C.ed: [256], C.hd: [256],
                })
            model_ly += tmp_ly
        ret_ly = data_ly * model_ly
        iu.json.dump([dict(p) for p in ret_ly.pairs_list], open('fuckk.json', mode='w'), indent=4)
        return ret_ly.eval()


if __name__ == '__main__':
    GridClu().main()
