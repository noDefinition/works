import uclu.me.main as m
from uclu.data.datasets import *
from uclu.me import U
from uclu.me.models import *
from utils import Nodes
from utils.tune.base_grid import BaseGrid
from utils.tune.tune_utils import LY


class Grid(BaseGrid):
    def __init__(self):
        super(Grid, self).__init__()
        # datas = [DataSo, DataZh, DataSf, DataAu]
        datas = [DataZh, DataSo]
        self.datas, self.vers, self.gpu_ids, self.gpu_max = Nodes.select(
            n1702=(datas, [UAtt], [0, 1], 2),
            nnew=(datas, [UAtt], [0, 1], 2),
        )
        if self.is_debug:
            self.gpu_ids, self.gpu_max = [1], 1
        self.comment = self.get_comment(default='basic_ablation')
        self.copy_modules = [m] + self.vers

    def grid_od_list(self):
        data_ly = LY({U.dn: [d.name], U.cn: [d.topic_num]} for d in self.datas)
        model_ly = LY()
        for v in self.vers:
            tmp_ly = LY({
                U.lid: [1, 2], U.vs: [v.__name__],
                U.ep: [100, ], U.bs: [64], U.lr: [1e-3, 1e-2, ],
            })
            # if v in {V2}:
            #     tmp_ly *= LY({
            #         U.ed: [64, 128],
            #         U.tpad: [30],
            #         U.bpad: [200],
            #         U.addb: [0, 1],
            #         U.addu: [0, 1],
            #         U.pair: [0],
            #     })
            if v in {UAtt}:
                tmp_ly *= LY({
                    U.ed: [128], U.tpad: [30],  # U.bpad: [100, 300],
                    U.bpad: Nodes.select(nnew=[100], n1702=[300]),
                }) * LY({
                    U.addb: [0.1, 0.5, 1],
                    U.addu: [0.1, 0.5, 1],
                    U.addpnt: [0.1, 0.5, 1],
                })
            if v in {V3}:
                tmp_ly *= LY({
                    U.ed: [32, 64, 128], U.tpad: [20],
                    U.bpad: Nodes.select(nnew=[100], n1702=[300]),
                }) * LY({
                    U.addu: [1, 0], U.addb: [1, 0], U.addpnt: [1, 0],
                    # U.addu: [1], U.addb: [1],
                    # U.addpnt: [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, ],
                    #     U.addu: [1], U.addb: [1], U.addpnt: [1, 0],
                    # }, {
                    #     U.addu: [1], U.addb: [0], U.addpnt: [1, 0],
                    #     # U.addb: [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 2, 5, 10, 0, ],
                    # }, {
                    #     U.addu: [0], U.addb: [1, 0], U.addpnt: [1],
                    #     # U.addu: [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 2, 5, 10, 0, ],
                })
            if v in {V4}:
                tmp_ly *= LY({
                    U.ed: [64], U.tpad: [20], U.bpad: [100],
                }) * LY({
                    U.addb: [1, 0.1, 0.01, 10, 0],
                    U.addpnt: [1],
                }, {
                    U.addb: [1],
                    U.addpnt: [0.1, 0.01, 10, 0],
                })
            if v in {V5}:
                tmp_ly *= LY({
                    U.ed: [64], U.tpad: [20], U.bpad: [100],
                }) * LY({
                    U.addb: [1, 0.1, 0.01, 10, 0],
                    U.addpnt: [1],
                }, {
                    U.addb: [1],
                    U.addpnt: [0.1, 0.01, 10, 0],
                })
            # if v in {N1}:
            #     tmp_ly *= LY({
            #         U.ed: [32, 64, 128],
            #         U.tpad: [20], U.bpad: [100],
            #     }) * LY({
            #         U.addu: [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0],
            #         U.addb: [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0],
            #         # U.addpnt: [10, 1, 0.1, 0.01, 0],
            #     })
            # if v in {T2}:
            #     tmp_ly *= LY({
            #         U.ed: [64, 128], U.tpad: [20], U.bpad: [100],
            #     }) * LY({
            #         U.addpnt: [10, 1, 0.1, 0.01],
            #     })
            if v in {T3}:
                tmp_ly *= LY({
                    U.ed: [64], U.tpad: [20], U.bpad: [200],
                }) * LY({
                    U.addu: [1],
                    U.addb: [1],
                    U.addpnt: [1],
                    # }, {
                    #     U.addu: [1],
                    #     U.addb: [1],
                    #     U.addpnt: [1],
                    # }, {
                    #     U.addu: [1],
                    #     U.addb: [1],
                    #     U.addpnt: [1],
                })
            if v in {Doc2vec}:
                tmp_ly *= LY({
                    U.ed: [64, 128], U.tpad: [0], U.bpad: [0],
                    U.ws: [5, 7, 3], U.addu: [0, 1],
                })
            model_ly += tmp_ly
        ret_ly = data_ly * model_ly
        return ret_ly.eval()


if __name__ == '__main__':
    Grid().main()
