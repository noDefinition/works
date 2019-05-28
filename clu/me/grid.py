import clu.me.main as m
from clu.me import C
from clu.me.gen1 import *
from clu.data.datasets import *
from utils import Nodes, tu
from utils.tune.base_grid import BaseGrid


class GridClu(BaseGrid):
    def __init__(self):
        super(GridClu, self).__init__()
        self.datas, self.gpu_ids, self.gpu_max, self.vers = Nodes.select(
            # n1702=([DataEvent, DataTREC, DataGoogle], [0, 1], 3, [N6, N7]),
            ngpu=([DataEvent, DataTREC, DataGoogle], [0], 1, [N6]),
        )
        self.comment = self.get_comment(default='try')
        self.modules_to_copy = [m] + self.vers

    def grid_od_list(self):
        # def diff_c_num(clu_num):
        #     return [int(i * clu_num / 10) for i in range(2, 10, 2)] + \
        #            [int(i * clu_num / 2) for i in range(2, 11)]
        # return (np.array([int(i * clu_num / 2) for i in range(1, 11)])).astype(int).tolist()
        LY = tu.LY
        com_ly = LY((
            (C.lid, list(range(5))), (C.ep, [10]), (C.bs, [64]), (C.ns, [16]),
        ))
        dat_ly = LY(*[((C.dn, [d.name]), (C.cn, [d.topic_num]),) for d in self.datas])
        ver_ly = LY()
        for v in self.vers:
            tmp_ly = LY((
                (C.vs, [v.__name__]),
            ))
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
            if v == N6:
                part1 = LY((  # different cluster num
                    (C.l3, [0]),
                ), (
                    (C.l3, [1]), (C.worc, [1]), (C.eps, [50]),
                )) * LY((
                    (C.l1, [1]), (C.l2, [1]),
                ))
                tmp_ly *= part1
            ver_ly += tmp_ly
        return (com_ly * dat_ly * ver_ly).eval()


if __name__ == '__main__':
    GridClu().main()
