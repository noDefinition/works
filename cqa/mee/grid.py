import re
from utils import Nodes, tu, iu, tmu
from cqa.data.datasets import DataSo, DataZh
from cqa.mee.models import *
from cqa.mee import K


class GridMee:
    def __init__(self):
        self.datas, self.vers, self.gpu_ids, self.gpu_max = Nodes.select(
            n1702=([DataSo, DataZh], [P3], [0, 1], 1),
            # ngpu=([DataSo, DataZh], [P4], [0, 1, 3], [2, 2, 2]),
            ngpu=([DataSo, DataZh], [P4], [0, 1, 2], 1),
        )
        # self.use_fda = 0 if len(self.gpu_ids) * self.gpu_max <= 1 else 1
        self.dnames = [d.name for d in self.datas]
        self.use_fda = 1
        # self.comment = 'tune_topk'
        while True:
            self.comment = input('comment:')
            if re.findall('\W', self.comment):
                print('invalid comment {}, reinput'.format(self.comment))
                continue
            break

    def grid_od_list(self):
        LY = tu.LY
        common_ly = LY((
            (K.ep, [20]), (K.es, [6]), (K.lr, [1e-4]),
            (K.dn, self.dnames), (K.fda, [self.use_fda]),
            (K.lid, list(range(3))),
        ))
        vers_ly = LY()
        for v in self.vers:
            tmp_ly = LY(((K.vs, [v.__name__]),))
            # if v == V1 or v == V5 or v == P1 or v == P2:
            #     tmp_ly *= LY((
            #         (K.reg, [0, 1, 1e-2]),
            #     ))
            # if v == U1:
            #     tmp_ly *= LY((
            #         # (K.reg, [0, 1]),
            #         # )) * LY((
            #         (K.woru, [0]),
            #     ), (
            #         (K.woru, [1, 2]),
            #         (K.eps, [100, 1000, 10000]),
            #     ))
            # if v == W1:
            #     tmp_ly *= LY((
            #         # (K.reg, [1e-2]),
            #         (K.reg, [0, 1, 1e-2]),
            #         (K.eps, [0, 1e-1, 1e-2, 1e-3]),
            #     ))
            # if v == S1:
            #     tmp_ly *= LY((
            #         (K.mix, [0, 0.5, 1, 2]),
            #     ))
            # if v == S2:
            #     tmp_ly *= LY((
            #         (K.mix, [0, 0.5, 2]),
            #         (K.dpt, [1, 3]),
            #         (K.act, [0]),
            #         # (K.act, [0, 1, 2]),
            #     ))
            if v == P3:
                tmp_ly *= LY((
                    (K.woru, [0]), (K.topk, [int(1e9)]),
                    # )) + LY((
                    # (K.woru, [1, 2]),
                    # (K.topk, [5, 10, 15, 20, 30, 40, 50]),
                ))
            if v == P4:
                tmp_ly *= LY((
                    #     (K.ttp, [0]), (K.atp, [0]), (K.topk, [5]),
                    # )) + LY((
                    #     (K.ttp, [0]), (K.atp, [1]),
                    # )) + LY((
                    (K.ttp, [1]), (K.atp, [0, 1]), (K.drp, [0.3, 0]),
                ))
            vers_ly += tmp_ly
        return (common_ly * vers_ly).eval()

    def make_log_path(self, make_new):
        vnames = [v.__name__ for v in self.vers]
        tstr, dstr, vstr = tmu.format_date()[2:], '+'.join(self.dnames), '+'.join(vnames)
        log_path = './log' + '_'.join([tstr, vstr, dstr, self.comment])
        if make_new:
            import inspect
            import cqa.mee.main as m
            iu.mkdir(log_path, rm_prev=True)
            iu.copy(__file__, log_path)
            iu.copy(inspect.getfile(m), log_path)
            for v in self.vers:
                iu.copy(inspect.getfile(v), log_path)
        return log_path

    def main(self):
        log_path = self.make_log_path(make_new=True)
        od_list = self.grid_od_list()
        od_list = tu.update_od_list(od_list, log_path, shuffle=True)
        tu.run_od_list('python3.6 ./main.py ', od_list, self.gpu_ids, self.gpu_max)
        new_name = log_path + '_OVER'
        iu.rename(log_path, new_name)
        iu.move(new_name, './logs')

    @staticmethod
    def test_od():
        log_path = './jupyter_wtffff'
        iu.mkdir(log_path, rm_prev=True)
        return dict([
            (K.gid, 0), (K.gi, 0), (K.gp, 0.2), (K.ep, 20), (K.es, 5), (K.lr, 5e-3), (K.fda, True),
            (K.vs, P1.__name__), (K.reg, 0), (K.dn, DataZh.name), (K.lg, log_path),
        ])


if __name__ == '__main__':
    GridMee().main()
