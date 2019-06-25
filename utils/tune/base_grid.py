from utils import iu, tmu, tu
from typing import List, Dict


class BaseGrid(object):
    def __init__(self):
        self.vers: List = None
        self.datas: List = None
        self.comment: str = None
        self.gpu_ids = self.gpu_max = None
        self.modules_to_copy: List = None

    def grid_od_list(self) -> List[Dict]:
        return None

    @staticmethod
    def get_comment(default: str = None):
        import re
        if default:
            return default
        while True:
            comment = input('comment:')
            if re.findall(r'\W', comment):
                print('invalid comment {}, reinput'.format(comment))
                continue
            return comment

    def make_log_path(self, make_new) -> str:
        tstr = tmu.format_date()[2:]
        dstr = '+'.join([d.name for d in self.datas])
        vstr = '+'.join([v.__name__ for v in self.vers])
        log_path = './log' + '_'.join([tstr, vstr, dstr, self.comment])
        if make_new:
            iu.mkdir(log_path, rm_prev=True)
        return log_path

    def copy_files_to(self, log_path):
        import inspect
        iu.copy(__file__, log_path)
        for m in self.modules_to_copy:
            iu.copy(inspect.getfile(m), log_path)

    def main(self):
        log_path = self.make_log_path(make_new=True)
        self.copy_files_to(log_path)
        od_list = self.grid_od_list()
        # od_list = tu.update_od_list(od_list, log_path, shuffle=True)
        tu.update_od_list(od_list, log_path, shuffle=True)
        tu.run_od_list('python3.6 ./main.py ', od_list, self.gpu_ids, self.gpu_max)
        new_path = log_path + '_OVER'
        iu.rename(log_path, new_path)
        out_path = './logs'
        iu.mkdir(out_path, rm_prev=False)
        iu.move(new_path, out_path)
