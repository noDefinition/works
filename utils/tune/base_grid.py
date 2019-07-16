import inspect
from typing import List, Dict
from utils import iu, tmu, tu


class BaseGrid(object):
    def __init__(self):
        self.vers: List = list()
        self.datas: List = list()
        self.comment: str = ''
        self.copy_modules: List = list()
        self.gpu_ids = self.gpu_max = None

    def grid_od_list(self) -> List[Dict]:
        return list()

    @staticmethod
    def is_debugging() -> bool:
        from argparse import ArgumentParser
        parser = ArgumentParser()
        parser.add_argument('-d', '--debug', action='store_true', default=False)
        return parser.parse_args().d

    @staticmethod
    def get_comment(default: str = None) -> str:
        import re
        if default:
            return default
        while True:
            comment = input('comment:')
            if re.findall(r'\W', comment):
                print('invalid comment {}, reinput'.format(comment))
                continue
            return comment

    def make_log_path(self) -> str:
        tstr = tmu.format_date()[2:]
        dstr = ''.join([d.name[0] for d in self.datas])
        vstr = '+'.join([v.__name__ for v in self.vers])
        log_path = './log' + '~'.join([tstr, vstr, dstr, self.comment])
        for i in range(5):
            ret_path = log_path + ('' if i == 0 else f'({i})')
            if not iu.exists(ret_path):
                return ret_path
        raise ValueError('You\'ve FUCKING assigned too many paths to one name, go check it out.')
        # if make_new:
        #     iu.mkdir(log_path, rm_prev=True)

    def copy_files_to(self, log_path):
        iu.copy(__file__, log_path)
        for m in self.copy_modules:
            iu.copy(inspect.getfile(m), log_path)

    def main(self):
        log_path = self.make_log_path()
        self.copy_files_to(log_path)
        od_list = self.grid_od_list()
        od_list = tu.update_od_list(od_list, log_path, shuffle=False)
        tu.run_od_list('python3.6 ./main.py ', od_list, self.gpu_ids, self.gpu_max)
        new_path = log_path + '=OVER'
        iu.rename(log_path, new_path)
        out_path = './logs'
        iu.mkdir(out_path, rm_prev=False)
        iu.move(new_path, out_path)
