import json
import os
import pickle
import re
import shutil
from pathlib import Path

dumps = json.dumps
loads = json.loads
dumpp = pickle.dumps
loadp = pickle.loads


def load_json(file):
    return json.load(open(file, mode='r'))


def dump_json(file, obj):
    json.dump(obj, open(file, mode='w'))


def read_lines(file, mode='r', newline='\n'):
    with open(file, mode=mode, newline=newline, errors='ignore') as fp:
        return [line.rstrip(newline) for line in fp.readlines()]


def write_lines(file, lines, mode='w', newline='\n'):
    with open(file, mode=mode, encoding='utf8') as fp:
        fp.writelines([line + newline for line in lines])


def load_array(file, **kwargs):
    lines = read_lines(file, mode='r')
    return [loads(line, **kwargs) for line in lines]


def dump_array(file, array, mode='w', **kwargs):
    lines = [dumps(obj, **kwargs) for obj in array]
    write_lines(file, lines=lines, mode=mode)


def load_pickle(file):
    return pickle.load(open(file, 'rb'))


def dump_pickle(file, obj, parents=True):
    if parents:
        mkprnts(file)
    pickle.dump(obj, open(file, mode='wb'), protocol=4)


def get_cwd(): return os.getcwd()


def rename(path, target): Path(path).rename(target)


def parent_name(path): return os.path.dirname(path)


def get_name(path): return Path(path).name


def is_dir(path): return Path(path).is_dir()


def is_file(path): return Path(path).is_file()


def exists(path): return Path(path).exists()


def join(*args): return os.path.join(*list(map(str, args)))


def remove(file):
    if exists(file):
        Path(file).unlink()


def rmtree(path):
    if os.path.exists(path):
        print('remove path {}'.format(path))
        shutil.rmtree(path)


def move(src, dst):
    shutil.move(src, dst)


def concat_files(input_files, output_file):
    os.popen('cat {} > {}'.format(' '.join(input_files), output_file)).close()


def copy(source, target):
    target_parent = Path(target).parent
    if not target_parent.exists():
        print('create', target_parent)
        target_parent.mkdir(parents=True)
    shutil.copy(source, target)


def mkdir(path, rm_prev=False):
    if rm_prev:
        rmtree(path)
    if not exists(path):
        os.makedirs(path)


def mkprnts(path):
    p = Path(path).parent
    if not p.exists():
        print('going to make dir', str(p))
        p.mkdir(parents=True)


ALL = 0
DIR = 1
FILE = 2


def list_children(path, ctype=FILE, pattern=None, full_path=False):
    children = list()
    for c in Path(path).iterdir():
        if not any((ctype == ALL, c.is_file() and ctype == FILE, c.is_dir() and ctype == DIR)):
            continue
        if pattern is not None and re.search(pattern, c.name) is None:
            continue
        children.append(c)
    children = sorted(str(c.absolute()) if full_path else c.name for c in children)
    return children


def choose_from(files, full_path=True):
    parr = sorted([Path(f) for f in files], key=lambda p: p.name)[::-1]
    for idx, c in enumerate(parr):
        print('*' if idx == 0 else ' ', '{} - {}'.format(idx, c.name))
    while True:
        try:
            choice = input('select idx (default 0): ').strip()
            index = 0 if choice == '' else int(choice)
            c = parr[index]
            break
        except KeyboardInterrupt:
            exit('ctrl + c')
        except:
            print('invalid index, please re-input')
    return str(c.absolute()) if full_path else c.name


def most_recent(files, full_path=True):
    assert len(files) >= 1
    c = sorted([Path(f) for f in files], key=lambda p: p.stat().st_mtime)[-1]
    return str(c.absolute()) if full_path else c.name


def list_in_days(*paths, pattern: str, in_days: float):
    import time

    def is_in_days(p_obj):
        diff_sec = int(time.time() - p_obj.stat().st_mtime)
        span_days = round(diff_sec / sec_one_day, 2)
        return span_days <= in_days

    def to_absolute(p_obj):
        return str(p_obj.absolute())

    subs = sum((list_children(p, DIR, pattern, True) for p in paths), list())
    parr = sorted((Path(s) for s in subs), key=lambda p: p.stat().st_mtime, reverse=True)
    sec_one_day = 60 * 60 * 24
    ret = [to_absolute(p) for p in filter(is_in_days, parr)]
    return ret

# if __name__ == '__main__':
#     for f in list_in_days('/home/cdong/works/uclu/me/logs', pattern='', in_days=16):
#         print(f)
