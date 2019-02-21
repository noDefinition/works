from utils import io_utils as iu

'/home/nfs/cdong/tw/src/clustering/data/'
dirs = iu.list_children('/home/nfs/cdong/tw/src', ctype=iu.DIR, full_path=True)
dirs = [d for d in dirs if 'venv' not in d and 'tools' not in d]

files = list()
while len(dirs) > 0:
    for i in range(len(dirs)):
        d = dirs.pop()
        # print(d)
        subdirs = iu.list_children(d, ctype=iu.DIR, full_path=True)
        subdirs = [d for d in subdirs if '__pycache__' not in d]
        dirs.extend(subdirs)
        subfiles = iu.list_children(d, ctype=iu.FILE, full_path=True, pattern='\.py$')
        subfiles = [f for f in subfiles if '__init__' not in f]
        files.extend(subfiles)

count = 0
for f in files:
    print(f)
    lines = iu.read_lines(f)
    for line in lines:
        line = line.strip()
        if not (line.startswith('#') or line.startswith('"') or line.startswith("'") or len(line) <= 1):
        # if not (len(line) <= 2):
            print(line)
            count += 1
    # print(count)
    # exit()
print(count)
