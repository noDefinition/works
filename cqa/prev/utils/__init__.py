from prev.utils.cls import *
from prev.utils.funs import *
from prev.utils.wraps import *

gun = UniqueName()
timer = Timer()

pname = 'QAR'
data_home = '/home/wwang/Data/{}'.format(pname)

phome = '/home/wwang/Projects/{}'.format(pname)
log_dir = '{}/log'.format(phome)
tensorboard_dir = '{}/tensorboard'.format(phome)
fig_dir = '{}/fig'.format(phome)

logger = Logger()
