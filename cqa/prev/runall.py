#########################################################################
# File Name: runall.py
# Author: wwang
# mail: 750636248@qq.com
# Created Time: 2018年08月10日 星期五 21时42分00秒
#########################################################################

import numpy as np, sys, math, os
sys.path.append('/home/wwang')
from pywang import myemail

def run1():
    #dss = ['zhihu']
    #dss = ['so']
    dss = ['so', 'zhihu10']
    #dss = ['test']
    #models_prev = ['Ans', 'User', 'AnsUser', 'QueAnsUser', 'SimpleMatch', 'CMatch', 'AAAI17', 'AAAI15', 'IJCAI15', 'SimpleAtt', ]
    models = ['AnsUser', 'MAU', 'CMatch']
    #models_prev = ['AAAI17', 'AAAI15']
    #models_prev = []
    for ds in dss:
        for model in models:
            cmd = 'python3 run.py -model={} -ds={} -gpu=2 -am=grid'.format(model, ds)
            ret = os.system(cmd)
            try:
                os.system('sh /home/wwang/login.sh')
                myemail.sent('a run over', 'cmd: {}, ret: {}'.format(cmd, ret))
            except Exception as e:
                print('err: {}'.format(e))

def run2():
    dss = ['so', 'zhihu']
    #models_prev = ['BasicPair']
    models = ['AAAI17', 'AAAI15']
    for ds in dss:
        for model in models:
            cmd = 'python3 run.py -model={} -ds={} -am=fix -msg=lr5 -gpu=2'.format(model, ds)
            ret = os.system(cmd)
            try:
                os.system('sh /home/wwang/login.sh')
                myemail.sent('a run over', 'cmd: {}, ret: {}'.format(cmd, ret))
            except Exception as e:
                print('err: {}'.format(e))

def main():
    print('hello world, runall.py')
    #dss = ['test']
    run1()
    #run2()

if __name__ == '__main__':
    main()

