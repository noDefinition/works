#########################################################################
# File Name: util.py
# Author: wwang
# mail: 750636248@qq.com
# Created Time: 2017年09月28日 星期四 14时26分00秒
#########################################################################

import numpy as np, sys, math, os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

class Object:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.init()

    def init(self):
        pass

# plot data
def plot(x, y, fn, fig = None, **kw):
    if fig is None:
        fig = plt.figure()
    plt.plot(x, y, **kw)
    plt.title(fn)
    plt.savefig('{}/{}.jpg'.format(config['Dirs']['fighome'], fn), bbox_inches='tight')

def set_gpu_using(memory_rate = 1.0, gpus = '0'):  
    os.environ['CUDA_VISIBLE_DEVICE'] = '0,1,2,3'
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    from keras import backend as K
    import tensorflow as tf
    K.clear_session()
    num_threads = os.environ.get('OMP_NUM_THREADS')  
    gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction = memory_rate,
            visible_device_list = gpus,
            allow_growth = True,
            )
  
    if num_threads:
        session = tf.Session(
                config = tf.ConfigProto(
                    gpu_options = gpu_options,
                    intra_op_parallelism_threads = num_threads))
    else:
        session = tf.Session(
                config = tf.ConfigProto(
                    gpu_options = gpu_options))
    K.set_session(session)
    return session

def fix_clip_norm():
    import keras
    from keras import backend as K
    import tensorflow as tf
    import copy
    def clip_norm(g, c, n):
        if c <= 0:  # if clipnorm == 0 no need to add ops to the graph
            return g

        # tf require using a special op to multiply IndexedSliced by scalar
        if K.backend() == 'tensorflow':
            condition = n >= c
            then_expression = tf.scalar_mul(c / n, g)
            else_expression = g

            # saving the shape to avoid converting sparse tensor to dense
            if isinstance(then_expression, tf.Tensor):
                g_shape = copy.copy(then_expression.get_shape())
            elif isinstance(then_expression, tf.IndexedSlices):
                g_shape = copy.copy(then_expression.dense_shape)
            if condition.dtype != tf.bool:
                condition = tf.cast(condition, 'bool')
            g = tf.cond(condition,
                        lambda: then_expression,
                        lambda: else_expression)
            if isinstance(then_expression, tf.Tensor):
                g.set_shape(g_shape)
            elif isinstance(then_expression, tf.IndexedSlices):
                g._dense_shape = g_shape
        else:
            g = K.switch(K.greater_equal(n, c), g * c / n, g)
        return g

    keras.optimizers.clip_norm = clip_norm

class UniqueName:
    def __init__(self):
        self.nc = {}

    def __call__(self, name):
        self.nc.setdefault(name, 0)
        self.nc[name] += 1
        return '{}_{}'.format(name, self.nc[name])

def test():
    pass

gun = UniqueName()

def main():
    print('hello world, util.py')
    plot([1,2,3], [4,3,5], 'testttt')

if __name__ == '__main__':
    main()

