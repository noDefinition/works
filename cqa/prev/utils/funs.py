import numpy as np, sys, math, os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt








# plot data
def plot(x, y, fn, fig = None, **kw):
    if fig is None:
        fig = plt.figure()
    plt.plot(x, y, **kw)
    plt.title(fn)
    plt.savefig('{}/{}.jpg'.format(config['Dirs']['fighome'], fn), bbox_inches='tight')









def set_gpu_using(memory_rate = 1.0, gpus = '0'):  
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
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






def get_session(memory_rate = 1.0, gpus = '0'):  
    import tensorflow as tf
    #os.environ['CUDA_VISIBLE_DEVICE'] = '0,1,2,3'
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction = memory_rate,
            visible_device_list = gpus,
            allow_growth = True,
            )
    config = tf.ConfigProto(gpu_options = gpu_options)
    session = tf.Session(config = config)
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
