from typing import List
import tensorflow as tf
import tensorflow.summary as su
from tensorflow.summary import histogram, scalar
from tensorflow.keras.layers import Dense

# from tensorflow.layers import Dense

i16 = tf.int16
i32 = tf.int32
i64 = tf.int64
f16 = tf.float16
f32 = tf.float32
f64 = tf.float64

relu = tf.nn.relu
tanh = tf.nn.tanh
sigmoid = tf.nn.sigmoid
softmax = tf.nn.softmax


# histogram = su.histogram
# scalar = su.scalar


def instant_denses(inputs, uas, name: str, use_bias: bool = True,
                   w_init=None, w_reg=None, output_seq: bool = False):
    output = inputs
    layers = list()
    for i, (u, a) in enumerate(uas):
        output = tf.layers.dense(
            output, units=u, activation=a, name='{}_{}'.format(name, i),
            use_bias=use_bias, kernel_initializer=w_init, kernel_regularizer=w_reg,
        )
        layers.append(output)
    return layers if output_seq else output


def build_denses(uas, name: str) -> List[Dense]:
    return [Dense(units=u, activation=a, name=f'{name}_{i}') for i, (u, a) in enumerate(uas)]


def postpone_denses(inputs, denses: List[Dense], name: str):
    for i, d in enumerate(denses):
        with tf.name_scope(f'{name}_{i}'):
            inputs = d.apply(inputs)
    return inputs


def inner_dot(a, b, axis=-1, keepdims=False, name=None):
    with tf.name_scope(name):
        return tf.reduce_sum(a * b, axis=axis, keepdims=keepdims)


def l2_norm_tensors(*tensors):
    return apply_tensors(lambda t: tf.nn.l2_normalize(t, axis=-1), *tensors)


def apply_tensors(func, *tensors):
    res = [func(t) for t in tensors]
    return res if len(res) > 1 else res[0]


def get_train_op(lr, loss, var_list=None, opt_cls='Adam', name=None,
                 grads=False, grad_vars_func=None):
    from tensorflow.train import AdagradOptimizer, AdamOptimizer
    learning_rate = tf.train.exponential_decay(**lr) if isinstance(lr, dict) else lr
    optimizer_cls = {'adagrad': AdagradOptimizer, 'adam': AdamOptimizer}[opt_cls.lower()]
    optimizer = optimizer_cls(learning_rate)
    grad_vars = optimizer.compute_gradients(loss, var_list=var_list)
    if grad_vars_func is not None:
        grad_vars = grad_vars_func(grad_vars)
    train_op = optimizer.apply_gradients(grad_vars, name=name)
    return (train_op, grad_vars) if grads else train_op


def get_session(gpu_id, gpu_frac, allow_growth=False, run_init=True) -> tf.Session:
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=gpu_frac,
        allow_growth=allow_growth, visible_device_list=str(gpu_id)
    )
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    if run_init:
        init_session(sess)
    return sess


def init_session(sess: tf.Session):
    sess.run(tf.global_variables_initializer())


def control_dependencies(cond, train_func):
    if cond:
        update_ops = tf.get_collection(key=tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_func()
    else:
        train_func()
