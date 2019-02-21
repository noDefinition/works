import os
import tensorflow as tf


def dump_matrices(obj_list, file_name):
    for o in obj_list:
        if isinstance(o, dict):
            name = o['name']
            mtx = o['value']
        else:
            name = None
            mtx = o
        tf.Variable(initial_value=mtx, name=name)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, file_name, global_step=0)


def set_visible_gpu_list(gpu_list):
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = '9'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(list(map(str, gpu_list)))
