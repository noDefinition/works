from cqa.baselinee.b1 import *


class Temp(B1):
    def define_inputs(self, len_q=None, len_a=None):
        super(Temp, self).define_inputs(len_q, len_a)
        self.train_inp = [self.inp_q0, self.inp_a0, self.inp_u0,
                          self.inp_q1, self.inp_a1, self.inp_u1]
        self.pred_inp = [self.inp_q0, self.inp_a0, self.inp_u0]

    def get_fd(self, q0, a0, u0, q1=None, a1=None, u1=None):
        # print([(k, v is not None) for k, v in locals().items()])
        if not (q1 is None or a1 is None or u1 is None):
            return dict(zip(self.train_inp, [q0, a0, u0, q1, a1, u1]))
        else:
            return dict(zip(self.pred_inp, [q0, a0, u0]))

    def train_step(self, q0, a0, u0, q1, a1, u1):
        fd = self.get_fd(q0, a0, u0, q1, a1, u1)
        fd[self.is_train] = True
        _, loss = self.sess.run([self.minimize, self.loss], feed_dict=fd)
        return loss

    def predict(self, q0, a0, u0):
        fd = self.get_fd(q0, a0, u0)
        return self.sess.run(self.pred, feed_dict=fd)


class AAAI15(Temp):
    def rep(self, ques, answ, user):
        with tf.variable_scope('S-matrix'):
            q_emb, q_mask = self.text_emb(ques)
            a_emb, a_mask = self.text_emb(answ)
            # print(q_emb.shape, a_emb.shape)
            q_emb = tf.tile(tf.expand_dims(q_emb, axis=0), [tf.shape(a_emb)[0], 1, 1])
            q_emb_l2 = tf.nn.l2_normalize(q_emb, axis=-1)  # (bs, nq, wd)
            a_emb_l2 = tf.nn.l2_normalize(a_emb, axis=-1)  # (bs, na, wd)
            a_emb_l2 = tf.transpose(a_emb_l2, perm=[0, 2, 1])  # (bs, wd, na)
            m = tf.matmul(q_emb_l2, a_emb_l2)  # (bs, nq, na)
            m = tf.expand_dims(m, -1)
        with tf.variable_scope('CNN'):
            k_reg = self.l2_reg
            m = tf.layers.conv2d(m, filters=20, kernel_size=5,
                                 kernel_initializer=self.x_init, kernel_regularizer=k_reg)
            m = tf.layers.max_pooling2d(m, (1, 3), (1, 3))
            m = tf.layers.conv2d(m, filters=50, kernel_size=5,
                                 kernel_initializer=self.x_init, kernel_regularizer=k_reg)
            m = tf.layers.max_pooling2d(m, (1, 3), (1, 3))
            o = tf.layers.flatten(m)
        return o


class AAAI17(Temp):
    def rep(self, ques, answ, user):
        ques = tf.expand_dims(ques, axis=0)
        q, q_mask = self.text_emb(ques)
        a, a_mask = self.text_emb(answ)
        q_rep = self.text_rep(q, q_mask)  # (1, dw)
        a_rep = self.text_rep(a, a_mask)  # (bs, dw)
        u_rep = self.user_emb(user)  # (bs, du)

        dim_q = int(q_rep.shape[-1])
        M = tf.get_variable('qM', shape=(dim_q, dim_q), initializer=self.x_init,
                            regularizer=self.l2_reg)
        qm = tf.matmul(q_rep, M)  # (1, dw)

        def dot(a, b, axis=-1):
            return tf.reduce_sum(a * b, axis, keepdims=False)

        qa = dot(qm, a_rep)  # (bs, )
        qu = dot(qm, u_rep)  # (bs, )
        return qa * qu

    def forward(self):
        with tf.variable_scope("train", reuse=tf.AUTO_REUSE):
            score0 = self.rep(self.inp_q0, self.inp_a0, self.inp_u0)
            score1 = self.rep(self.inp_q1, self.inp_a1, self.inp_u1)
        diff_score = score0 - score1
        ones, zeros = tf.ones_like(diff_score), tf.zeros_like(diff_score)
        self.loss = tf.reduce_mean(tf.maximum(ones - diff_score, zeros))
        self.pred = score0


class IJCAI15(Temp):
    def cnn_topk(self, t, ns):
        initer, reger = self.x_init, self.l2_reg
        with tf.variable_scope('CNN_tk'):
            def bias_tanh(_t, b_name):
                shape = [1] * (len(_t.shape) - 1) + [int(_t.shape[-1])]
                _b = tf.get_variable(b_name, initializer=initer, regularizer=reger, shape=shape)
                return tf.nn.tanh(_t + _b)

            dim_t = int(t.shape[-1])
            print('dim t', dim_t)
            k1 = tf.get_variable('conv1', initializer=initer, regularizer=reger,
                                 shape=(3, dim_t, 20))
            t1 = tf.nn.conv1d(t, k1, stride=1, padding='VALID', data_format='NWC')  # (bs,tn-3,20)
            t1 = tf.nn.top_k(t1, ns + (20 - ns) // 2).values  # (bs, tn-3, 5+(20-5)/2)
            t1 = bias_tanh(t1, 'conv_b1')  # (bs, tn-3, 5+(20-5)/2)
            print('t1.shape', t1.shape)

            dim_t = int(t1.shape[-1])
            k2 = tf.get_variable('conv2', initializer=initer, regularizer=reger,
                                 shape=(3, dim_t, 10))
            t2 = tf.nn.conv1d(t1, k2, stride=1, padding='VALID', data_format='NWC')  # (bs,tn-6,10)
            print('t2.shape', t2.shape)
            t2 = tf.nn.top_k(tf.layers.flatten(t2), ns).values  # (bs, ns)
            t2 = bias_tanh(t2, 'conv_b2')  # (bs, ns)
        return t2

    def rep(self, ques, answ, user):
        ns, r = 10, 5
        a_emb, a_mask = self.text_emb(answ)  # (bs, la, dw)
        a_rep = self.cnn_topk(a_emb, ns)  # (bs, ns)

        ques = tf.expand_dims(ques, axis=0)  # (1, lq)
        q_emb, q_mask = self.text_emb(ques)  # (1, lq, dw)
        q_rep = self.cnn_topk(q_emb, ns)  # (1, ns)
        q_rep = tf.tile(q_rep, (tf.shape(a_emb)[0], 1))  # (bs, ns)

        with tf.variable_scope('match'):
            initer, reger = self.x_init, self.l2_reg
            M = tf.get_variable(name='M', shape=(ns, ns, r), initializer=initer, regularizer=reger)
            V = tf.get_variable(name='V', shape=(2 * ns, r), initializer=initer, regularizer=reger)
            b = tf.get_variable(name='b', shape=(1, r), initializer=initer, regularizer=reger)

        qm = tf.tensordot(q_rep, M, axes=[[-1], [0]])  # (bs, ns, r)
        a_rep_ = tf.expand_dims(a_rep, 1)  # (bs, 1, ns)
        qma = tf.squeeze(tf.matmul(a_rep_, qm), axis=1)  # (bs, r)

        qa_rep = tf.concat([q_rep, a_rep], -1)  # (bs, ns * 2)
        qav = tf.matmul(qa_rep, V)  # (bs, r)
        return tf.nn.relu(qma + qav + b)
