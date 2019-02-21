class LR(BasicPair):
    Data = dataset.sk_data
    tc = {}
    tc['fc'] = [[]]

    def make_inputs(self):
        if self.args.rank:
            self.inp_pos = tf.placeholder(tf.float32, (None, None), name = 'pos')
            self.inp_neg = tf.placeholder(tf.float32, (None, None), name = 'neg')
            self.train_inputs = [self.inp_pos, self.inp_neg]
            self.pred_inputs = [self.inp_pos]
        else:
            self.inp = tf.placeholder(tf.float32, (None, None), name = 'input')
            self.out_y = tf.placeholder(tf.float32, (None, ), name = 'output_y')
            self.train_inputs = [self.inp, self.out_y]
            self.pred_inputs = [self.inp]

    def _make_model(self):
        if self.args.rank:
            with tf.variable_scope("train", reuse=tf.AUTO_REUSE):
                p_score = self.TopLayer(self.rep(self.inp_pos))
                n_score = self.TopLayer(self.rep(self.inp_neg))
            diff_score = p_score - n_score
            self.loss = tf.reduce_mean(tf.maximum(self.args.mc - diff_score, tf.zeros_like(diff_score)))
            self.outputs = p_score
        else:
            with tf.variable_scope("train", reuse=tf.AUTO_REUSE):
                score = self.TopLayer(self.rep(self.inp))
            self.loss = tf.losses.mean_squared_error(self.out_y, score)
            self.outputs = score

    def pre_fit(self):
        dataset.rank_metric = self.args.rank
        dataset.add_features = self.args.add_features
        dataset.word2vec_size = self.args.dim_k
        #print('change size:', self.args.dim_k); input()
        self.data = self.Data(self.dataset)

        if self.args.rank:
            data_generator = self.data.gen_train_pair(batch_size = self.args.batch_size)
        else:
            data_generator = self.data.gen_data(batch_size = self.args.batch_size)
        queue_size = 10
        train_data_queue = Queue(queue_size)
        self.preload_process = Process(target = iter_and_put, args = (data_generator, train_data_queue))
        self.preload_process.daemon = True
        self.preload_process.start()
        def load_from_queue():
            while True:
                yield train_data_queue.get()
        self.data_generator = load_from_queue()


    def after_fit(self):
        ret, t = self.data.evaluate('test', self.predict)
        tf.reset_default_graph()
        self.sess.close()
        self.preload_process.terminate()
        return ret

    def fit(self, args, args_i, args_n):
        self.args_i = args_i
        self.args_n = args_n
        self.args = utils.Object(**args)

        self.pre_fit()
        self.prt_info()
        self.make_model()
        unique_fn = '{}-{}'.format(logger.unique_fn, self.args_i)

        tensorboard_dir = 'tensorboard/{}'.format(unique_fn)
        self.tb_dirs.append(tensorboard_dir)
        train_writer = tf.summary.FileWriter(tensorboard_dir, self.sess.graph)

        saver = tf.train.Saver()

        summ_loss = tf.Summary()
        summ_loss_v = summ_loss.value.add()
        summ_loss_v.tag = 'loss_per_batch'

        summaries = tf.summary.merge_all()

        batch_cnt = 0
        best_vali = None
        brk = 0
        #ret, _ = self.data.evaluate('vali', self.predict)
        #print(ret)
        has_ckpt = False
        self.start_time = time.time()
        try:
            for epochs in range(self.max_epochs):
                loss = []
                progress_bar = utils.ProgressBar(self.args.batch_steps, msg = 'training')
                for step in range(self.args.batch_steps):
                    batch = next(self.data_generator)
                    data = dict(zip(self.train_inputs, batch))
                    #print(len(self.train_inputs), len(batch)); input()
                    #print(data); input()
                    if step == 0 and summaries is not None:
                        summ = self.sess.run(summaries, data)
                        train_writer.add_summary(summ, global_step = batch_cnt)

                    if self.minimize_emb_w is None:
                        _, _loss = self.sess.run([self.minimize, self.loss], data)
                    else:
                        _, _, _loss = self.sess.run([self.minimize, self.minimize_emb_w, self.loss], data)

                    batch_cnt += 1
                    loss.append(_loss)
                    summ_loss_v.simple_value = _loss
                    train_writer.add_summary(summ_loss, global_step = batch_cnt)
                    progress_bar.make_a_step()
                train_time = progress_bar.stop()

                vali, vali_time = self.data.evaluate('vali', self.predict)
                if vali.is_better_than(best_vali):
                    brk = 0
                    best_vali = vali
                    saver.save(self.sess, 'tensorboard/{}/model.ckpt'.format(unique_fn))
                    has_ckpt = True
                else:
                    brk += 1

                if self.run_test:
                    test, test_time = self.data.evaluate('test', self.predict)
                    vali = '{} {}'.format(vali, test)
                    vali_time += test_time

                msg = '#{}/{}, loss: {:.5f}, vali: {}, brk: {}, time: {:.1f}s {:.1f}s'.format(
                        epochs + 1, self.max_epochs, np.mean(loss), vali, brk, train_time, vali_time)
                log(msg, i = -1, red = (brk == 0))
                if self.early_stop > 0 and brk >= self.early_stop:
                    break

        except KeyboardInterrupt:
            utils.timer.stop()
            log('KeyboardInterrupt')
        except Exception as e:
            utils.timer.stop()
            log('Exception: {}'.format(e), red = True)
        if has_ckpt:
            saver.restore(self.sess, 'tensorboard/{}/model.ckpt'.format(unique_fn))
        return self.after_fit()
