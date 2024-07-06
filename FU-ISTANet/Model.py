'''
Test Platform: Tensorflow version: 1.15.0
FL basic model

2022/12/31
'''


import tensorflow as tf


class ISTA_net(object):
    def __init__(self, encoded_dim, Phi_input, PhiInv_input, PhiTPhi_input, PhaseNumber, X_input, X_output, learning_rate):
        # 训练数据维度
        self.img_height = 32
        self.img_width = 32
        self.img_channels = 2
        self.img_total = self.img_height * self.img_width * self.img_channels
        # 网络维度
        self.cmp = encoded_dim
        self.n_input = self.img_total
        self.n_output = self.img_total

        # ISTA参数
        self.Phi = Phi_input
        self.PhiTPhi = PhiTPhi_input
        self.PhiInv = PhiInv_input

        self.PhaseNumber = PhaseNumber

        self.X_input = X_input
        self.X_output = X_output
        self.s = tf.matmul(X_input, Phi_input)

        self.X0 = tf.matmul(self.s, self.PhiInv)

        self.PhiTb = tf.matmul(self.s, tf.transpose(Phi_input))

        [self.Prediction, self.Pre_symetric] = self.inference_ista(self.X0, self.PhaseNumber, self.X_output, reuse=False)
        self.cost0 = tf.reduce_mean(tf.square(self.X0 - self.X_output))

        [self.cost, self.cost_sym] = self.compute_cost(self.X_output, self.PhaseNumber)

        self.cost_all = self.cost + 0.01 * self.cost_sym

        self.optm_all = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost_all)

    def add_con2d_weight_bias(self, w_shape, b_shape, order_no):
        Weights = tf.get_variable(shape=w_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='Weights_%d'%order_no)
        biases = tf.Variable(tf.random_normal(b_shape, stddev=0.05), name='biases_%d' % order_no)
        return [Weights, biases]

    def ista_block(self, input_layers, input_data, layer_no):
        tau_value = tf.Variable(0.1, dtype=tf.float32)
        lambda_step = tf.Variable(0.1, dtype=tf.float32)
        soft_thr = tf.Variable(0.1, dtype=tf.float32)
        conv_size = 32
        filter_size = 3
        
        x1_ista = input_layers[-1] - tf.scalar_mul(lambda_step, tf.matmul(tf.add(tf.matmul(input_layers[-1], self.Phi),-self.s), tf.transpose(self.Phi)))
        #x1_ista = tf.add(input_layers[-1] - tf.scalar_mul(lambda_step, tf.matmul(input_layers[-1], self.PhiTPhi)), tf.scalar_mul(lambda_step, self.PhiTb))  # X_k - lambda*A^TAX
        
        x2_ista = tf.reshape(x1_ista, shape=[-1, self.img_height, self.img_width, self.img_channels])

        [Weights0, bias0] = self.add_con2d_weight_bias([filter_size, filter_size, 2, conv_size], [conv_size], 0)

        [Weights1, bias1] = self.add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 1)
        [Weights11, bias11] = self.add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 11)

        [Weights2, bias2] = self.add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 2)
        [Weights22, bias22] = self.add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 22)

        [Weights3, bias3] = self.add_con2d_weight_bias([filter_size, filter_size, conv_size, 2], [1], 3)

        x3_ista = tf.nn.conv2d(x2_ista, Weights0, strides=[1, 1, 1, 1], padding='SAME')

        x4_ista = tf.nn.relu(tf.nn.conv2d(x3_ista, Weights1, strides=[1, 1, 1, 1], padding='SAME'))
        x44_ista = tf.nn.conv2d(x4_ista, Weights11, strides=[1, 1, 1, 1], padding='SAME')

        x5_ista = tf.multiply(tf.sign(x44_ista), tf.nn.relu(tf.abs(x44_ista) - soft_thr))

        x6_ista = tf.nn.relu(tf.nn.conv2d(x5_ista, Weights2, strides=[1, 1, 1, 1], padding='SAME'))
        x66_ista = tf.nn.conv2d(x6_ista, Weights22, strides=[1, 1, 1, 1], padding='SAME')

        x7_ista = tf.nn.conv2d(x66_ista, Weights3, strides=[1, 1, 1, 1], padding='SAME')

        x7_ista = x7_ista + x2_ista

        x8_ista = tf.reshape(x7_ista, shape=[-1, self.img_total])

        x3_ista_sym = tf.nn.relu(tf.nn.conv2d(x3_ista, Weights1, strides=[1, 1, 1, 1], padding='SAME'))
        x4_ista_sym = tf.nn.conv2d(x3_ista_sym, Weights11, strides=[1, 1, 1, 1], padding='SAME')
        x6_ista_sym = tf.nn.relu(tf.nn.conv2d(x4_ista_sym, Weights2, strides=[1, 1, 1, 1], padding='SAME'))
        x7_ista_sym = tf.nn.conv2d(x6_ista_sym, Weights22, strides=[1, 1, 1, 1], padding='SAME')

        x11_ista = x7_ista_sym - x3_ista

        return [x8_ista, x11_ista]


    def inference_ista(self, input_tensor, n, X_output, reuse):
        layers = []
        layers_symetric = []
        layers.append(input_tensor)
        for i in range(n):
            with tf.variable_scope('conv_%d' %i, reuse=reuse):
                [conv1, conv1_sym] = self.ista_block(layers, X_output, i)
                layers.append(conv1)
                layers_symetric.append(conv1_sym)
        return [layers, layers_symetric]

    def compute_cost(self, X_output, PhaseNumber):
        cost = tf.reduce_mean(tf.reduce_sum(tf.square(self.Prediction[-1] - X_output), axis=1))
        cost_sym = 0
        for k in range(PhaseNumber):
            cost_sym += tf.reduce_mean(tf.reduce_sum(tf.square(self.Pre_symetric[k]), axis=1))

        return [cost, cost_sym]

'''
if __name__=='__main__':

    envir = 'indoor'  # 'indoor' or 'outdoor'
    # image params
    img_height = 32
    img_width = 32
    img_channels = 2
    img_total = img_height * img_width * img_channels
    # network params
    residual_num = 2
    encoded_dim = 512  # compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32

    n_input = encoded_dim
    n_output = img_total
    batch_size = 200
    PhaseNumber = 5
    nrtrain = 100000
    learning_rate = 0.0001
    EpochNum = 300
    
    print('Load Data...')
    # Data loading
    if envir == 'indoor':
        mat = sio.loadmat('../data/DATA_Htrainin.mat')
        x_train = mat['HT']  # array
        mat = sio.loadmat('../data/DATA_Hvalin.mat')
        x_val = mat['HT']  # array
        mat = sio.loadmat('../data/DATA_Htestin.mat')
        x_test = mat['HT']  # array

    elif envir == 'outdoor':
        mat = sio.loadmat('../data/DATA_Htrainout.mat')
        x_train = mat['HT']  # array
        mat = sio.loadmat('../data/DATA_Hvalout.mat')
        x_val = mat['HT']  # array
        mat = sio.loadmat('../data/DATA_Htestout.mat')
        x_test = mat['HT']  # array

    Phi_data = np.random.normal(0, 1, (encoded_dim, img_total)) * 0.01
    Phi_input = Phi_data.transpose()




    init = tf.global_variables_initializer()
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    
    sess = tf.Session(config=config)
    sess.run(init)
    
    print("...............................")
    print("Phase Number is %d, CS ratio is %d%%" % (PhaseNumber, encoded_dim))
    print("...............................\n")
    
    
    model_dir = 'Phase_%d_encod_0_%d_ISTA_Net_plus_Model' % (PhaseNumber, encoded_dim)
    
    output_file_name = "Log_output_%s.txt" % (model_dir)
    
    val_input = np.dot(x_val, Phi_input)
    feed_dict_val = {X_input: val_input[0:500,:], X_output: x_val[0:500,:]}
    
    for epoch_i in range(0, EpochNum+1):
        randidx_all = np.random.permutation(nrtrain)
        for batch_i in range(nrtrain // batch_size):
            randidx = randidx_all[batch_i*batch_size:(batch_i+1)*batch_size]
    
            batch_ys = x_train[randidx, :]
            batch_xs = np.dot(batch_ys, Phi_input)
    
            feed_dict = {X_input: batch_xs, X_output: batch_ys}
            sess.run(optm_all, feed_dict=feed_dict)
            print('\r', end='')
            print("epoch Progress: {}/{}:{}".format(batch_i, nrtrain // batch_size, sess.run(cost, feed_dict=feed_dict)), end="", flush=True)
    
        output_data = "[{}/{}] cost: {},  val:{},  \n".format (epoch_i, EpochNum, sess.run(cost, feed_dict=feed_dict), sess.run(cost, feed_dict=feed_dict_val))
        print(output_data)
        print('--------------------------')
    
        output_file = open(output_file_name, 'a')
        output_file.write(output_data)
        output_file.close()
    
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if epoch_i <= 30:
            saver.save(sess, './%s/CS_Saved_Model_%d.cpkt' % (model_dir, epoch_i), write_meta_graph=False)
        else:
            if epoch_i % 20 == 0:
                saver.save(sess, './%s/CS_Saved_Model_%d.cpkt' % (model_dir, epoch_i), write_meta_graph=False)
    
    
    
    print("Training Finished")
    
    test_input = np.dot(x_test, Phi_input)
    feed_dict_test = {X_input: test_input[0:500,:], X_output: x_test[0:500,:]}
    output_test = sess.run(Prediction[-1], feed_dict=feed_dict_test)
    
    # Calcaulating the NMSE and rho
    if envir == 'indoor':
        mat = sio.loadmat('../data/DATA_HtestFin_all.mat')
        X_test = mat['HF_all']  # array
    
    elif envir == 'outdoor':
        mat = sio.loadmat('../data/DATA_HtestFout_all.mat')
        X_test = mat['HF_all']  # array
    
    def trans_F(x_train):
        x_train = x_train.astype('float32')
        x_train = np.reshape(x_train, (
            len(x_train), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format
        return x_train
    
    
    def nmse_rho(X_test, x_hat):
        X_test = np.reshape(X_test, (len(X_test), img_height, 125))
        x_test_real = np.reshape(x_test[:, 0, :, :], (len(x_test), -1))
        x_test_imag = np.reshape(x_test[:, 1, :, :], (len(x_test), -1))
        x_test_C = x_test_real - 0.5 + 1j * (x_test_imag - 0.5)
        x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
        x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
        x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
        x_hat_F = np.reshape(x_hat_C, (len(x_hat_C), img_height, img_width))
        X_hat = np.fft.fft(np.concatenate((x_hat_F, np.zeros((len(x_hat_C), img_height, 257 - img_width))), axis=2), axis=2)
        X_hat = X_hat[:, :, 0:125]
    
        n1 = np.sqrt(np.sum(np.conj(X_test) * X_test, axis=1))
        n1 = n1.astype('float64')
        n2 = np.sqrt(np.sum(np.conj(X_hat) * X_hat, axis=1))
        n2 = n2.astype('float64')
        aa = abs(np.sum(np.conj(X_test) * X_hat, axis=1))
        rho = np.mean(aa / (n1 * n2), axis=1)
        X_hat = np.reshape(X_hat, (len(X_hat), -1))
        X_test = np.reshape(X_test, (len(X_test), -1))
        power = np.sum(abs(x_test_C) ** 2, axis=1)
        power_d = np.sum(abs(X_hat) ** 2, axis=1)
        mse = np.sum(abs(x_test_C - x_hat_C) ** 2, axis=1)
    
        return [mse, rho, power]
    x_test = trans_F(x_test)
    output_test = trans_F(output_test)
    [mse, rho, power] = nmse_rho(X_test, output_test)
    
    print("In " + envir + " environment")
    print("When dimension is", encoded_dim)
    print("NMSE is ", 10 * math.log10(np.mean(mse / power)))
    print("Correlation is ", np.mean(rho))
    print("=========================================")
    sess.close()
'''