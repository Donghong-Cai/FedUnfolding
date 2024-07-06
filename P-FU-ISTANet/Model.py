
import tensorflow as tf


class ISTA_net(object):
    def __init__(self, encoded_dim, Phi_input, PhiInv_input, PhaseNumber, X_input, X_output, learning_rate):
        # 训练数据维度
        self.img_height = 32
        self.img_width = 32
        self.img_channels = 2
        self.img_total = self.img_height * self.img_width * self.img_channels
        # 网络维度
        self.cmp = encoded_dim
        self.n_input = self.img_total
        self.n_output = self.img_total


        self.Phi = Phi_input

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
        self.opt_q = tf.train.AdamOptimizer(learning_rate=0.0004)
        self.grad_placeholder = [(tf.placeholder(tf.float32, shape=w.get_shape()), w) for w in tf.trainable_variables()]
        self.optim_q = self.opt_q.apply_gradients(self.grad_placeholder)
        

    def add_con2d_weight_bias(self, w_shape, b_shape, order_no):
        Weights = tf.get_variable(shape=w_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='Weights_%d'%order_no)

        return Weights


    def ista_block(self, input_layers, input_data, layer_no):
        
        lambda_step = tf.Variable(0.1, dtype=tf.float32)
        soft_thr = tf.Variable(0.1, dtype=tf.float32)

        conv_size = 32
        filter_size = 3
        
        x1_ista = input_layers[-1] - tf.scalar_mul(lambda_step, tf.matmul(tf.add(tf.matmul(input_layers[-1], self.Phi),-self.s), tf.transpose(self.Phi)))
        
        x2_ista = tf.reshape(x1_ista, shape=[-1, self.img_height, self.img_width, self.img_channels])

        Weights0 = self.add_con2d_weight_bias([filter_size, filter_size, 2, conv_size], [conv_size], 0)

        Weights1 = self.add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 1)
        Weights11 = self.add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 11)

        Weights2 = self.add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 2)
        Weights22 = self.add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 22)

        Weights3 = self.add_con2d_weight_bias([filter_size, filter_size, conv_size, 2], [1], 3)

        
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

    
    def get_grad(self, var):
        grad = self.opt_q.compute_gradients(self.cost_all, var)
        return grad

