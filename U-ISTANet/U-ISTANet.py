'''
Code of U-ISTANet
'''


import tensorflow as tf
import scipy.io as sio
import numpy as np
import os
import math
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--envir", type=str, help="envir of data")
parser.add_argument("--dim", type=int, help="encoded_dim")
args = parser.parse_args()

envir = args.envir  # 'indoor' or 'outdoor'
# image params
img_height = 32
img_width = 32
img_channels = 2
img_total = img_height * img_width * img_channels
# network params
residual_num = 2
encoded_dim = args.dim # compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/64->dim.=64, compress rate=1/64->dim.=32


n_input = img_total #encoded_dim
n_output = img_total
batch_size = 400
PhaseNumber = 5
nrtrain = 100000
nrval = 30000

# 训练设置
learning_rate = tf.placeholder(tf.float32)
warmup_step = 50
EpochNum = 400
decay_steps=EpochNum - warmup_step
initial_learning_rate = 0.001
end_learning_rate = 0.0001
apply_rate = 0.0001
power = 2

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

x_test =  x_test[0:500,:]
x_train = x_train - 0.5
x_val = x_val - 0.5
x_test = x_test - 0.5

Phi_input = tf.get_variable(shape=[img_total, encoded_dim], initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                              name='Weights_enc')
PhiInv = tf.get_variable(shape=[encoded_dim,img_total], initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                              name='Weights_x0')


#Phi = tf.constant(Phi_input, dtype=tf.float32)
PhiTPhi = tf.get_variable(shape=[img_total,img_total], initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                              name='PhiTPhi')

X_input = tf.placeholder(tf.float32, [None, n_input])
X_output = tf.placeholder(tf.float32, [None, n_output])

s = tf.matmul(X_input, Phi_input)

X0 = tf.matmul(s, PhiInv)

PhiTb = tf.matmul(s, tf.transpose(Phi_input))

def add_con2d_weight_bias(w_shape, b_shape, order_no):
    Weights = tf.get_variable(shape=w_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='Weights_%d' % order_no)
    biases = tf.Variable(tf.random_normal(b_shape, stddev=0.05), name='biases_%d' % order_no)
    return [Weights, biases]

'''
def prelu(_x, name=None, alpha=None):
    """parametric ReLU activation"""
    if alpha == None:
        with tf.variable_scope("prelu"):
            _alpha = tf.get_variable(name='alpha'+name, shape=_x.get_shape()[-1], dtype=_x.dtype,
                                 initializer=tf.constant_initializer(0.1))
    else:
        _alpha = alpha
    pos = tf.nn.relu(_x)
    neg = _alpha * (_x - tf.abs(_x)) * 0.5

    return pos + neg
'''

def ista_block(input_layers, input_data, layer_no):
    tau_value = tf.Variable(0.1, dtype=tf.float32)
    lambda_step = tf.Variable(0.1, dtype=tf.float32)
    soft_thr = tf.Variable(0.1, dtype=tf.float32)
    conv_size = 32
    filter_size = 3

    x1_ista = tf.add(input_layers[-1] - tf.scalar_mul(lambda_step, tf.matmul(input_layers[-1], PhiTPhi)), tf.scalar_mul(lambda_step, PhiTb))  # X_k - lambda*A^TAX

    x2_ista = tf.reshape(x1_ista, shape=[-1, img_height, img_width, img_channels])

    [Weights0, bias0] = add_con2d_weight_bias([filter_size, filter_size, 2, conv_size], [conv_size], 0)

    [Weights1, bias1] = add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 1)
    [Weights11, bias11] = add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 11)

    [Weights2, bias2] = add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 2)
    [Weights22, bias22] = add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 22)

    [Weights3, bias3] = add_con2d_weight_bias([filter_size, filter_size, conv_size, 2], [1], 3)

    x3_ista = tf.nn.conv2d(x2_ista, Weights0, strides=[1, 1, 1, 1], padding='SAME')

    x4_ista =tf.nn.relu(tf.nn.conv2d(x3_ista, Weights1, strides=[1, 1, 1, 1], padding='SAME'))
    x44_ista = tf.nn.conv2d(x4_ista, Weights11, strides=[1, 1, 1, 1], padding='SAME')

    x5_ista = tf.multiply(tf.sign(x44_ista), (tf.abs(x44_ista) - soft_thr))


    x6_ista = tf.nn.relu(tf.nn.conv2d(x5_ista, Weights2, strides=[1, 1, 1, 1], padding='SAME'))
    x66_ista = tf.nn.conv2d(x6_ista, Weights22, strides=[1, 1, 1, 1], padding='SAME')

    x7_ista = tf.nn.conv2d(x66_ista, Weights3, strides=[1, 1, 1, 1], padding='SAME')

    x7_ista = x7_ista + x2_ista
    #x7_ista = x7_ista

    #x_in = tf.reshape(input_layers, shape=[-1, img_height, img_width, img_channels])
    x8_ista = tf.reshape(x7_ista, shape=[-1, img_total])
    x3_ista_sym = tf.nn.leaky_relu(tf.nn.conv2d(x3_ista, Weights1, strides=[1, 1, 1, 1], padding='SAME'))
    x4_ista_sym = tf.nn.conv2d(x3_ista_sym, Weights11, strides=[1, 1, 1, 1], padding='SAME')
    x6_ista_sym = tf.nn.leaky_relu(tf.nn.conv2d(x4_ista_sym, Weights2, strides=[1, 1, 1, 1], padding='SAME'))
    x7_ista_sym = tf.nn.conv2d(x6_ista_sym, Weights22, strides=[1, 1, 1, 1], padding='SAME')

    x11_ista = x7_ista_sym - x3_ista

    return [x8_ista, x11_ista]


def inference_ista(input_tensor, n, X_output, reuse):
    layers = []
    layers_symetric = []
    layers.append(input_tensor)
    for i in range(n):
        with tf.variable_scope('conv_%d' %i, reuse=reuse):
            [conv1, conv1_sym] = ista_block(layers, X_output, i)
            layers.append(conv1)
            layers_symetric.append(conv1_sym)
    return [layers, layers_symetric]


[Prediction, Pre_symetric] = inference_ista(X0, PhaseNumber, X_output, reuse=False)

cost0 = tf.reduce_mean(tf.square(X0 - X_output))


def compute_cost(Prediction, Pre_symetric, X_output, PhaseNumber):
    cost = tf.reduce_mean(tf.reduce_sum(tf.square(Prediction[-1] - X_output), axis=1))
    cost_sym = 0
    for k in range(PhaseNumber):
        cost_sym += tf.reduce_mean(tf.reduce_sum(tf.square(Pre_symetric[k]), axis=1))

    return [cost, cost_sym]


[cost, cost_sym] = compute_cost(Prediction, Pre_symetric, X_output, PhaseNumber)


cost_all = cost + 0.01*cost_sym


optm_all = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_all)

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = False

saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

sess = tf.Session(config=config)
sess.run(init)

print("...............................")
print("Phase Number is %d, CS ratio is %d%%" % (PhaseNumber, encoded_dim))
print("...............................\n")



model_dir = 'Phase_%d_encod_%s_%d_Learnpro' % (PhaseNumber, envir, encoded_dim)

output_file_name = "Log_output_%s.txt" % (model_dir)

#val_input = np.dot(x_val, Phi_input)

val_min = 100

for epoch_i in range(0, EpochNum+1):
    randidx_all = np.random.permutation(nrtrain)
    loss_tr = 0
    if epoch_i <= warmup_step:
        apply_rate = end_learning_rate + (epoch_i/warmup_step)*(initial_learning_rate-end_learning_rate)
    else:
        apply_rate = (initial_learning_rate - end_learning_rate) * (1 - (epoch_i-warmup_step) / decay_steps)**power + end_learning_rate
    for batch_i in range(nrtrain // batch_size):
        randidx = randidx_all[batch_i*batch_size:(batch_i+1)*batch_size]

        batch_ys = x_train[randidx, :]
        #batch_xs = np.dot(batch_ys, Phi_input)

        feed_dict = {X_input: batch_ys, X_output: batch_ys, learning_rate: apply_rate}
        sess.run(optm_all, feed_dict=feed_dict)
        print('\r', end='')
        loss_temp = sess.run(cost, feed_dict=feed_dict)
        loss_tr = loss_temp * batch_size + loss_tr
        print("epoch Progress: {}/{}:{}".format(batch_i, nrtrain // batch_size, loss_temp), end="", flush=True)
    
    loss_val = 0
    for batch_i in range(nrval// batch_size):
        randidx = randidx_all[batch_i*batch_size:(batch_i+1)*batch_size]

        batch_ys = x_train[randidx, :]
        #batch_xs = np.dot(batch_ys, Phi_input)

        feed_dict_val = {X_input: batch_ys, X_output: batch_ys, learning_rate: apply_rate}
        print('\r', end='')
        loss_temp = sess.run(cost, feed_dict=feed_dict_val)
        loss_val = loss_temp * batch_size + loss_val
        print("epoch Progress: {}/{}:{}".format(batch_i, nrval // batch_size, loss_temp), end="", flush=True)
    
    val_loss = loss_val/nrval

    output_data = "[{}/{}], train:{}  val:{} lr{},  \n".format(epoch_i, EpochNum, loss_tr/nrtrain, val_loss, apply_rate)
    print(output_data)
    print('--------------------------')

    output_file = open(output_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if epoch_i <= 60 and epoch_i % 20 == 0:
        saver.save(sess, './%s/CS_Saved_Model_%d.cpkt' % (model_dir, epoch_i), write_meta_graph=False)
    else:
        if epoch_i % 40 == 0:
            saver.save(sess, './%s/CS_Saved_Model_%d.cpkt' % (model_dir, epoch_i), write_meta_graph=False)
        if val_loss < val_min:
            val_min = val_loss
            saver.save(sess, './%s/CS_Saved_Model_%d.cpkt' % (model_dir, -1), write_meta_graph=False)


print("Training Finished")

#test_input = np.dot(x_test, Phi_input)
feed_dict_test = {X_input: x_test[0:500, :], X_output: x_test[0:500, :]}
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
    x_test_C = x_test_real + 1j * (x_test_imag)
    x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
    x_hat_C = x_hat_real + 1j * (x_hat_imag)
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
[mse, rho, power] = nmse_rho(X_test[0:500, :], output_test)

print("In " + envir + " environment")
print("When dimension is", encoded_dim)
print("NMSE is ", 10 * math.log10(np.mean(mse / power)))
print("Correlation is ", np.mean(rho))
print("=========================================")


sess.close()