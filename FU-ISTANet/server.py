import tensorflow as tf
import scipy.io as sio
import numpy as np
import os
from Model import ISTA_net
from clients import clients
import math
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--numOfClients", type=int, help="num 0f Clients")
args = parser.parse_args()

EpochNum = 60
apply_rate = 0.0002
learning_rate =tf.placeholder(tf.float32)
PhaseNumber = 5
encoded_dim = 512

img_height = 32
img_width = 32
img_channels = 2
img_total = img_height * img_width * img_channels

n_input = img_total
n_output = img_total
envir = 'indoor'  # 'indoor' or 'outdoor' 


Phi_input = tf.get_variable(shape=[img_total, encoded_dim], initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                              name='Phi_input')
PhiInv = tf.get_variable(shape=[encoded_dim,img_total], initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                              name='PhiInv')
PhiTPhi = tf.get_variable(shape=[img_total,img_total], initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                              name='PhiTPhi')

X_input = tf.placeholder(tf.float32, [None, n_input])
X_output = tf.placeholder(tf.float32, [None, n_output])

myModel = ISTA_net(encoded_dim, Phi_input, PhiInv, PhiTPhi, PhaseNumber, X_input, X_output, learning_rate) #基本模型

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = False

saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

sess = tf.Session(config=config)
sess.run(init)

print(tf.test.is_gpu_available())
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




print("shape:", x_train.shape[0], "|", x_test.shape[0], "|", x_val.shape[0])
x_test =  x_test[0:500, :]
x_train = x_train - 0.5
x_val = x_val - 0.5
x_test = x_test - 0.5


numOfClients = args.numOfClients
fraction = 0.8
bLocalBatchSize = 400
eLocalEpoch = 10
optim = myModel.optm_all
cost = myModel.cost



myClients = clients(numOfClients, bLocalBatchSize,
                 eLocalEpoch, sess, optim, X_input, X_output, x_train, cost, learning_rate)

vars = tf.trainable_variables()
global_vars = sess.run(vars)
num_in_comm = int(max(numOfClients * fraction, 1))

print("...............................")
print("Phase Number is %d, CS ratio is %d%%" % (PhaseNumber, encoded_dim))
print("...............................\n")


model_dir = 'Phase_%d_encod_0_%d_Learn_%dclient' % (PhaseNumber, encoded_dim,numOfClients)

output_file_name = "Log_output_%s.txt" % (model_dir)

#val_input = np.dot(x_val, Phi_input)
#feed_dict_val = {X_input: x_val[0:500,:], X_output: x_val[0:500,:]}

val_min = 100
flag = 1


for epoch_i in range(0, EpochNum+1):
    print("communicate round {}".format(epoch_i))
    order = np.arange(numOfClients)
    np.random.shuffle(order)
    clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

    sum_vars = None
    for client in clients_in_comm:
        local_vars = myClients.ClientUpdate(client, global_vars, apply_rate)
        if sum_vars is None:
            sum_vars = local_vars
        else:
            for sum_var, local_var in zip(sum_vars, local_vars):
                sum_var += local_var

    global_vars = []
    for var in sum_vars:
        global_vars.append(var / num_in_comm)
    
    loss_val = 0
    for variable, value in zip(vars, global_vars):
        variable.load(value, sess)
        
        batch_size = 200
        for batch_i in range(nrval// batch_size):
        randidx = randidx_all[batch_i*batch_size:(batch_i+1)*batch_size]

        batch_ys = x_train[randidx, :]
        #batch_xs = np.dot(batch_ys, Phi_input)

        feed_dict_val = {X_input: batch_ys, X_output: batch_ys, learning_rate: apply_rate}
        print('\r', end='')
        loss_temp = sess.run(myModel.cost, feed_dict=feed_dict_val)
        loss_val = loss_temp * batch_size + loss_val
        print("epoch Progress: {}/{}:{}".format(batch_i, nrval // batch_size, loss_temp), end="", flush=True)
    
    val_loss = loss_val/nrval


    output_data = "[{}/{}] val:{}, lr:{}  \n".format(epoch_i, EpochNum, val_loss, apply_rate)
    print(output_data)
    print('--------------------------')

    output_file = open(output_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if epoch_i <= 30:
        saver.save(sess, './%s/CS_Saved_Model_%d.cpkt' % (model_dir, epoch_i), write_meta_graph=False)
    elif epoch_i % 10 == 0:
        saver.save(sess, './%s/CS_Saved_Model_%d.cpkt' % (model_dir, epoch_i), write_meta_graph=False)
    if val_loss < val_min:
        val_min = val_loss
        saver.save(sess, './%s/CS_Saved_Model_%d.cpkt' % (model_dir, -1), write_meta_graph=False)

print("Training Finished")