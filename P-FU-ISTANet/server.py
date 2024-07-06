import tensorflow as tf
import scipy.io as sio
import numpy as np
import os
from Model import ISTA_net
from clients import clients
import h5py
import math

EpochNum = 60
learning_rate = 0.0002
PhaseNumber = 5
encoded_dim = 128

img_height = 32
img_width = 32
img_channels = 2
img_total = img_height * img_width * img_channels

n_input = img_total
n_output = img_total
envir = 'cdl'  # 'indoor' or 'outdoor' or 'all' or ‘cdl’


Phi_input = tf.get_variable(shape=[img_total, encoded_dim], initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                              name='Phi_input')
PhiInv = tf.get_variable(shape=[encoded_dim,img_total], initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                              name='PhiInv')


X_input = tf.placeholder(tf.float32, [None, n_input])
X_output = tf.placeholder(tf.float32, [None, n_output])

myModel = ISTA_net(encoded_dim, Phi_input, PhiInv,  PhaseNumber, X_input, X_output, learning_rate) #基本模型

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = False

saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

sess = tf.Session(config=config)
sess.run(init)
sess.run(tf.local_variables_initializer()) # 函数内参数初始化

print("...............................")
print("Phase Number is %d, CS ratio is %d%%" % (PhaseNumber, encoded_dim))
print('Total number of the parameters: ', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

#model_dir = 'Phase_%d_encod_0_%d_ISTA_Net_plus_Model' % (PhaseNumber, 512)
#saver.restore(sess, './%s/CS_Saved_Model_380.cpkt' % (model_dir))

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

elif envir == 'cdl':
    mat = h5py.File('../processedData_CDL_A1/CDLChannelEst_train_processed.mat',"r")
    x_train = mat['H'][:]  # array
    mat = h5py.File('../processedData_CDL_B1/CDLChannelEst_train_processed.mat',"r")
    x_train = np.append(x_train, mat['H'][:], axis=0)  # array
    mat = h5py.File('../processedData_CDL_D1/CDLChannelEst_train_processed.mat',"r")
    x_train = np.append(x_train, mat['H'][:], axis=0)  # array
    mat = h5py.File('../processedData_CDL_Av2/CDLChannelEst_train_processed.mat',"r")
    x_train = np.append(x_train, mat['H'][:], axis=0)  # array

    mat = h5py.File('../processedData_CDL_A1/CDLChannelEst_val_processed.mat',"r")
    x_val = mat['H'][:]  # array
    mat = h5py.File('../processedData_CDL_B1/CDLChannelEst_val_processed.mat',"r")
    x_val = np.append(x_val, mat['H'][:], axis=0)  # array
    mat = h5py.File('../processedData_CDL_D1/CDLChannelEst_val_processed.mat',"r")
    x_val = np.append(x_val, mat['H'][:], axis=0)  # array
    mat = h5py.File('../processedData_CDL_Av2/CDLChannelEst_val_processed.mat',"r")
    x_val = np.append(x_val, mat['H'][:], axis=0)  # array

    mat = h5py.File('../processedData_CDL_A1/CDLChannelEst_test_processed.mat',"r")
    x_test = mat['H'][:]  # array
    mat = h5py.File('../processedData_CDL_B1/CDLChannelEst_test_processed.mat',"r")
    x_test = np.append(x_test, mat['H'][:], axis=0)  # array
    mat = h5py.File('../processedData_CDL_D1/CDLChannelEst_test_processed.mat',"r")
    x_test = np.append(x_test, mat['H'][:], axis=0)  # array
    mat = h5py.File('../processedData_CDL_Av2/CDLChannelEst_test_processed.mat',"r")
    x_test = np.append(x_test, mat['H'][:], axis=0)  # array

print("shape:", x_train.shape[0], "|", x_test.shape[0], "|", x_val.shape[0])
x_test =  x_test[0:500, :] - 0.5
x_train = x_train - 0.5
x_val = x_val - 0.5
x_test = x_test - 0.5

# 本地训练参数设置
numOfClients = 4
fraction = 1
bLocalBatchSize = 100
eLocalEpoch = 10

vars = tf.global_variables()
global_vars = sess.run(vars)
num_in_comm = int(max(numOfClients * fraction, 1))

myClients = clients(numOfClients, bLocalBatchSize,
                 eLocalEpoch, myModel, sess, X_input, X_output, x_train)

print("...............................")
print("Phase Number is %d, CS ratio is %d%%" % (PhaseNumber, encoded_dim))
print("...............................\n")


model_dir = 'Phase_%d_encod_%s_%d_ISTA_mate_batch%d' % (PhaseNumber,envir, encoded_dim,bLocalBatchSize)

output_file_name = "Log_output_%s.txt" % (model_dir)

#val_input = np.dot(x_val, Phi_input)
feed_dict_val = []
for i in range(0,numOfClients):
    index = i*5000
    feed_temp = {X_input: x_val[index:index+5000,:], X_output: x_val[index:index+5000,:]}
    feed_dict_val.append(feed_temp)

# for i in range(0,numOfClients):  #test code
    # index = i*500
    # feed_temp = {X_input: x_val[index:index+500,:], X_output: x_val[index:index+500,:]}
    # feed_dict_val.append(feed_temp)


val_min = 100
flag = 1

for epoch_i in range(0, EpochNum+1):
    print("communicate round {}".format(epoch_i))
    order = np.arange(numOfClients)
    np.random.shuffle(order)
    clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

    sum_vars = None
    for client in clients_in_comm:
        local_vars = myClients.ClientUpdate(client, global_vars)
        if sum_vars is None:
            sum_vars = local_vars
        else:
            for sum_var, local_var in zip(sum_vars, local_vars):
                sum_var += local_var


    global_vars = []
    for var in sum_vars:
        global_vars.append(var / num_in_comm)

    for variable, value in zip(vars, global_vars):
        variable.load(value, sess)
    
    val_loss = 0 
    out_part = ""
    for i in range(0,numOfClients):
        #val_loss_temp = sess.run(myModel.cost, feed_dict=feed_dict_val[i]) #test code

        loss_val = 0
        for variable, value in zip(vars, global_vars):
            variable.load(value, sess)
            
            batch_size = 200
            nrval = 5000
            for batch_i in range(nrval// batch_size):
            randidx = randidx_all[batch_i*batch_size:(batch_i+1)*batch_size]

            batch_ys = feed_dict_val[i][randidx, :]
            #batch_xs = np.dot(batch_ys, Phi_input)

            feed_dict = {X_input: batch_ys, X_output: batch_ys, learning_rate: apply_rate}
            print('\r', end='')
            loss_temp = sess.run(myModel.cost, feed_dict=feed_dict)
            loss_val = loss_temp * batch_size + loss_val
            print("epoch Progress: {}/{}:{}".format(batch_i, nrval // batch_size, loss_temp), end="", flush=True)
        
        val_loss_temp = loss_val/nrval
        
        out_part = out_part+"val{}:{} ".format(i,val_loss_temp)
        val_loss = val_loss+val_loss_temp/numOfClients
    output_data = "[{}/{}]  val:{}, ".format (epoch_i, EpochNum, val_loss)
    output_data = output_data + out_part + "\n"
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
        if epoch_i % 1 == 0:
            saver.save(sess, './%s/CS_Saved_Model_%d.cpkt' % (model_dir, epoch_i), write_meta_graph=False)
        elif val_loss < val_min:
            val_min = val_loss
            saver.save(sess, './%s/CS_Saved_Model_%d.cpkt' % (model_dir, -1), write_meta_graph=False)

print("Training Finished")