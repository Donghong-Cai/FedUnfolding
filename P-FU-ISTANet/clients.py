import numpy as np
import tensorflow as tf


class clients(object):
    def __init__(self, numOfClients, bLocalBatchSize,
                 eLocalEpoch, model, sess, X_input, X_output, data):
        self.num_of_clients = numOfClients
        self.data = data

        self.dataset_size = data.shape[0]
        self.test_data = None
        self.test_label = None
        self.B = bLocalBatchSize
        self.E = eLocalEpoch
        self.model = model
        self.X_input = X_input
        self.X_output = X_output
        self.clientsSet = {}

        self.dataset_balance_allocation()
        self.sess = sess
        self.grad = self.model.get_grad(tf.trainable_variables())
        # self.update = self.optim_q.apply_gradients(self.grad)

    def dataset_balance_allocation(self):

        localDataSize = self.dataset_size // self.num_of_clients
        print("datasize:", self.dataset_size," number:", self.num_of_clients)
        #shards_id = np.random.permutation(self.dataset_size // localDataSize)

        for i in range(self.num_of_clients):
            data_shards = self.data[i * localDataSize: i * localDataSize + localDataSize]
            self.clientsSet['client{}'.format(i)] = data_shards
            print(data_shards.shape[0])

    def ClientUpdate(self, client, global_vars):
        all_vars = tf.global_variables()
        for variable, value in zip(all_vars, global_vars):
            variable.load(value, self.sess)
        
        nrtrain = self.clientsSet[client].shape[0]
        ntrain_s = int(nrtrain * 0.8)
        ntrain_q = int(nrtrain * 0.2)
        rand_set = np.random.permutation(nrtrain)
        randid_s = rand_set[0:ntrain_s]
        randid_q = rand_set[ntrain_s:]
        clientSet_s = self.clientsSet[client][randid_s, :]
        clientSet_q = self.clientsSet[client][randid_q, :]
        
        #batch_q = int(self.B*ntrain_q/ntrain_s)
        #print("batch_q = ", batch_q)

       # support set train
        for epoch_i in range(self.E):
            randidx_all = np.random.permutation(ntrain_s)
            # temp = tf.trainable_variables()
            grad_inner = []
            for batch_i in range(ntrain_s // self.B):
                global_vars = self.sess.run(tf.trainable_variables())
            
                randidx = randidx_all[batch_i * self.B:(batch_i + 1) * self.B]

                batch_ys = clientSet_s[randidx, :]

                feed_dict = {self.X_input: batch_ys, self.X_output: batch_ys}
                self.sess.run(self.model.optm_all, feed_dict=feed_dict)
                loss_temp = self.sess.run(self.model.cost, feed_dict=feed_dict)

                print('\r', end='')
                print("{} epoch{} Progress: {}/{}:{}".format(client, epoch_i, batch_i, ntrain_s // self.B, loss_temp),
                end="", flush=True)


            # query set train
            grads = []
            for batch_i in range(ntrain_q // self.B):
                randidx_all = np.random.permutation(ntrain_q)
                

                randidx = randidx_all[batch_i * self.B:(batch_i + 1) * self.B]
                batch_ys = clientSet_q[randidx, :]
                feed_dict = {self.X_input: batch_ys, self.X_output: batch_ys}
                    
                cost_temp = self.sess.run(self.model.cost_all, feed_dict=feed_dict)

                grad_temp = self.sess.run(self.grad, feed_dict=feed_dict)

                grads.append(grad_temp)

                print('\r', end='')
                print("Query {} epoch{} Progress: {}/{}:{}".format(client, epoch_i, batch_i, ntrain_s // self.B, cost_temp),
                end="", flush=True)

                

            for variable, value in zip(all_vars, global_vars):   
                variable.load(value, self.sess)

            grads_out = np.mean(grads, axis=0)
            grads_and_vars = {self.model.grad_placeholder[i][0]: grads_out[i][0] for i in range(len(grads_out))}


            self.sess.run(self.model.optim_q, feed_dict=grads_and_vars)


        return self.sess.run(tf.trainable_variables())

