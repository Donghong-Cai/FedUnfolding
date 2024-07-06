import numpy as np
import tensorflow as tf


class clients(object):
    def __init__(self, numOfClients, bLocalBatchSize,
                 eLocalEpoch, sess, optim, X_input, X_output, data, cost,learning_rate ):
        self.num_of_clients = numOfClients
        self.data = data

        self.dataset_size = data.shape[0]
        self.test_data = None
        self.test_label = None
        self.B = bLocalBatchSize
        self.E = eLocalEpoch
        self.session = sess
        self.optim = optim
        self.X_input = X_input
        self.X_output = X_output
        self.learnig_rate = learning_rate
        self.clientsSet = {}
        self.cost = cost

        self.dataset_balance_allocation()

    def dataset_balance_allocation(self):

        localDataSize = self.dataset_size // self.num_of_clients
        print("datasize:", self.dataset_size," number:", self.num_of_clients)
        #shards_id = np.random.permutation(self.dataset_size // localDataSize)

        for i in range(self.num_of_clients):
            data_shards = self.data[i * localDataSize: i * localDataSize + localDataSize]
            self.clientsSet['client{}'.format(i)] = data_shards
            print(data_shards.shape[0])

    def ClientUpdate(self,  client,  global_vars, apply_rate):
        all_vars = tf.trainable_variables()
        for variable, value in zip(all_vars, global_vars):
            variable.load(value, self.session)

        for epoch_i in range(self.E):
            nrtrain = self.clientsSet[client].shape[0]
            randidx_all = np.random.permutation(nrtrain)
            for batch_i in range(nrtrain // self.B):
                randidx = randidx_all[batch_i * self.B:(batch_i + 1) * self.B]

                batch_ys = self.clientsSet[client][randidx, :]

                feed_dict = {self.X_input: batch_ys, self.X_output: batch_ys, self.learnig_rate: apply_rate}
                self.session.run(self.optim, feed_dict=feed_dict)
                print('\r', end='')
                print("{} epoch {} Progress: {}/{}:{}".format(client,epoch_i,batch_i, nrtrain // self.B,
                                                        self.session.run(self.cost, feed_dict=feed_dict)),
                      end="", flush=True)

        return self.session.run(tf.trainable_variables())

