# coding: utf-8

import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


class BinaryAdder(object):
    def __init__(self):
        self.epochs = 1000
        self.max_binary_dim = 7
        self.batch_size = 5
        self.num_classes = 2
        self.hidden_dim = self.max_binary_dim + 5
        self.total_series_length = 50000 * self.max_binary_dim * self.batch_size
        self.num_batches = self.total_series_length // self.batch_size // self.max_binary_dim

        assert self.hidden_dim >= self.max_binary_dim

    def run(self):
        os.putenv('CUDA_VISIBLE_DEVICES', '')
        data = tf.placeholder(tf.float32, [None, self.hidden_dim, 2])
        target = tf.placeholder(tf.float32, [None, self.hidden_dim])

        data_seq = tf.unstack(data, num=self.hidden_dim, axis=1)
        # define the rnn
        cell = rnn.BasicLSTMCell(self.hidden_dim, state_is_tuple=True)
        # cell = rnn.GRUCell(self.hidden_dim)
        # outputs should be the size of (batch_size, hidden_dim)
        outputs, states = rnn.static_rnn(cell, data_seq, dtype=tf.float32)
        last = outputs[-1]

        # add a softmax layer
        weight = tf.Variable(tf.truncated_normal([self.hidden_dim, int(target.get_shape()[1])]))
        bias = tf.Variable(tf.constant(0.1, shape=[int(target.get_shape()[1])]))
        # prediction is the shape[self.batch_size, self.hidden_dim]
        prediction = tf.matmul(last, weight) + bias   # prediction tensor is the result

        # define the cost
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=target))
        # cost = tf.reduce_sum(tf.abs(target - tf.clip_by_value(prediction, 1e-10, 1.0)))
        cost = tf.reduce_mean(tf.abs(target - prediction))

        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

        pred_rounded = tf.round(prediction)
        mistakes = tf.not_equal(target, pred_rounded)
        error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

        # begin the training
        init_op = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init_op)
        for i in xrange(self.epochs):
            print '*********epoch: %d**********' % i
            xs1, xs2 = self.generateData(self.total_series_length)
            # 记录当前取到哪个数据
            ptr = 0
            display_batch = 1000
            zs = [[0] * (self.hidden_dim - self.max_binary_dim) for i in xrange(self.batch_size)]
            for j in xrange(self.num_batches):
                x1 = xs1[ptr:(ptr + self.batch_size * self.max_binary_dim)]
                x2 = xs2[ptr:(ptr + self.batch_size * self.max_binary_dim)]
                ptr += self.batch_size * self.max_binary_dim

                x1 = np.array(x1.reshape([self.batch_size, self.max_binary_dim]))
                x2 = np.array(x2.reshape([self.batch_size, self.max_binary_dim]))
                x1 = np.concatenate((zs, x1), axis=1)
                x2 = np.concatenate((zs, x2), axis=1)

                y = np.array([self.binArrAdder(x1[i], x2[i]) for i in xrange(self.batch_size)])

                x1 = x1.reshape([self.batch_size, self.hidden_dim, 1])
                x2 = x2.reshape([self.batch_size, self.hidden_dim, 1])
                xs = np.concatenate((x1, x2), axis=2)

                sess.run(optimizer, feed_dict={data: xs, target: y})

                # print 'last: ', sess.run(last, feed_dict={data: xs, target: y})
                # print 'data, shape: ', xs.shape
                # print 'target, shape: ', y.shape
                # print 'data_seq: ', sess.run(data_seq, feed_dict={data: xs, target: y})
                if j % display_batch == 0:
                    err = sess.run(error, feed_dict={data: xs, target: y})
                    print 'current batch: %d, err: %f' % (j, err)
                    tx1, tx2 = self.generateData(self.max_binary_dim)
                    # tx1 = np.array([1 for i in xrange(self.max_binary_dim)])
                    # tx2 = np.array([1 for i in xrange(self.max_binary_dim)])
                    ty = np.array([self.binArrAdder(tx1, tx2)])
                    txs = self.plastic_input(tx1, tx2, [[0] * (self.hidden_dim - self.max_binary_dim)])
                    print 'tx1: ', tx1, ', 10-base: ', self.binArrToBase10(tx1)
                    print 'tx2: ', tx2, ', 10-base: ', self.binArrToBase10(tx2)
                    print 'ty: ', ty, ', 10-base: ', self.binArrToBase10(ty[0])
                    p = sess.run(pred_rounded, feed_dict={data: txs, target: ty})
                    print 'pred: ', p, ', 10-base: ', self.binArrToBase10(p[0])

    def plastic_input(self, x1, x2, zs):
        x1 = np.array(x1.reshape([1, self.max_binary_dim]))
        x2 = np.array(x2.reshape([1, self.max_binary_dim]))
        x1 = np.concatenate((zs, x1), axis=1)
        x2 = np.concatenate((zs, x2), axis=1)

        x1 = x1.reshape([1, self.hidden_dim, 1])
        x2 = x2.reshape([1, self.hidden_dim, 1])
        return np.concatenate((x1, x2), axis=2)

    def generateData(self, data_len):
        assert data_len > 0
        x1 = np.array(np.random.choice(2, data_len, p=[0.5, 0.5]))
        x2 = np.array(np.random.choice(2, data_len, p=[0.5, 0.5]))
        return x1, x2

    def supplementZeros(self, binArr):
        if isinstance(binArr, list):
            if len(binArr) == self.hidden_dim:
                return binArr
            else:
                return [0 for i in xrange(self.hidden_dim - len(binArr))].extend(binArr)
        else:
            return None

    def binArrToBase10(self, arr):
        len_arr = len(arr)
        return sum([arr[i] * (2 ** (len_arr - 1 - i)) for i in xrange(len_arr)])

    def binArrAdder(self, bin_seq_1, bin_seq_2):
        len_seq1 = len(bin_seq_1)
        len_seq2 = len(bin_seq_2)
        assert len_seq1 <= self.hidden_dim and len_seq2 <= self.hidden_dim

        uint_1 = sum([bin_seq_1[i] * (2 ** (len_seq1 - 1 - i)) for i in xrange(len_seq1)])
        uint_2 = sum([bin_seq_2[i] * (2 ** (len_seq2 - 1 - i)) for i in xrange(len_seq2)])
        s = bin(uint_1 + uint_2)
        # 不到32位的话，前面补0
        s_bin = [int(b) for b in s[2:]]
        if len(s_bin) == self.hidden_dim:
            return np.array(s_bin)
        else:
            final = [0 for i in xrange(self.hidden_dim - len(s_bin))]
            final.extend(s_bin)
            return np.array(final)


def main():
    ba = BinaryAdder()
    ba.run()


if __name__ == '__main__':
    main()

