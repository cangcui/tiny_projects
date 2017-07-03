# coding: utf-8

import tensorflow as tf


class CNNModel(object):
    def __init__(
            self,
            plane_height,
            plane_width,
            channels,
            hidden_layers,
            filter_sizes_dim0,      # 2 dim, size: [layers, filter_sizes]
            filter_sizes_dim1,      # 2 dim, size: [layers, filter_sizes]
            num_filters,            # 1 dim, size: [layers]: every layer's every filter_size's filters number
            output_len,
            l2_reg_lambda=0.0,
            is_dense_output=False
    ):
        assert len(filter_sizes_dim0) == len(filter_sizes_dim1)
        assert len(filter_sizes_dim0[0]) == len(filter_sizes_dim1[0])
        assert len(filter_sizes_dim0) == len(num_filters)

        self.plane_height = plane_height
        self.plane_width = plane_width
        self.channels = channels
        self.hidden_layers = hidden_layers
        self.filter_sizes_dim0 = filter_sizes_dim0
        self.filter_sizes_dim1 = filter_sizes_dim1
        self.num_filters = num_filters
        self.num_classes = output_len
        self.l2_reg_lambda = l2_reg_lambda
        self.l2_loss = 0.0
        self.input_x = tf.placeholder(tf.float32, [None, self.plane_height, self.plane_width, self.channels])
        self.input_y = tf.placeholder(tf.float32, [output_len])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.hidden_output = None
        self.hidden_output_flat = None
        self.scores = None
        self.predictions = None
        self.loss = None
        self.accuracy = None
        self.is_dense_output = is_dense_output

        self._cnn_model()

    def _cnn_model(self):
        # cnn layers
        layer_input = self.input_x
        layer_outputs = []
        output_channels = -1
        for i in xrange(self.hidden_layers):
            hs, output_channels = self._cnn_layer(layer_input, i)
            layer_input = hs
            layer_outputs.append(hs)
        self.hidden_output = layer_outputs[-1]

        # flatten
        self.hidden_output_flat = tf.reshape(self.hidden_output, [-1, output_channels])
        # drop
        self._dropout()
        # output layer
        self._def_output_layer(output_channels)
        # loss
        self._def_loss()
        # accuracy
        self._def_accuracy()

    def _def_output_layer(self, output_channels):
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[output_channels, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(
                "b",
                shape=[self.num_classes],
                initializer=tf.constant_initializer(0.1))
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            if not self.is_dense_output:
                self.predictions = tf.argmax(self.scores, axis=1, name="predictions")
            else:
                self.predictions = None

    def _def_loss(self):
        with tf.name_scope("loss"):
            if not self.is_dense_output:
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss
            else:
                # for example, input_y = [1,0,1,0,1], this is a dense representation,
                # input_y=[1,0,0,0,0] this is a sparce representation
                self.loss = tf.reduce_mean(tf.abs(self.scores - self.input_y))

    def _def_accuracy(self):
        with tf.name_scope("accuracy"):
            if not self.is_dense_output:
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            else:
                scores_rounded = tf.round(self.scores)
                self.accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(self.input_y, scores_rounded), "float"),
                    name="accuracy"
                )

    def _dropout(self):
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.hidden_output_flat, self.dropout_keep_prob)

    def _cnn_layer(self, layer_input, layer_index):
        assert layer_index < self.hidden_layers
        layer_filter_sizes_dim0 = self.filter_sizes_dim0[layer_index] # list of first dimension's size
        layer_filter_sizes_dim1 = self.filter_sizes_dim1[layer_index] # list of second dimension's size
        layer_num_filters = self.num_filters[layer_index]

        with tf.name_scope("layer-%s" % (layer_index + 1)):
            layer_outputs = []
            for i in xrange(len(layer_filter_sizes_dim0)):
                h = self._conv_node(
                    layer_input,
                    layer_num_filters,
                    layer_filter_sizes_dim0[i],
                    layer_filter_sizes_dim1[i])
                layer_outputs.append(h)
            layer_num_filters_total = layer_num_filters * len(layer_filter_sizes_dim0)
            # hs's shape: [batch_size, in_height,
            #              in_width, layer_num_filters_total]
            hs = tf.concat(layer_outputs, 3)
            return hs, layer_num_filters_total

    def _conv_node(self, layer_input, num_filter, filter_size_dim0, filter_size_dim1):
        with tf.name_scope("conv-%s-%s" % (filter_size_dim0, filter_size_dim1)):
            # conv's shape: [batch_size, in_height, in_width, num_filter]
            conv = tf.layers.conv2d(
                inputs=layer_input,
                filters=num_filter,
                kernel_size=[filter_size_dim0, filter_size_dim1],
                padding='same',
                activation=tf.nn.relu
            )

            return conv




