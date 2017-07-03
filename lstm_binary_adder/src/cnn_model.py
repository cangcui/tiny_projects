# coding: utf-8

import tensorflow as tf


class CNNModel(object):
    def __init__(
            self,
            plane_height,
            plane_width,
            channels,
            layers,
            filter_sizes_dim0,      # 2 dim, size: [layers, filter_sizes]
            filter_sizes_dim1,      # 2 dim, size: [layers, filter_sizes]
            num_filters,            # 1 dim, size: [layers]: every layer's every filter_size's filters number
            output_shape):
        assert len(filter_sizes_dim0) == len(filter_sizes_dim1)
        assert len(filter_sizes_dim0[0]) == len(filter_sizes_dim1[0])
        assert len(filter_sizes_dim0) == len(num_filters)

        self.plane_height = plane_height
        self.plane_width = plane_width
        self.channels = channels
        self.layers = layers
        self.filter_sizes_dim0 = filter_sizes_dim0
        self.filter_sizes_dim1 = filter_sizes_dim1
        self.num_filters = num_filters
        self.output_shape = output_shape
        self.l2_loss = 0.0
        self.input_x = None
        self.input_y = None

    def cnn_graph(self):
        self.input_x = tf.placeholder(tf.float32, [None, self.plane_height, self.plane_width, self.channels])
        self.input_y = tf.placeholder(tf.float32, self.output_shape)


    def _cnn_layer(self, layer_input, layer_index):
        assert layer_index < self.layers
        layer_filter_sizes_dim0 = self.filter_sizes_dim0[layer_index] # list of first dimension's size
        layer_filter_sizes_dim1 = self.filter_sizes_dim1[layer_index] # list of second dimension's size
        layer_num_filters = self.num_filters[layer_index]

        with tf.name_scope("layer-%s" % (layer_index + 1)):
            layer_outputs = []
            for i in xrange(len(layer_filter_sizes_dim0)):
                h = self._cnn_node(
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

    def _cnn_node(self, layer_input, num_filter, filter_size_dim0, filter_size_dim1):
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

    def _loss_layer(self, layer_input, y):
        pass




