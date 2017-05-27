import tensorflow as tf
import os
import numpy as np
from gmc.conf import settings

class CNN:
    def __init__(self, dataset):
        self.data = dataset
        self.conv_layers = []
        self.pool_layers = []
        self.dense_layers = []
        self.conv_weights = []
        self.conv_bias = []
        self.dense_weights = []
        self.dense_bias = []
        self.results_dir = os.path.join(settings.BRAIN_DIR, "cnn")
        if not os.path.isdir(self.results_dir):
            os.mkdir(self.results_dir)

    def train(self, display_step=100):
        n_classes = len(settings.GENRES)
        n_input = settings.CNN['INPUT_SHAPE'][0]*settings.CNN['INPUT_SHAPE'][1]
        x = self.x = tf.placeholder("float", [None, n_input])
        y = self.y = tf.placeholder("float", [None, n_classes])

        new_shape = [-1, settings.CNN['INPUT_SHAPE'][0],
            settings.CNN['INPUT_SHAPE'][1], 1]

        new_x = self.new_x = tf.reshape(x, new_shape)

        keep_prob = self.keep_prob = tf.placeholder(tf.float32)

        cw, dw = self.get_weights(n_input, n_classes)
        cb, db = self.get_bias(n_classes)
        y_ = self.y_ = self.prepare_layers()
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_, labels=self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=settings.NN['LEARNING_RATE']).minimize(cost)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        self.correct_pred = correct_pred = tf.equal(tf.argmax(self.y_, 1), tf.argmax(self.y, 1))
        self.accuracy = accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            sess.run(init)
            step = 1
            for i in range(settings.NN['TRAINING_CYCLES']):
                bx, by = self.data.train.next_batch(settings.NN['BATCH_SIZE'])
                req_shape = settings.CNN['INPUT_SHAPE'][0]*settings.CNN['INPUT_SHAPE'][1]
                bx = np.pad(bx, ((0, 0),(0, req_shape-bx.shape[1])), "reflect")
                sess.run(optimizer, feed_dict={x: bx, y:by, keep_prob:settings.NN['DROPOUT_PROB']})
                if step % display_step == 0:
                    # Calculate batch accuracy
                    acc = sess.run(accuracy, feed_dict={x: bx, y: by, keep_prob: 1.})
                    # Calculate batch loss
                    loss = sess.run(cost, feed_dict={x: bx, y: by, keep_prob: 1.})
                    print("Iter " + str(step * settings.NN['BATCH_SIZE']) + ", Minibatch Loss= " + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

                    save_path = saver.save(sess, os.path.join(self.results_dir, "model.ckpt"))
                    print("Model saved in file: %s" % save_path)
                step += 1
            save_path = saver.save(sess, os.path.join(self.results_dir, "model.final"))
            print("Model saved in file: %s" % save_path)

            req_shape = settings.CNN['INPUT_SHAPE'][0]*settings.CNN['INPUT_SHAPE'][1]
            tm = np.pad(self.data.test.music, ((0,0),(0, req_shape-bx.shape[1])), "reflect")
            print("Testing Accuracy:", sess.run(self.accuracy, 
                feed_dict={x: tm, y: self.data.test.labels, keep_prob: 1.}))


    def prepare_layers(self):
        prev_layer = self.new_x
        cw = self.conv_weights
        cb = self.conv_bias
        dw = self.dense_weights
        db = self.dense_bias
        print(dw, db)
        shape = np.array(settings.CNN['INPUT_SHAPE'])
        for i in range(settings.CNN['NUM_HIDDEN_LAYERS']):
            layer = tf.add(conv2d(prev_layer, cw[i]), cb[i])
            layer = tf.nn.relu(layer)
            self.conv_layers.append(layer)
            pool = max_pool_2x2(layer)
            self.pool_layers.append(pool)
            prev_layer = pool
            shape[0] = int(np.ceil(shape[0]/2))
            shape[1] =  int(np.ceil(shape[1]/2))

        prev_layer = tf.reshape(self.pool_layers[-1], 
            [-1, settings.CNN['HIDDEN_FEATURES'][-1]*shape[0]*shape[1]])

        print(prev_layer)
        for i in range(settings.CNN['NUM_DENSE_LAYERS']):
            layer = tf.add(tf.matmul(prev_layer, dw[i]), db[i])
            layer = tf.nn.relu(layer)
            self.dense_layers.append(layer)
            prev_layer = layer

        if len(self.dense_layers) > 0:
            drop_layer = tf.nn.dropout(self.dense_layers[-1], self.keep_prob)
        else:
            drop_layer = tf.nn.dropout(prev_layer, self.keep_prob)
        self.dense_layers.append(drop_layer)
        out_layer = tf.matmul(self.dense_layers[-1], dw[-1]) + db[-1]
        self.dense_layers.append(out_layer)
        return out_layer

    def get_weights(self, inp, out):
        prev_inp_channel = 1
        num_channels = settings.CNN['HIDDEN_FEATURES']
        shape = np.array(settings.CNN['INPUT_SHAPE'])
        patch = np.array(settings.CNN['PATCH_SIZE'])
        for i in range(settings.CNN['NUM_HIDDEN_LAYERS']):
            next_out = num_channels[i]
            w = weight_variable([patch[0], patch[1], prev_inp_channel, next_out])
            prev_inp_channel = next_out
            self.conv_weights.append(w)
            shape[0] = int(np.ceil(shape[0]/2))
            shape[1] =  int(np.ceil(shape[1]/2))

        prev_inputs = settings.CNN['HIDDEN_FEATURES'][-1]*shape[0]*shape[1]
        num_inputs = settings.CNN['DENSE_INPUTS']
        num_inputs.append(out)
        for i in range(settings.CNN['NUM_DENSE_LAYERS']+1):
            w = weight_variable([prev_inputs, num_inputs[i]])
            self.dense_weights.append(w)
            prev_inputs = num_inputs[i]

        return self.conv_weights, self.dense_weights

    def get_bias(self, out):
        for i in range(settings.CNN['NUM_HIDDEN_LAYERS']):
            b = tf.Variable(tf.random_normal([settings.CNN['HIDDEN_FEATURES'][i]]))
            self.conv_bias.append(b)

        for i in range(settings.CNN['NUM_DENSE_LAYERS']):
            b = tf.Variable(tf.random_normal([settings.CNN['DENSE_INPUTS'][i]]))
            self.dense_bias.append(b)

        b = tf.Variable(tf.random_normal([out]))
        self.dense_bias.append(b)
        return self.conv_bias, self.dense_bias

    def eval(self):
        print("Testing Accuracy:", self.sess.run(self.accuracy, 
            feed_dict={x: self.data.test.music, y: self.data.test.labels, keep_prob: 1.}))

def weight_variable(shape):
    print(shape)
    if settings.CNN['RANDOM']:
        initial = tf.random_normal(shape)
    else:
        initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=settings.CNN['STRIDES'], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')