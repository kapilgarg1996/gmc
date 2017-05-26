import tensorflow as tf
import os
from gmc.conf import settings

class NN:
    def __init__(self, dataset):
        self.data = dataset
        self.layers = []
        self.weights = []
        self.bias = []
        self.results_dir = os.path.join(settings.BRAIN_DIR, "nn")
        if not os.path.isdir(self.results_dir):
            os.mkdir(self.results_dir)

    def train(self, display_step=100):
        n_classes = len(settings.GENRES)
        n_input = 0
        for f in settings.FEATURES:
            n_input += settings.FEATURES_LENGTH[f]
        x = self.x = tf.placeholder("float", [None, n_input])
        y = self.y = tf.placeholder("float", [None, n_classes])
        keep_prob = self.keep_prob = tf.placeholder(tf.float32)

        w = self.get_weights(n_input, n_classes)
        b = self.get_bias(n_classes)
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

            print("Testing Accuracy:", sess.run(self.accuracy, 
                feed_dict={x: self.data.test.music, y: self.data.test.labels, keep_prob: 1.}))


    def prepare_layers(self):
        prev_layer = self.x
        w = self.weights
        b = self.bias
        for i in range(settings.NN['NUM_HIDDEN_LAYERS']):
            layer = tf.add(tf.matmul(prev_layer, w[i]), b[i])
            layer = tf.nn.relu(layer)
            self.layers.append(layer)
            prev_layer = layer

        drop_layer = tf.nn.dropout(self.layers[-1], self.keep_prob)
        self.layers.append(drop_layer)
        out_layer = tf.matmul(self.layers[-1], w[-1]) + b[-1]
        self.layers.append(out_layer)
        return out_layer

    def get_weights(self, inp, out):
        prev_layer_out = inp
        num_weights = settings.NN['HIDDEN_INPUTS']
        num_weights.append(out)
        for i in range(settings.NN['NUM_HIDDEN_LAYERS']+1):
            next_out = num_weights[i]
            if settings.NN['RANDOM']:
                w = tf.Variable(tf.random_normal([prev_layer_out, next_out]))
            else:
                w = tf.Variable(tf.truncated_normal([prev_layer_out, next_out], stddev=0.1))
            prev_layer_out = next_out
            self.weights.append(w)
        return self.weights

    def get_bias(self, out):
        for i in range(settings.NN['NUM_HIDDEN_LAYERS']):
            b = tf.Variable(tf.random_normal([settings.NN['HIDDEN_INPUTS'][i]]))
            self.bias.append(b)

        b = tf.Variable(tf.random_normal([out]))
        self.bias.append(b)
        return self.bias

    def eval(self):
        print("Testing Accuracy:", self.sess.run(self.accuracy, 
            feed_dict={x: self.data.test.music, y: self.data.test.labels, keep_prob: 1.}))
