import tensorflow as tf
import os
from gmc.conf import settings
from gmc.core.cache import store

class NN:
    def __init__(self, dataset, n_input=None):
        self.data = dataset
        self.layers = []
        self.weights = []
        self.bias = []
        self.results_dir = os.path.join(settings.BRAIN_DIR, "nn")
        if n_input is None:
            n_input = 0
            for f in settings.FEATURES:
                n_input += settings.FEATURES_LENGTH[f]

        self.n_input = n_input
        if not os.path.isdir(self.results_dir):
            os.mkdir(self.results_dir)

    def train(self, display_step=100, out=False, path='model.final'):
        n_classes = len(settings.GENRES)
        n_input = self.n_input
        x = self.x = tf.placeholder("float", [None, n_input], name='x')
        y = self.y = tf.placeholder("float", [None, n_classes], name='y')
        keep_prob = self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        w = self.get_weights(n_input, n_classes)
        b = self.get_bias(n_classes)
        y_ = self.y_ = self.prepare_layers()
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_, labels=self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=settings.NN['LEARNING_RATE']).minimize(cost)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        self.correct_pred = correct_pred = tf.equal(tf.argmax(self.y_, 1), tf.argmax(self.y, 1))
        self.accuracy = accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        train_acc = []
        #val_acc = []
        test_acc = []
        train_cost = []
        test_cost = []
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            sess.run(init)
            step = 1
            for i in range(settings.NN['TRAINING_CYCLES']):
                bx, by = self.data.train.next_batch(settings.NN['BATCH_SIZE'])
                sess.run(optimizer, feed_dict={x: bx, y:by, keep_prob:settings.NN['DROPOUT_PROB']})
                if step % display_step == 0:
                    # Calculate batch accuracy
                    acc = sess.run(accuracy, feed_dict={x: bx, 
                        y: by, keep_prob: 1.})
                    # Calculate batch loss
                    loss = sess.run(cost, feed_dict={x: bx, 
                        y: by, keep_prob: 1.})

                    train_acc.append(acc)

                    #vac = sess.run(accuracy, 
                    #    feed_dict={x: self.data.validation.music, y: self.data.validation.labels, keep_prob: 1.})
                    tac = sess.run(accuracy, 
                        feed_dict={x: self.data.test.music, y: self.data.test.labels, keep_prob: 1.})

                    te_loss = sess.run(cost, 
                        feed_dict={x: self.data.test.music, y: self.data.test.labels, keep_prob: 1.})

                    train_cost.append(loss)
                    test_cost.append(te_loss)
                    #val_acc.append(vac)
                    test_acc.append(tac)
                    save_path = saver.save(sess, os.path.join(self.results_dir, "model.ckpt"))
                    if out:
                        print("Iter " + str(step * settings.NN['BATCH_SIZE']) + ", Minibatch Loss= " + \
                              "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

                        #print("Validation Accuracy:", vac)
                        print("Testing Accuracy:", tac)
                        print("Model saved in file: %s" % save_path)
                step += 1
            save_path = saver.save(sess, os.path.join(self.results_dir, path))
            storage = store(self.results_dir)
            storage['save.path'] = save_path
            print("Model saved in file: %s" % save_path)

            print("Testing Accuracy:", sess.run(self.accuracy, 
                feed_dict={x: self.data.test.music, y: self.data.test.labels, keep_prob: 1.}))

        return train_acc, test_acc, train_cost, test_cost

    def prepare_layers(self):
        prev_layer = self.x
        w = self.weights
        b = self.bias
        for i in range(settings.NN['NUM_HIDDEN_LAYERS']):
            layer = tf.add(tf.matmul(prev_layer, w[i]), b[i])
            layer = tf.nn.relu(layer)
            self.layers.append(layer)
            prev_layer = self.layers[-1]

        drop_layer = tf.nn.dropout(self.layers[-1], self.keep_prob)
        self.layers.append(drop_layer)
        out_layer = tf.add(tf.matmul(self.layers[-1], w[-1]), b[-1], name='y_')
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
