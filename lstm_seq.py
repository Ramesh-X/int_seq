import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Tuple, List
import matplotlib.pyplot as plt
import sys

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train_file', './data/train.csv', 'Input training file')
flags.DEFINE_string('test_file', './data/test.csv', 'Testing file')
flags.DEFINE_string('save_file', './data/test_result', 'Test Results file')
flags.DEFINE_string('backup', './models/lstm_seq_model.ckpt', 'Directory for storing data')

dtype = tf.float32
BATCH_SIZE = 1000
max_len = 200       # maximum length of an integer sequence
max_bit = 512       # maximum number of bits using to represent an integer
max_int = 2**(max_bit - 1) - 1    # maximum number that will allow in the integer sequence
num_layers = 1      # Number of layers in the LSTM
hidden_size = 1024  # Hidden size (Maximum time steps) of LSTM
invalid_int = None
invalid_seq = None


def generate_invalid():
    global invalid_seq, invalid_int
    invalid_int = convert_binary(max_int+1)
    invalid_seq = np.full((3, max_bit), invalid_int, dtype=np.float32)


def weight_variable(shape, name):
    # return tf.get_variable(name, shape=shape, dtype=dtype, initializer=tf.contrib.layers.xavier_initializer())
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial, name=name, dtype=tf.float32)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, dtype=dtype)
    return tf.Variable(initial, name=name, dtype=dtype)


'''
def read_data():    # -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = pd.read_csv(FLAGS.train_file)
    train = train.drop("Id", axis=1).values
    trn_x = []
    trn_y = []
    for i in range(len(train)):
        seq = get_binary_seq([int(x) for x in train[i][0].split(',')])
        if seq is invalid_seq:
            continue
        trn_x.append(seq[:-1])
        trn_y.append(seq[-1])
    splt = int(len(trn_x) * 0.2)
    validate_x = trn_x[:splt]
    validate_y = trn_y[:splt]
    trn_x = trn_x[splt:]
    trn_y = trn_y[splt:]
    test = pd.read_csv(FLAGS.train_file)
    ids = test['Id'].values
    test = test.drop("Id", axis=1).values
    test_x = []
    for i in range(len(test)):
        seq = [int(x) for x in train[i][0].split(',')]
        test_x.append(get_binary_seq(seq))
    return trn_x, trn_y, validate_x, validate_y, test_x, ids'''


def read_data() -> Tuple[List[List[int]], List[int], List[List[int]], List[int], List[List[int]], np.ndarray]:
    train = pd.read_csv(FLAGS.train_file)
    train = train.drop("Id", axis=1).values
    trn_x = []
    trn_y = []
    for i in range(len(train)):
        seq = [int(x) for x in train[i][0].split(',')]
        if len(seq) < 2:
            continue
        trn_x.append(seq[:-1])
        trn_y.append(seq[-1])
    splt = int(len(trn_x) * 0.2)
    validate_x = trn_x[:splt]
    validate_y = trn_y[:splt]
    trn_x = trn_x[splt:]
    trn_y = trn_y[splt:]
    test = pd.read_csv(FLAGS.test_file)
    ids = test['Id'].values
    test = test.drop("Id", axis=1).values
    test_x = []
    for i in range(len(test)):
        seq = [int(x) for x in test[i][0].split(',')]
        test_x.append(seq)
    return trn_x, trn_y, validate_x, validate_y, test_x, ids


def get_binary_seq(seq):
    l = len(seq)
    if l > max_len:
        return invalid_seq
    out_seq = []
    for x in seq:
        bin_x = convert_binary(x)
        if bin_x is invalid_int:
            return invalid_seq
        out_seq.append(bin_x)
    return np.array(out_seq)


def convert_binary(num: int):
    neg = False
    if num < 0:
        neg = True
        num *= -1
    if num > max_int + 1:
        return invalid_int
    out = np.zeros(max_bit, np.float32)
    i = 0
    while num:
        out[i] = num & 1
        num >>= 1
        i += 1
    if neg:
        out[max_bit - 1] = 1
    return out


def convert_decimal(num):
    out = 0
    for i in num[-2::-1]:
        out = (out << 1) | int(i)
    if num[-1]:
        out *= -1
    return out


def invalid_seq_count(seq_list):
    num = 0
    for seq in seq_list:
        if len(seq) > max_len:
            num += 1
        else:
            for x in seq:
                if abs(x) > max_int:
                    num += 1
                    break
    return num


def batch_generator(batch_size, x, y, p=None):
    if p is None:
        p = range(len(x))
    batch_x = []
    batch_y = []
    for i in p:
        x_i = get_binary_seq(x[i])
        y_i = convert_binary(y[i])
        if x_i is invalid_seq or y_i is invalid_int:
            continue
        batch_x.append(x_i)
        batch_y.append(y_i)
        if len(batch_x) == batch_size:
            ls = [len(seq) for seq in batch_x]
            l_max = np.max(ls)
            for i in range(batch_size):
                batch_x[i] = np.concatenate([np.full((l_max - ls[i], max_bit), invalid_int, dtype=np.float32), batch_x[i]])
            yield np.reshape(batch_x, [batch_size, l_max, max_bit]), np.reshape(batch_y, [batch_size, max_bit])
            batch_x = []
            batch_y = []


def test_batch_generator(batch_size, x):
    batch_x = []
    for i in range(len(x)):
        x_i = get_binary_seq(x[i])
        batch_x.append(x_i)
        if len(batch_x) == batch_size:
            ls = [len(seq) for seq in batch_x]
            l_max = np.max(ls)
            for i in range(batch_size):
                batch_x[i] = np.concatenate([np.full((l_max - ls[i], max_bit), invalid_int, dtype=np.float32), batch_x[i]])
            yield np.reshape(batch_x, [batch_size, l_max, max_bit])
            batch_x = []
    l = len(batch_x)
    if l > 0:
        for i in range(batch_size-l):
            batch_x.append(invalid_seq)
        ls = [len(seq) for seq in batch_x]
        l_max = np.max(ls)
        for i in range(batch_size):
            batch_x[i] = np.concatenate([np.full((l_max - ls[i], max_bit), invalid_int, dtype=np.float32), batch_x[i]])
        yield np.reshape(batch_x, [batch_size, l_max, max_bit])


def main():
    print("Reading Data..")
    generate_invalid()
    train_x, train_y, validate_x, validate_y, test_x, ids = read_data()
    print("Train set:\n\tX: %i\n\tY: %i\n\nValidation set:\n\tX: %i\n\tY: %i\n\nTest set: %i" % (len(train_x), len(train_y), len(validate_x), len(validate_y), len(test_x)))
    print("Creating the model")
    X = tf.placeholder(dtype, [None, None, max_bit])    # batch_size, max_time, max_bit
    # x_ = tf.reshape(X, [1, -1, max_bit])
    Y = tf.placeholder(dtype, [None, max_bit])
    # y_ = tf.reshape(Y, [1, max_bit])
    tr = tf.placeholder(dtype)
    pr = tf.placeholder(dtype)

    cell = tf.contrib.rnn.LSTMBlockCell(hidden_size, forget_bias=0.0)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=pr)
    cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], state_is_tuple=True)
    val, _ = tf.nn.dynamic_rnn(cell, X, dtype=dtype)
    val = tf.reshape(val[:, -1, :], [-1, hidden_size])
    w = weight_variable([hidden_size, max_bit], 'final_w')
    b = bias_variable([max_bit], 'final_b')
    val = tf.matmul(val, w) + b
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=val, labels=Y))
    train_step = tf.train.AdamOptimizer(tr).minimize(cross_entropy)

    out_y = tf.cast(tf.sigmoid(val) >= 0.5, dtype)
    correct_prediction = tf.equal(out_y, Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype))

    print("Initializing variables..")
    tr_tmp = 1.00e-4
    acc = 0.1
    pre_acc = 0
    epoch = 0
    saver = tf.train.Saver()
    data_size = len(train_x)
    test_size = len(ids)

    def save_file(name: str):
        lasts = []
        batch_gen = test_batch_generator(1200, test_x)
        for tst_x in batch_gen:
            tst_y = sess.run(out_y, feed_dict={X: tst_x, pr: 1.0})
            [lasts.append(str(convert_decimal(i))) for i in tst_y]
        df = pd.Series(lasts[:test_size], index=ids)
        df.to_csv(FLAGS.save_file + name, header=['Last'], index_label='Id')
        print('file saved: ', name)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        use_previous = 0
        # use the previous model or don't and initialize variables
        if use_previous:
            saver.restore(sess, './models/mnist_model.ckpt-5.data-00000-of-00001')
            print("Model restored.")

        print("Training...")
        sys.stdout.flush()
        while acc < 0.99999:
            epoch += 1
            # Shuffle the data before each training iteration.
            p = np.random.permutation(range(data_size))
            splt = int(data_size * 0.8)
            batch_gen = batch_generator(BATCH_SIZE, train_x, train_y, p[:splt])
            for trX, trY in batch_gen:
                sess.run(train_step, feed_dict={X: trX, Y: trY, tr: tr_tmp, pr: 0.45})

            pre_acc = acc
            acc = []
            batch_gen = batch_generator(BATCH_SIZE, train_x, train_y, p[splt:])
            for trX, trY in batch_gen:
                acc.append(sess.run(accuracy, feed_dict={X: trX, Y: trY, pr: 1.0}))
            acc = np.mean(acc)
            print('epoch: ', epoch, ', Accuracy: ', acc, ', learning_rate: ', tr_tmp)
            if acc >= pre_acc:
                # tr_tmp *= 0.95
                saver.save(sess, FLAGS.backup, global_step=epoch)
            else:
                pass
                # tr_tmp *= 1.06
            # tr_tmp = tr_tmp / (1 + 0.009*epoch) if acc > 0.95 else 1.0e-4 if acc < 0.8 else tr_tmp / (1 + 0.02*epoch)
            if epoch % 5 == 0:
                acc = []
                batch_gen = batch_generator(BATCH_SIZE, validate_x, validate_y)
                for tx, ty in batch_gen:
                    acc = sess.run(accuracy, feed_dict={X: tx, Y: ty, pr: 1.0})
                acc = np.mean(acc)
                print('epoch: ', epoch, ', Test Accuracy: ', acc, ' : Testing on test set finished >>>>>>>>>>>')
                if acc > 0.97:
                    save_file(str(epoch) + '_a' + str(acc))
            sys.stdout.flush()

        print('\n\n\nTraining finished..!')
        save_file('final')

if __name__ == "__main__":
    main()
