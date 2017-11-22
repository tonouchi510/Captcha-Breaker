import tensorflow as tf
import TFRecord

class Model:
    def __init__(self, mode):
        with tf.Graph().as_default():
            self.setup_model(mode)
            self.setup_session(mode)

    def setup_model(self, mode):
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, shape=[None, 100, 100, 1], name='input')  # input (第1引数のNoneは実際にデータ数に対応)

        # convolution
        with tf.name_scope('conv1'):
            W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1), name='W_conv1')  # 縦、横、チャネル数、フィルターの数
            b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]), name='b_conv1')
            h_conv1 = tf.nn.relu(self.conv2d(x, W_conv1) + b_conv1)

        with tf.name_scope('pool1'):
            h_pool1 = self.max_pool_2x2(h_conv1)

        with tf.name_scope('conv2'):
            W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1), name='W_conv2')  # 標準偏差0.1のガウス乱数分布
            b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]), name='b_conv2')
            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        with tf.name_scope('pool2'):
            h_pool2 = self.max_pool_2x2(h_conv2)

        with tf.name_scope('conv3'):
            W_conv3 = tf.Variable(tf.truncated_normal([5, 5, 64, 128], stddev=0.1), name='W_conv3')
            b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]), name='b_conv3')
            h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3) + b_conv3)
        with tf.name_scope('pool3'):
            h_pool3 = self.max_pool_2x2(h_conv3)

        with tf.name_scope('conv4'):
            W_conv4 = tf.Variable(tf.truncated_normal([5, 5, 128, 256], stddev=0.1), name='W_conv4')
            b_conv4 = tf.Variable(tf.constant(0.1, shape=[256]), name='b_conv4')
            h_conv4 = tf.nn.relu(self.conv2d(h_pool3, W_conv4) + b_conv4)
        with tf.name_scope('pool4'):
            h_pool4 = self.max_pool_2x2(h_conv4)

        # full connected
        # 今回畳み込みでpadding = 'SAME'を指定しているため、プーリングでのみ画像サイズが変わる。
        with tf.name_scope('fc1'):
            h_pool4_flat = tf.reshape(h_pool4, [-1, 7 * 7 * 256])
            W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 256, 2048], stddev=0.1), name='W_fc1')
            b_fc1 = tf.Variable(tf.constant(0.1, shape=[2048]), name='b_fc1')
            h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

        # dropout
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32, name='rate')
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # output
        with tf.name_scope('output'):
            W_fc2 = tf.Variable(tf.truncated_normal([2048, 26], stddev=0.1), name='W_output')
            b_fc2 = tf.Variable(tf.constant(0.1, shape=[26]), name='b_output')
            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # 学習モードの時だけ最適化、評価、グラフ生成
        if mode == 'train':
            with tf.name_scope('optimizer'):
                y_ = tf.placeholder(tf.float32, shape=[None, 26], name='labels')  # output
                cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_), name='loss')
                train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)

            with tf.name_scope('evaluator'):
                correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

            # 折れ線グラフ
            tf.summary.scalar("Loss", cross_entropy)
            tf.summary.scalar("Train Accuracy", accuracy)
            test_sum = tf.summary.merge([tf.summary.scalar("Test Accuracy", accuracy, collections=[])])
            # ヒストグラム
            tf.summary.histogram("weights_conv1", W_conv1)
            tf.summary.histogram("biases_conv1", b_conv1)
            tf.summary.histogram("weights_conv2", W_conv2)
            tf.summary.histogram("biases_conv2", b_conv2)
            tf.summary.histogram("weights_conv3", W_conv3)
            tf.summary.histogram("biases_conv3", b_conv3)
            tf.summary.histogram("weights_conv4", W_conv4)
            tf.summary.histogram("biases_conv4", b_conv4)
            tf.summary.histogram("weights_fc1", W_fc1)
            tf.summary.histogram("biases_fc1", b_fc1)
            tf.summary.histogram("weights_output", W_fc2)
            tf.summary.histogram("biases_output", b_fc2)

            # クラス外部から参照する必要のある変数をインスタンス変数として公開
            self.train_step = train_step
            self.cross_entropy = cross_entropy
            self.accuracy = accuracy
            self.test_sum = test_sum
            self.y_ = y_
        self.x, self.y_conv = x, y_conv
        self.keep_prob = keep_prob

    def setup_session(self, mode):
        sess = tf.Session()
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('./models/model/')
        if ckpt:  # checkpointがある場合
            last_model = ckpt.model_checkpoint_path  # 最後に保存したmodelへのパス
            print("load " + last_model)
            saver.restore(sess, last_model)  # 変数データの読み込み
        else:  # 保存データがない場合
            sess.run(tf.global_variables_initializer())  # 変数を初期化して実行
        self.sess, self.saver = sess, saver
        # 学習モードの時だけグラフ生成
        if mode == 'train':
            summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter("./logs/log", sess.graph)
            self.summary = summary
            self.writer = writer

    @staticmethod
    def inference(x):
        y = cnn.sess.run(cnn.y_conv, feed_dict={cnn.x:x, cnn.keep_prob:1.0})
        return y

    @staticmethod
    def conv2d(x, W):
        # stridesは[1, dy, dx, 1]で両サイドの1は固定、paddingはSAMEにすることで入力と同サイズの出力ができるようにする
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name='filter')

    @staticmethod
    def max_pool_2x2(x):
        # プーリング層は、ウィンドウサイズとストライドの大きさは揃える
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def inference(x):
    x_image = tf.reshape(x, shape=[1, 100, 100, 1])
    cnn = Model('test')
    output = cnn.sess.run([cnn.y_conv], feed_dict={cnn.x: x_image, cnn.keep_prob: 1.0})
    max = tf.arg_max(output, 1)
    print(max)
    y = max + 65
    return y


if __name__ == '__main__':
    dset1, dset2 = TFRecord.Reader('./dataset.tfrecords')
    tset1, tset2 = TFRecord.Reader('./testset.tfrecords')
    cnn = Model('train')
    # ピクセル値が0 ~ 255の範囲の値を取ってしまっているので、0 ~ 1の範囲の値になるように調整
    x1 = tf.cast(dset1, tf.float32) * (1. / 255)
    y1 = tf.cast(dset2, dtype=tf.float32)
    x2 = tf.cast(tset1, tf.float32) * (1. / 255)
    y2 = tf.cast(tset2, dtype=tf.float32)

    # ミニバッチ単位で取り出せるようにする
    batch_size=100
    x_batch, y_batch = tf.train.batch(
        [x1, y1], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size)

    # こっちはテストデータ用
    tx_batch, ty_batch = tf.train.batch(
        [x2, y2], batch_size=1000, num_threads=2,
        capacity=2 * 1000)

    # あとはsession張ってsess.run()すればミニバッチ単位でデータを取り出せる
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        # テストセットのデータを取り出す
        x_test, y_test = sess.run([tx_batch, ty_batch])
        # 入力画像の一例としてtensorboardに追加(Model内でやらないのはmargeしたくないため)
        cnn.writer.add_summary(sess.run(tf.summary.image("input", x_test, 10)))

        # 学習ループ
        for epoch in range(2000):
            # ミニバッチ分のデータを取り出す
            x_train, y_train = sess.run([x_batch, y_batch])
            if epoch % 100 == 0:
                # tensorboardへのグラフ書き込み
                summary, loss, train_acc = cnn.sess.run(
                    [cnn.summary, cnn.cross_entropy, cnn.accuracy],
                    feed_dict={cnn.x: x_train, cnn.y_: y_train, cnn.keep_prob: 1.0})
                cnn.writer.add_summary(summary, epoch)
                # test summary
                test_sum, test_acc = cnn.sess.run(
                    [cnn.test_sum, cnn.accuracy], feed_dict={cnn.x: x_test, cnn.y_: y_test, cnn.keep_prob: 1.0})
                cnn.writer.add_summary(test_sum, epoch)

                print("epoch: %d, Loss: %f, Train_Acc: %f, Test_Acc: %f" % (epoch, loss, train_acc, test_acc))
                # モデルの保存
                cnn.saver.save(cnn.sess, './models/model/my_model', global_step=epoch, write_meta_graph=False)
            cnn.sess.run(cnn.train_step, feed_dict={cnn.x: x_train, cnn.y_: y_train, cnn.keep_prob: 0.5})  # 学習
    finally:
        coord.request_stop()
        coord.join(threads)

    cnn.sess.close()
    sess.close()

