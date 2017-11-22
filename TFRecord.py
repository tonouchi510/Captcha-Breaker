import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image


def Writer(file_path,  data, target):
    # 作成するデータセットのファイルパスを指定
    dataset_path = file_path  # 例:dataset.tfrecords

    # 格納する画像サイズの指定
    width, height = [100, 100]
    # 画像の色階調(RGB, 2値など)
    depth = 1

    # クラス数(分類したい種類の数): ラベルをone-hot表現に変換する為に利用
    # one-hot表現については https://ja.wikipedia.org/wiki/One-hot を参照
    class_count = 26

    # 正解データ: 画像ファイル名と正解ラベルのリスト
    # [[画像ファイル名, 正解ラベル], ...]
    datas = []
    n_sample = len(data)
    for i in range(n_sample):
        datas.append([data[i], target[i]])

    # TFRecordsファイルに書き出す為、TFRecordWriterオブジェクトを生成
    writer = tf.python_io.TFRecordWriter(dataset_path)

    # datasから、画像ファイル名と正解ラベルの対を1件ずつ取り出す
    for img_name, label in datas:
        # 画像ファイルを読み込み、リサイズ ＆ バイト文字列に変換
        img_obj = Image.open(img_name).convert("1").resize((width, height))
        img = np.array(img_obj).tostring()

        # 画像ファイル1件につき、1つのtf.train.Exampleを作成
        record = tf.train.Example(features=tf.train.Features(feature={
            "class_count": tf.train.Feature(int64_list=tf.train.Int64List(value=[class_count])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
            "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
            "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
            "depth": tf.train.Feature(int64_list=tf.train.Int64List(value=[depth])),
        }))

        # tf.train.ExampleをTFRecordsファイルに書き込む
        writer.write(record.SerializeToString())

    writer.close()

def Reader(file_path):
    # 読み込み対象のファイルをqueueに詰める: TFRecordReaderはqueueを利用してファイルを読み込む
    file_name_queue = tf.train.string_input_producer([file_path])

    # TFRecordsファイルを読み込む為、TFRecordReaderオブジェクトを生成
    reader = tf.TFRecordReader()

    # 読み込み: ファイルから読み込み、serialized_exampleに格納する
    _, serialized_example = reader.read(file_name_queue)

    # デシリアライズ: serialized_exampleはシリアライズされているので、デシリアライズする
    #                 → Tensorオブジェクトが返却される
    features = tf.parse_single_example(
        serialized_example,
        features={
            "class_count": tf.FixedLenFeature([], tf.int64),
            "label": tf.FixedLenFeature([], tf.int64),
            "image": tf.FixedLenFeature([], tf.string),
            "height": tf.FixedLenFeature([], tf.int64),
            "width": tf.FixedLenFeature([], tf.int64),
            "depth": tf.FixedLenFeature([], tf.int64),
        })

    # featuresオブジェクト内の要素はTensorオブジェクトとなっている
    # でも、Tensorオブジェクトに直接アクセスしても中身が見えない
    #
    # → 中身を見る為には、session張ってeval()する
    # → eval()する為にはCoordinatorオブジェクトを生成して、start_queue_runner()しておく必要がある
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            # --- 実数値に変換する必要があるもののみeval() ---
            height = tf.cast(features["height"], tf.int32).eval()
            width = tf.cast(features["width"], tf.int32).eval()
            depth = tf.cast(features["depth"], tf.int32).eval()
            class_count = tf.cast(features["class_count"], tf.int32).eval()

            # --- 画像データとラベルは学習時に適宜取り出したいのでeval()しない ---
            label = tf.cast(features["label"], tf.int32)
            # バイト文字列をdecodeし、元のshapeに戻す
            img = tf.reshape(tf.decode_raw(features["image"], tf.uint8),
                             tf.stack([height, width, depth]))
        finally:
            coord.request_stop()
            coord.join(threads)

    # labelをone-hot表現に変換
    label = tf.one_hot(label, class_count)

    return img, label

'''
# これは外部からは使わない(ミニバッチ処理のコード例として表示)
# Readerの結果のimgとlabelをミニバッチ単位で取り出す
def batchReader(img, label, batch):
    # ピクセル値が0 ~ 255の範囲の値を取ってしまっているので、0 ~ 1の範囲の値になるように調整
    img = tf.cast(img, tf.float32) * (1. / 255)
    label = tf.cast(label, dtype=tf.float32)

    # ミニバッチのサイズを指定
    batch_size = batch

    # ミニバッチ単位で取り出せるようにする
    # 詳細は https://www.tensorflow.org/api_docs/python/tf/train/batch
    images, sparse_labels = tf.train.batch(
        [img, label], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size)

    # あとはsession張ってsess.run()すればミニバッチ単位でデータを取り出せる
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        # ミニバッチ分のデータを取り出す
        # この中で学習処理なども行う
        imgs, labels = sess.run([images, sparse_labels])
    finally:
        coord.request_stop()
        coord.join(threads)

    sess.close()
    return imgs, labels
'''

def load_csv(filename):
    file = pd.read_csv(filename, header=0)

    #get header
    n_train = int(file.columns[0])
    n_test = int(file.columns[1])
    n_feature = int(file.columns[2])

    x_train = []
    y_train = []
    for i in range(n_train):
        x_train.append(file.index[i][1])
        y_train.append(ord(file.index[i][2]) - 65)

    x_test = []
    y_test = []
    for i in range(n_train, n_train + n_test):
        x_test.append(file.index[i][1])
        y_test.append(ord(file.index[i][2]) - 65)

    return x_train, y_train, x_test, y_test

'''
x_train, y_train, x_test, y_test = load_csv('./out/info.csv')
Writer('./dataset.tfrecords', x_train, y_train)
Writer('./testset.tfrecords', x_test, y_test)
'''
