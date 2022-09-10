import numpy as np
import cv2 as cv
import os
# import tensorflow as tf
# import tensorflow.compat.v1 as tf
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"




def normalize_data(data):
    return (data - data.mean()) / data.max()

def load_image(image_path, width, height):
    gray_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    resized_image = cv.resize(gray_image, (width, height))
    normalized_image = normalize_data(resized_image)
    data = []
    data.append(normalized_image.ravel())
    return np.array(data)

### 模型定义开始
IMAGE_WIDTH = 24
IMAGE_HEIGHT = 48
CLASSIFICATION_COUNT = 31
ENGLISH_LABELS = ['chuan', 'e', 'gan', 'gan1', 'gui', 'gui1', 'hei', 'hu', 'ji', 'jin',
'jing', 'jl', 'liao', 'lu', 'meng', 'min', 'ning', 'qing', 'qiong', 'shan',
'su', 'sx', 'wan', 'xiang', 'xin', 'yu', 'yu1', 'yue', 'yun', 'zang',
'zhe']

def onehot_labels(labels):
    onehots = np.zeros((len(labels), CLASSIFICATION_COUNT))
    for i in np.arange(len(labels)):
        onehots[i, labels[i]] = 1
    return onehots

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # padding='SAME',使卷积输出的尺寸=ceil(输入尺寸/stride)，必要时自动padding
    # padding='VALID',不会自动padding，对于输入图像右边和下边多余的元素，直接丢弃
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
g2 = tf.Graph()
sess2 = tf.Session(graph=g2)
with sess2.as_default():
    with sess2.graph.as_default():
        x = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT * IMAGE_WIDTH])
        y_ = tf.placeholder(tf.float32, shape=[None, CLASSIFICATION_COUNT])
        x_image = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

        W_conv1 = weight_variable([5, 5, 1, 32])                       # color channel == 1; 32 filters
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)       # 20x20
        h_pool1 = max_pool_2x2(h_conv1)                                # 20x20 => 10x10

        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)        # 10x10
        h_pool2 = max_pool_2x2(h_conv2)                                 # 10x10 => 5x5

        # 全连接神经网络的第一个隐藏层
        # 池化层输出的元素总数为：5(H)*5(W)*64(filters)
        W_fc1 = weight_variable([6 * 12 * 64, 1024])                     # 全连接第一个隐藏层神经元1024个
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 6 * 12 * 64])            # 转成1列
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)      # Affine+ReLU

        keep_prob = tf.placeholder(tf.float32)                          # 定义Dropout的比例
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)                    # 执行dropout

        # 全连接神经网络输出层
        W_fc2 = weight_variable([1024, CLASSIFICATION_COUNT])                             # 全连接输出为10个
        b_fc2 = bias_variable([CLASSIFICATION_COUNT])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        ### 模型定义结束

        ENGLISH_MODEL_PATH = "model/cnn_chs/chs.ckpt"

        saver = tf.train.Saver()
        saver.restore(sess2, ENGLISH_MODEL_PATH)
        def get_chs(path):
            digit_image_path = path
            digit_image = load_image(digit_image_path, IMAGE_WIDTH, IMAGE_HEIGHT)
            results = sess2.run(y_conv, feed_dict={x: digit_image, keep_prob: 1.0})
            predict = np.argmax(results[0])
            if ENGLISH_LABELS[predict]=='chuan':
                return '川'
            if ENGLISH_LABELS[predict]=='e':
                return '鄂'
            if ENGLISH_LABELS[predict]=='gan':
                return '赣'
            if ENGLISH_LABELS[predict]=='gan1':
                return '甘'
            if ENGLISH_LABELS[predict]=='gui':
                return '贵'
            if ENGLISH_LABELS[predict]=='gui1':
                return '桂'
            if ENGLISH_LABELS[predict]=='hei':
                return '黑'
            if ENGLISH_LABELS[predict]=='hu':
                return '沪'
            if ENGLISH_LABELS[predict]=='ji':
                return '冀'
            if ENGLISH_LABELS[predict]=='jin':
                return '津'
            if ENGLISH_LABELS[predict]=='jing':
                return '京'
            if ENGLISH_LABELS[predict]=='jl':
                return '吉'
            if ENGLISH_LABELS[predict]=='liao':
                return '辽'
            if ENGLISH_LABELS[predict]=='lu':
                return '鲁'
            if ENGLISH_LABELS[predict]=='meng':
                return '蒙'
            if ENGLISH_LABELS[predict]=='min':
                return '闽'
            if ENGLISH_LABELS[predict]=='ning':
                return '宁'
            if ENGLISH_LABELS[predict]=='qing':
                return '青'
            if ENGLISH_LABELS[predict]=='qiong':
                return '琼'
            if ENGLISH_LABELS[predict]=='shan':
                return '陕'
            if ENGLISH_LABELS[predict]=='su':
                return '苏'
            if ENGLISH_LABELS[predict]=='sx':
                return '晋'
            if ENGLISH_LABELS[predict]=='wan':
                return '皖'
            if ENGLISH_LABELS[predict]=='xiang':
                return '湘'
            if ENGLISH_LABELS[predict]=='xin':
                return '新'
            if ENGLISH_LABELS[predict]=='yu':
                return '豫'
            if ENGLISH_LABELS[predict]=='yu1':
                return '渝'
            if ENGLISH_LABELS[predict]=='yue':
                return '粤'
            if ENGLISH_LABELS[predict]=='yun':
                return '云'
            if ENGLISH_LABELS[predict]=='zang':
                return '藏'
            if ENGLISH_LABELS[predict]=='zhe':
                return '浙'
