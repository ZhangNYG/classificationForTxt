import random

import tensorflow as tf
import os.path
import hdfs
import numpy as np

HADOOP_IP_PORT = "http://10.1.0.42:50070"
HADOOP_PATH = "/user/cdh/guojie/Weibo_EmotionDataSet/"

all_STEP = 0


# tensorflow层
def add_layer(layername, inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    with tf.variable_scope(layername, reuse=None):
        Weights = tf.get_variable("weights", shape=[in_size, out_size],
                                  initializer=tf.truncated_normal_initializer(stddev=2))
        biases = tf.get_variable("biases", shape=[1, out_size],
                                 initializer=tf.truncated_normal_initializer(stddev=2))

    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def word2vector(line_words):
    list_words = line_words.split(',')[:-1]
    list_word_value = []
    for word in list_words:
        if word not in dictionary.keys():
            word = "UNK"
        list_word_value.append(dictionary[word])
    sum_vector = np.zeros(128)
    for i in range(len(list_word_value)):
        sum_vector += vectorall_words[list_word_value[i]]
    av_vector = sum_vector / len(list_word_value)
    # print(av_vector)
    return av_vector


def get_data(client, filename):
    global all_STEP
    global epoch_times

    with client.read(filename, encoding='utf-8') as f:
        counter = 0

        sum_loss = 0
        for line in f:
            line = line.strip('\n').strip('').strip('\r')
            data_tmp = []
            if line != "":
                counter += 1
                data_tmp = [word for word in line.split("\t") if word != '']

            if len(data_tmp) == 2:
                if len(data_tmp[-1]) <= 4:
                    result = word2vector(data_tmp[0])
                    result_np = np.array(result, dtype=np.float32)
                    if len(result_np) != 128:
                        continue
                    label_dict = {'-1.0': 0, '0.0': 1, '1.0': 2}
                    index_np = label_dict[data_tmp[-1]]
                    if index_np == 2:
                        if random.uniform(0, 1) > 0.0687:
                            continue
                    if index_np == 0:
                        if random.uniform(0,1) > 0.39164:
                            continue
                    label_np = np.zeros(3)
                    label_np[index_np] = 1
                    x_input_data = result_np.reshape(1, 128)
                    y_label_data = label_np.reshape(1, 3)
                    _, train_loss = sess.run([train_step, loss],
                                             feed_dict={x_input: x_input_data, y_lable: y_label_data})
                    all_STEP += 1
                    sum_loss += train_loss

                    if all_STEP % 2000 == 0:
                        av_loss = sum_loss / 2000.0
                        print("epoch_times  :", epoch_times, "     all_STEP    :", all_STEP, "    av_loss: ", av_loss)
                        sum_loss = 0
                        # print(train_loss)
                    if all_STEP % 100000 == 0:
                        test_alg()
        print('counter: ', counter, '\n')  # 9829
        # x_data =
        # y_data =
        # return result_np,label_np


# 读取字典和词向量
def get_dic(dic_file='dictionary_data.txt', np_file='vectorForWords.npy'):
    # 字典文件
    isExists_dic = os.path.exists(dic_file)
    if not isExists_dic:
        # 如果不存在就提醒不存在字典
        print(dic_file + "   字典文件不存在！！")
    else:
        f = open(dic_file, 'r', encoding='utf-8')
        dictionary_file = f.read()
        reverse_dictionary = eval(dictionary_file)
        dictionary = dict(zip(reverse_dictionary.values(), reverse_dictionary.keys()))
        f.close()
    # 向量文件
    isExists_np = os.path.exists(np_file)
    if not isExists_np:
        print(np_file + "   不存在！")
        # 如果不存在就提醒不存在向量
    else:
        vectorall_words = np.load(np_file)
    return dictionary, vectorall_words


def mode_base():
    x_input = tf.placeholder(tf.float32, shape=(None, 128))
    y_lable = tf.placeholder(tf.float32, shape=(None, 3))  # 不指定 暂时3个

    hiddenLayer1 = add_layer("layer1", x_input, in_size=128, out_size=64, activation_function=tf.tanh)

    hiddenLayer2 = add_layer("layer2", hiddenLayer1, in_size=64, out_size=32, activation_function=tf.tanh)

    hiddenLayer3 = add_layer("layer3", hiddenLayer2, in_size=32, out_size=16, activation_function=tf.tanh)

    prediction = add_layer("end", hiddenLayer3, in_size=16, out_size=3, activation_function=tf.nn.softmax)

    # loss = tf.reduce_mean(tf.reduce_sum(y_lable - prediction))
    loss = -tf.reduce_mean(y_lable * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))

    train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    return x_input, y_lable, prediction, loss, train_step, sess


# 测试正确率函数
def test_alg():
    counter_test = 0
    true_counter = 0
    file_name = fileList[200]
    print("\n\n\n Begin test", file_name)
    with client.read( HADOOP_PATH + file_name, encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n').strip('').strip('\r')
            data_tmp = []
            if line != "":
                counter_test += 1
                data_tmp = [word for word in line.split("\t") if word != '']

            if len(data_tmp) == 2:
                if len(data_tmp[-1]) <= 4:
                    result = word2vector(data_tmp[0])
                    result_np = np.array(result, dtype=np.float32)
                    if len(result_np) != 128:
                        continue
                    label_dict = {'-1.0': 0, '0.0': 1, '1.0': 2}
                    index_np = label_dict[data_tmp[-1]]
                    x_input_data = result_np.reshape(1, 128)
                    prediction_val = sess.run(prediction, feed_dict={x_input: x_input_data})
                    prediction_list = prediction_val.tolist()
                    # np_index = np.where(prediction_val == np.max(prediction_val))
                    # list_index = np_index.tolist()
                    list_index = prediction_list[0].index(max(prediction_list[0]))
                    # print("prediction_val :",prediction_val)
                    # print(index_np,list_index)
                    if index_np == list_index:
                        true_counter += 1
                        # print("true_counter",true_counter)
    print("\n\n\n 准确率   ：", true_counter / float(counter_test), "\n\n\n")
    return true_counter / float(counter_test)


if __name__ == '__main__':

    client = hdfs.Client(HADOOP_IP_PORT, root="/", timeout=500, session=False)
    fileList = client.list(HADOOP_PATH)
    # 读取字典词向量
    dictionary, vectorall_words = get_dic()
    # 获取模型
    x_input, y_lable, prediction, loss, train_step, sess = mode_base()

    epoch_times = 0
    while True:
        epoch_times += 1
        for file_loop in fileList:  # 每个数据集中有一批数据
            print('\n', file_loop)
            get_data(client, HADOOP_PATH + file_loop)





            # for i in range(2000):
            #
            #
            #
            #     _,train_loss= sess.run([train_step,loss],feed_dict={x_input :x_input_data ,y_lable: y_label_data})
            #     print(train_loss)
