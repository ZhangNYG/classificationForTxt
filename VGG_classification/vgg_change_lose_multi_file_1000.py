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
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", shape=[1, out_size],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))

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

        data_tmp_batch = []
        sum_loss = 0
        for line in f:
            line = line.strip('\n').strip('').strip('\r')
            data_tmp = []
            if line != "":
                counter += 1
                data_tmp = [word for word in line.split("\t") if word != '']
            data_tmp_batch.append(data_tmp)
            if counter != 0 and counter % 128 ==0:
                # 生成batch
                result_batch = []
                label_batch = []
                for i_batch in range(128):
                    if len(data_tmp_batch[i_batch]) == 2:
                        if len(data_tmp_batch[i_batch][-1]) <= 4:
                            result = word2vector(data_tmp_batch[i_batch][0])
                            # print(result)

                            result_list_two = []
                            for i_i in result.tolist():
                                list_i = []
                                list_i.append(i_i)
                                result_list_two.append(list_i)

                            # print("result_list_two  :",result_list_two)

                            list_list_tmp = []
                            list_list_tmp.append(result_list_two)
                            result_batch.append(list_list_tmp)

                            label_dict = {'-1.0': 0, '0.0': 1, '1.0': 2}
                            index_np = label_dict[data_tmp_batch[i_batch][-1]]
                            tmp_label = [0,0,0]
                            tmp_label[index_np] = 1
                            label_batch.append(tmp_label)
                # print("result_batch   :",result_batch)
                # print("\n len(data_tmp_batch)",len(data_tmp_batch),len(result_batch))
                label_np_batch = np.array(label_batch)
                result_batch_np = np.array(result_batch)
                result_batch_np = result_batch_np * 1000
                data_tmp_batch = []

                # eval_y = sess.run(prediction, feed_dict={x_input: result_batch_np})
                # print(eval_y)
                # print(label_np_batch)

                _, train_loss = sess.run([train_step, loss],
                                         feed_dict={x_input: result_batch_np, y_lable: label_np_batch})
                all_STEP += 1
                sum_loss += train_loss

                if all_STEP % 20 == 0:
                    av_loss = sum_loss / 20.0
                    print("epoch_times  :", epoch_times, "     all_STEP    :", all_STEP, "    av_loss: ", av_loss)
                    sum_loss = 0
                    # print(train_loss)
                if all_STEP % 20 == 0:
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
    parameters = []
    x_input = tf.placeholder(tf.float32, shape=(None, 1, 128, 1))
    y_lable = tf.placeholder(tf.float32, shape=(None, 3))  # 不指定 暂时3个

    # conv1_1
    with tf.name_scope('conv1_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 3, 1, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(x_input, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1_1 = tf.nn.tanh(out, name=scope)
        parameters += [kernel, biases]

    # conv1_2
    with tf.name_scope('conv1_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 3, 64, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1_2 = tf.nn.tanh(out, name=scope)
        parameters += [kernel, biases]

    # pool1
    pool1 = tf.nn.max_pool(conv1_2,
                                ksize=[1, 1, 2, 1],
                                strides=[1, 1, 2, 1],
                                padding='SAME',
                                name='pool1')

    # conv2_1
    with tf.name_scope('conv2_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 3, 64, 128], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2_1 = tf.nn.tanh(out, name=scope)
        parameters += [kernel, biases]

    # conv2_2
    with tf.name_scope('conv2_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 3, 128, 128], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2_2 = tf.nn.tanh(out, name=scope)
        parameters += [kernel, biases]

    # pool2
    pool2 = tf.nn.max_pool(conv2_2,
                                ksize=[1, 1, 2, 1],
                                strides=[1, 1, 2, 1],
                                padding='SAME',
                                name='pool2')

    # conv3_1
    with tf.name_scope('conv3_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 3, 128, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_1 = tf.nn.tanh(out, name=scope)
        parameters += [kernel, biases]

    # conv3_2
    with tf.name_scope('conv3_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 3, 256, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_2 = tf.nn.tanh(out, name=scope)
        parameters += [kernel, biases]

    # conv3_3
    with tf.name_scope('conv3_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 3, 256, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_3 = tf.nn.tanh(out, name=scope)
        parameters += [kernel, biases]

    # pool3
    pool3 = tf.nn.max_pool(conv3_3,
                                ksize=[1, 1, 2, 1],
                                strides=[1, 1, 2, 1],
                                padding='SAME',
                                name='pool3')

    # conv4_1
    with tf.name_scope('conv4_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 3, 256, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_1 = tf.nn.tanh(out, name=scope)
        parameters += [kernel, biases]

    # conv4_2
    with tf.name_scope('conv4_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_2 = tf.nn.tanh(out, name=scope)
        parameters += [kernel, biases]

    # conv4_3
    with tf.name_scope('conv4_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_3 = tf.nn.tanh(out, name=scope)
        parameters += [kernel, biases]

    # pool4
    pool4 = tf.nn.max_pool(conv4_3,
                                ksize=[1, 1, 2, 1],
                                strides=[1, 1, 2, 1],
                                padding='SAME',
                                name='pool4')

    # conv5_1
    with tf.name_scope('conv5_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_1 = tf.nn.tanh(out, name=scope)
        parameters += [kernel, biases]

    # conv5_2
    with tf.name_scope('conv5_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_2 = tf.nn.tanh(out, name=scope)
        parameters += [kernel, biases]

    # conv5_3
    with tf.name_scope('conv5_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_3 = tf.nn.tanh(out, name=scope)
        parameters += [kernel, biases]

    # pool5
    pool5 = tf.nn.max_pool(conv5_3,
                                ksize=[1, 1, 2, 1],
                                strides=[1, 1, 2, 1],
                                padding='SAME',
                                name='pool4')



    shape = int(np.prod(pool5.get_shape()[1:]))

    pool5_flat = tf.reshape(pool5, [-1, shape])


    hiddenLayer1 = add_layer("layer1", pool5_flat, in_size=shape, out_size=2048, activation_function=tf.tanh)

    hiddenLayer2 = add_layer("layer2", hiddenLayer1, in_size=2048, out_size=1024, activation_function=tf.tanh)

    hiddenLayer3 = add_layer("layer3", hiddenLayer2, in_size=1024, out_size=512, activation_function=tf.tanh)

    hiddenLayer4 = add_layer("layer4", hiddenLayer3, in_size=512, out_size=128, activation_function=tf.tanh)

    hiddenLayer5 = add_layer("layer5", hiddenLayer4, in_size=128, out_size=16, activation_function=tf.tanh)

    prediction = add_layer("end", hiddenLayer5, in_size=16, out_size=3)

    losses = tf.nn.softmax_cross_entropy_with_logits(logits= prediction , labels=y_lable)

    # loss = tf.reduce_mean(tf.reduce_sum(y_lable - prediction))
    # loss = -tf.reduce_mean(y_lable * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
    loss = tf.reduce_mean(losses)

    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    return x_input, y_lable, prediction, loss, train_step, sess


# 测试正确率函数
def test_alg():
    counter_test = 0
    true_counter = 0
    file_name = fileList[1]
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
                    result_list_two = []
                    for i_i in result.tolist():
                        list_i = []
                        list_i.append(i_i)
                        result_list_two.append(list_i)

                    result_batch = []
                    list_list_tmp = []
                    list_list_tmp.append(result_list_two)
                    result_batch.append(list_list_tmp)
                    # print(result_batch)
                    result_np = np.array(result_batch, dtype=np.float32)

                    label_dict = {'-1.0': 0, '0.0': 1, '1.0': 2}
                    index_np = label_dict[data_tmp[-1]]

                    result_np = result_np * 1000
                    prediction_val = sess.run(prediction, feed_dict={x_input: result_np})
                    prediction_list = prediction_val.tolist()
                    # print(prediction_list)
                    # np_index = np.where(prediction_val == np.max(prediction_val))
                    # list_index = np_index.tolist()
                    list_index = prediction_list[0].index(max(prediction_list[0]))
                    # print(list_index)

                    # print("prediction_val :",prediction_val)
                    # print(index_np,list_index)
                    if index_np == list_index:
                        true_counter += 1
                        # print("true_counter",true_counter)
                    if counter_test>1000:
                        print("\n 准确率   ：", true_counter / float(counter_test), "\n")
                        break
    # print("\n\n\n 准确率   ：", true_counter / float(counter_test), "\n\n\n")
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
        fileListForloop = fileList[2:20]
        for file_loop in fileListForloop:  # 每个数据集中有一批数据
            print('\n', file_loop)
            get_data(client, HADOOP_PATH + file_loop)
