import tensorflow as tf
import os
import os.path
# from jpype import *
# import hdfs
import numpy as np

HADOOP_IP_PORT = "http://10.1.0.41:50070"
HADOOP_PATH = "/user/cdh/guojie/Weibo_EmotionDataSet/"


# ############################ Jpype local Variable Define ###########################
# # Define the directory where the jar package is located (define the lib)
# jarpath = os.path.join(os.path.abspath('.'), '/home/cdh/guojie/')
# # Define the jar package name
# jarname = 'TextVectExpression-0.0.1-SNAPSHOT.jar'
# # Define the class file path
# classpath = 'com.neusoft.sts.textVector.TextVectExpression'
#
#
# startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=%s" % (jarpath + jarname))
#
# JDClass = JClass(classpath)
#
# javaInstance = JDClass()
# ###############################  Referenced Function  ##############################



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
    av_vector = sum_vector/len(list_word_value)
    print(av_vector)
    return  av_vector







def get_data(client,filename):

    with client.read(filename,encoding='utf-8') as f:
        counter = 0
        for line in f:
            line = line.strip('\n').strip('').strip('\r')
            data_tmp = []
            if line != "" :
                counter += 1
                data_tmp = [word for word in line.split("\t") if word != '']

            if len(data_tmp) == 2 :
                if len(data_tmp[-1]) <= 4 :
                    result = word2vector(data_tmp[0])
                    result_np = np.array(result, dtype=np.float32)
                    if len(result_np) != 128:
                        continue
                    # # data.append(data_tmp[-1])
                    # print(data_tmp[0])
                    # print(result)
                    # print(type(result))
                    # print(data_tmp[-1])
                    label_dict = {'-1.0':0,'0.0':1,'1.0':2}
                    index_np = label_dict[data_tmp[-1]]
                    label_np = np.zeros(3)
                    label_np[index_np] = 1
                    x_input_data = result_np.reshape(1,128)
                    y_label_data = label_np.reshape(1,3)
                    _, train_loss = sess.run([train_step, loss],
                                             feed_dict={x_input: x_input_data, y_lable: y_label_data})
                    print(train_loss)
        print('counter: ',counter) #9829
    # x_data =
    # y_data =
    # return result_np,label_np

# 字典文件
isExists_dic = os.path.exists('dictionary_data.txt')
if not isExists_dic:
    # 如果不存在就提醒不存在字典
    print('dictionary_data.txt  ' + " 字典文件不存在！！")
else:
    f = open('dictionary_data.txt', 'r', encoding='utf-8')
    dictionary_file = f.read()
    reverse_dictionary = eval(dictionary_file)
    dictionary = dict(zip(reverse_dictionary.values(), reverse_dictionary.keys()))
    f.close()
# 向量文件
isExists_np = os.path.exists('vectorForWords.npy')
if not isExists_np:
    print("vectorForWords.npy   不存在！")
    # 如果不存在就提醒不存在向量
else:
    vectorall_words = np.load('vectorForWords.npy')
line_words = "你好,闪电湖,中央,过节,中国,"
word2vector(line_words)



x_input = tf.placeholder(tf.float32,shape=(None,128))
y_lable = tf.placeholder(tf.float32,shape =(None,3))  # 不指定 暂时3个

hiddenLayer1 = add_layer("layer1", x_input,in_size=128, out_size= 512, activation_function= tf.tanh)

hiddenLayer2 = add_layer("layer2",hiddenLayer1,in_size = 512 ,out_size = 128, activation_function= tf.tanh)

prediction = add_layer("end",hiddenLayer2,in_size= 128, out_size=3 ,activation_function= tf.nn.softmax)

loss = tf.reduce_mean(tf.reduce_sum(y_lable - prediction))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)


client = hdfs.Client(HADOOP_IP_PORT, root="/", timeout=500, session=False)
fileList = client.list(HADOOP_PATH)
for file_loop in fileList:  # 每个数据集中有一批数据
    get_data(client, HADOOP_PATH + file_loop)




# for i in range(2000):
#
#
#
#     _,train_loss= sess.run([train_step,loss],feed_dict={x_input :x_input_data ,y_lable: y_label_data})
#     print(train_loss)