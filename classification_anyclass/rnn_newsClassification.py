# 准确率最高 0.9 不加抽样
import tensorflow as tf
import os.path
# import hdfs
import numpy as np
import time

from classification_anyclass.read_file import Readfileutils
from classification_anyclass.model import BiRNN


# HADOOP_IP_PORT = "http://10.1.0.42:50070"
# HADOOP_PATH = "/user/cdh/guojie/Weibo_EmotionDataSet/"

# Parameters
# =================================================
tf.flags.DEFINE_integer('embedding_size', 128, 'embedding dimension of tokens')
tf.flags.DEFINE_integer('rnn_size', 128,
                        'hidden units of RNN , as well as dimensionality of character embedding (default: 100)')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability (default : 0.5)')
tf.flags.DEFINE_integer('layer_size', 2, 'number of layers of RNN (default: 2)')
tf.flags.DEFINE_integer('batch_size', 128, 'Batch Size (default : 32)')
tf.flags.DEFINE_integer('sequence_length', 50, 'Sequence length (default : 32)')
tf.flags.DEFINE_integer('attn_size', 200, 'attention layer size')
tf.flags.DEFINE_float('grad_clip', 5.0, 'clip gradients at this value')
tf.flags.DEFINE_integer("num_epochs", 1, 'Number of training epochs (default: 200)')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
# tf.flags.DEFINE_string('train_file', 'train.txt', 'train raw file')
tf.flags.DEFINE_string('test_file', 'test.txt', 'train raw file')
tf.flags.DEFINE_string('data_dir', 'data', 'data directory')
tf.flags.DEFINE_string('save_dir', 'save', 'model saved directory')
tf.flags.DEFINE_string('log_dir', 'log', 'log info directiory')
tf.flags.DEFINE_string('pre_trained_vec', 'vectorForWords.npy', 'using pre trained word embeddings, npy file format')
tf.flags.DEFINE_string('init_from', None, 'continue training from saved model at this path')
tf.flags.DEFINE_integer('save_steps', 1000, 'num of train steps for saving model')
tf.flags.DEFINE_integer('n_classes', 21, 'num of n_classes')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
    print('{0}={1}'.format(attr.upper(), value))


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






if __name__ == '__main__':

    # client = hdfs.Client(HADOOP_IP_PORT, root="/", timeout=500, session=False)
    # fileList = client.list(HADOOP_PATH)
    # 读取字典词向量
    dictionary, embeddings = get_dic()
    print(embeddings.shape)
    FLAGS.vocab_size = embeddings.shape[0]
    FLAGS.embedding_size = embeddings.shape[1]
    # 获取模型
    if FLAGS.init_from is not None:
        assert os.path.isdir(FLAGS.init_from), '{} must be a directory'.format(FLAGS.init_from)
        ckpt = tf.train.get_checkpoint_state(FLAGS.init_from)
        assert ckpt, 'No checkpoint found'
        assert ckpt.model_checkpoint_path, 'No model path found in checkpoint'

    # Define specified Model
    model = BiRNN(embedding_size=FLAGS.embedding_size, rnn_size=FLAGS.rnn_size, layer_size=FLAGS.layer_size,
                  vocab_size=FLAGS.vocab_size, attn_size=FLAGS.attn_size, sequence_length=FLAGS.sequence_length,
                  n_classes=FLAGS.n_classes, grad_clip=FLAGS.grad_clip, learning_rate=FLAGS.learning_rate)

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=1)

        # using pre trained embeddings
        if FLAGS.pre_trained_vec:
            sess.run(model.embedding.assign(embeddings))
            del embeddings
        # restore model
        if FLAGS.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)

        readFileClass = Readfileutils()
        epoch_times = 0
        all_STEP = 0
        while True:
            epoch_times += 1
            # fileList_loop = fileList[1:100]
            # file_loop = fileList[1]  # 每个数据集中有一批数据
            for file_loop in range(3):
                print("part-m-0000"+ str(file_loop))
                readFileClass.create_batches("part-m-0000"+ str(file_loop),128,50)
                readFileClass.reset_batch()

                if readFileClass.skipfile == False:
                    for b in range(readFileClass.num_batches):
                        start = time.time()
                        x_batch, y_batch = readFileClass.next_batch()
                        feed = {model.input_data: x_batch, model.targets: y_batch, model.output_keep_prob: FLAGS.dropout_keep_prob}
                        train_loss, _ = sess.run([model.cost, model.train_op], feed_dict=feed)
                        end = time.time()
                        all_STEP += 1
                        print("epoch_times  :", epoch_times, "     all_STEP    :", all_STEP, "    train_loss: ", train_loss)
                        print('{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}'.format(all_STEP,
                                                                                                  all_STEP,all_STEP,
                                                                                                train_loss,
                                                                                                  end - start))
                        if all_STEP % FLAGS.save_steps == 0:
                            checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
                            saver.save(sess, checkpoint_path, global_step=all_STEP)
                            print('model saved to {}'.format(checkpoint_path))
            embedingeval = sess.run(model.embedding)
            print(embedingeval[0])
            test_data_loader = Readfileutils()
            test_data_loader.create_batches("part-m-00000",128,50)
            test_data_loader.reset_batch()
            test_accuracy = []
            for i in range(test_data_loader.num_batches):
                test_x, test_y = test_data_loader.next_batch()
                feed = {model.input_data: test_x, model.targets: test_y, model.output_keep_prob: 1.0}
                accuracy = sess.run(model.accuracy, feed_dict=feed)
                test_accuracy.append(accuracy)

            print('test accuracy:{0}'.format(np.average(test_accuracy)))
