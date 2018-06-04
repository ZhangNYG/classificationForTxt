# -*-coding:utf-8 -*-

from six.moves import xrange
import hdfs
import collections

##############################
# 参数设置
# VOCABULARY_SIZE = LinuxDidtributionWord2vector.VOCABULARY_SIZE
# hadoop中的路径
HADOOP_IP_PORT = "http://10.1.0.42:50070"
HADOOP_PATH = "/user/cdh/guojie/News_CategoryDataSet/"

def read_data(client,filename):
    with client.read(filename,encoding='utf-8') as f:
        data = []
        counter = 0
        for line in f:
            line = line.strip('\n').strip('').strip('\r')
            data_tmp = []
            if line != "" :
                counter += 1
                data_tmp = [word for word in line.split("\t") if word != '']

            if len(data_tmp) == 2 :
                if len(data_tmp[-1]) <= 40 :
                    # print(data_tmp)
                    # print(data_tmp[-1],type(data_tmp),len(data_tmp))
                    data.append(data_tmp[-1])
                    print(data_tmp[-1])
                    # print(len(data_tmp[-1]))
                # print(data)
            # print(data_tmp)
        print('counter: ',counter) #9829
        print('data-words: ', len(data))
    return data


############################################################################
##############################################################################
# Step 2: Build the dictionary and replace rare words with UNK token.
# 建立数据字典
def build_dic(collectionsCounter):
    count = [['UNK', -1]]
    count.extend(collectionsCounter.most_common())
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    unk_count = 0
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return count, dictionary, reverse_dictionary


####################################################################################
# 读取数据
if __name__ == '__main__':

    client = hdfs.Client(HADOOP_IP_PORT, root="/", timeout=500, session=False)
    # 存储文件路径名
    path_file_dict = dict()
    collectionsCounter = collections.Counter()

    fileList = client.list(HADOOP_PATH)
    for file_loop in fileList:  # 每个数据集中有一批数据
        # 对路径进行储存
        path_file_dict[HADOOP_PATH + file_loop] = len(path_file_dict) + 1
        # 产生读取每个文件
        words = read_data(client, HADOOP_PATH + file_loop)
        # print(type(words),'   ',words)
        print("文件路径名称： ", HADOOP_PATH + file_loop, '    Data size: ', len(words))
        # count = [['UNK', -1]]
        collectionsCounter = sum((collectionsCounter,collections.Counter(words)),collections.Counter())
        print('collectionsCounter: ',len(collectionsCounter))
    # 保存collectionsCounter
    dict_all_counter = dict(collectionsCounter)
    f_all_dict = open('dict_all_counter.txt', 'w', encoding='utf-8')
    f_all_dict.write(str(dict_all_counter))
    f_all_dict.close()
    # 保存所有的计数
    all_words_num = sum(collectionsCounter.values())
    print('所有文件总共字数: ',all_words_num)
    f_words_sum = open('sum_all_words','w',encoding='utf-8')
    f_words_sum.write('所有文件总共字数: ')
    f_words_sum.write(str(all_words_num))
    f_words_sum.close()
    # 利用collectionsCounter生成字典
    count, dictionary, reverse_dictionary = build_dic(collectionsCounter)
    # 保存字典
    f_dict = open('dictionary_data.txt', 'w', encoding='utf-8')
    f_dict.write(str(reverse_dictionary))
    f_dict.close()
    # 保存统计字频
    f_count = open('count_data.txt', 'w', encoding='utf-8')
    f_count.write(str(count))
    f_count.close()
    # 存储的文件路径名
    reverse_path_file_dict = dict(zip(path_file_dict.values(), path_file_dict.keys()))
    print(reverse_path_file_dict,len(reverse_path_file_dict))
    # 保存hadoop中所有文件路径键值对
    f_count = open('reverse_path_file_dict.txt', 'w', encoding='utf-8')
    f_count.write(str(reverse_path_file_dict))
    f_count.close()



    #aaa = collectionsCounter

    # 保存字典与统计数据



    # for file_loop in fileList:
    #     # 产生读取每个文件
    #     words = read_data(client, HADOOP_PATH + file_loop)
    #     print("文件路径名称： ",HADOOP_PATH+file_loop,'    Data size: ', len(words))
    #     # loop 统计全部字频
    #
    #     # 统计数据，建立字典
    #     data, count, dictionary, reverse_dictionary = build_dataset(words)
    #     # 保存字典
    #     f_dict = open('dictionary_data.txt', 'w', encoding='utf-8')
    #     f_dict.write(str(reverse_dictionary))
    #     f_dict.close()
    #     # 保存统计字频
    #     f_count = open('count_data.txt', 'w', encoding='utf-8')
    #     f_count.write(str(count))
    #     f_count.close()
    #
    #     del words  # Hint to reduce memory.
    #     print('Most common words (+UNK)', count[:5])
    #     print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
