import os
import numpy as np
import hdfs


class Readfileutils:
    def __init__(self):
        # 字典
        dic_file = 'dictionary_data.txt'
        isExists_dic = os.path.exists(dic_file)
        dictionary = {}
        if not isExists_dic:
            # 如果不存在就提醒不存在字典
            print(dic_file + "   字典文件不存在！！")
        else:
            f = open(dic_file, 'r', encoding='utf-8')
            dictionary_file = f.read()
            reverse_dictionary = eval(dictionary_file)
            dictionary = dict(zip(reverse_dictionary.values(), reverse_dictionary.keys()))
            f.close()
        self.token_dictionary = dictionary


        self.label_dict = {"未分类": 0, "娱乐": 1, "艺术": 2, "体育": 3, "收藏": 4, "时政": 5, "时尚": 6, "社会": 7,
                                          "亲子": 8, "汽车": 9, "女性": 10, "旅游": 11, "科技": 12, "军事": 13, "教育": 14, "健康": 15,
                                          "航空": 16, "国内": 17, "国际": 18, "传媒": 19, "财经": 20}
        pass
    def create_batches(self,client,train_file, batch_size, sequence_length):
        self.x_data = []
        self.y_data = []
        # TODO padding_index
        padding_index = 0

        with client.read(train_file, encoding='utf-8') as f:
            for line in f:
                if len(line.rstrip().split('\t')) == 2:
                    text, label = line.rstrip().split('\t')
                    tokens = text.split(',')[:-1]
                    if len(tokens) >= 10 and label in self.label_dict:

                        seq_ids = [self.token_dictionary.get(token) for token in tokens
                                   if self.token_dictionary.get(token) is not None]
                        seq_ids = seq_ids[:sequence_length]
                        for _ in range(len(seq_ids),sequence_length):
                            seq_ids.append(padding_index)
                        self.x_data.append(seq_ids)
                        self.y_data.append(self.label_dict.get(label))
        self.num_batches = int(len(self.x_data) / batch_size)
        self.skipfile = False

        if self.num_batches == 0:
            self.skipfile = True

        if self.skipfile == False:
            self.x_data = self.x_data[:self.num_batches * batch_size]
            self.y_data = self.y_data[:self.num_batches * batch_size]

            self.x_data = np.array(self.x_data, dtype=int)
            self.y_data = np.array(self.y_data, dtype=int)
            self.x_batches = np.split(self.x_data.reshape(batch_size, -1), self.num_batches, 1)
            self.y_batches = np.split(self.y_data.reshape(batch_size, -1), self.num_batches, 1)
            self.pointer = 0

    def label_one_hot(self, label_id):
        self.n_classes = len(self.label_dict)
        y = [0] * self.n_classes
        y[int(label_id)] = 1.0
        return np.array(y)
    def next_batch(self):
        index = self.batch_index[self.pointer]
        self.pointer += 1
        x_batch, y_batch = self.x_batches[index], self.y_batches[index]
        y_batch = [self.label_one_hot(y) for y in y_batch]
        return x_batch, y_batch
    def reset_batch(self):
        self.batch_index = np.random.permutation(self.num_batches)
        self.pointer = 0

if __name__ == '__main__':
    example = Readfileutils()
    print()
    example.create_batches("part-m-00000",128,20)
    example.reset_batch()
    xx,yy = example.next_batch()
    print()