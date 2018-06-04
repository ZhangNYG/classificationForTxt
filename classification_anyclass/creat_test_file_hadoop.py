import random
import hdfs

label_dict = {"未分类": 0, "娱乐": 1, "艺术": 2, "体育": 3, "收藏": 4, "时政": 5, "时尚": 6, "社会": 7,
                                          "亲子": 8, "汽车": 9, "女性": 10, "旅游": 11, "科技": 12, "军事": 13, "教育": 14, "健康": 15,
                                          "航空": 16, "国内": 17, "国际": 18, "传媒": 19, "财经": 20}



HADOOP_IP_PORT = "http://10.1.0.41:50070"
HADOOP_PATH = "/user/cdh/guojie/News_ArticleCategorySet/"
client = hdfs.Client(HADOOP_IP_PORT, root="/", timeout=500, session=False)
fileList = client.list(HADOOP_PATH)
fileList_true = fileList[1:]
count_lable = [0] * 21
list_test_data = []

while True:
    for train_file in fileList_true:
        print(train_file)
        with client.read(HADOOP_PATH + train_file, encoding='utf-8') as f:
            for line in f:
                if len(line.rstrip().split('\t')) == 2:
                    text, label = line.rstrip().split('\t')
                    if label in label_dict:
                        random_haha = random.uniform(0,1)
                        if random_haha <= 0.001 and count_lable[label_dict[label]] < 300:
                            list_test_data.append(line)
                            count_lable[label_dict[label]] += 1
                            print(line)
                            print(count_lable)
        if sum(count_lable) == 21*300:
            with open("test_baocun.txt", mode="w", encoding="utf-8") as save_file:
                for i in range(len(list_test_data)):
                    save_file.write(str(list_test_data[i]))
            exit()




