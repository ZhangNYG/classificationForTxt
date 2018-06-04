# tmpset = set()
# with open("part-m-00000", "r", encoding="utf-8") as f:
#     for line in f:
#         print(line)
#         tmpset.add(line)
import random

list = [0, 56, 1, 300, 182, 119, 156, 257, 10, 13, 0, 4, 300, 24, 196, 14, 0, 114, 51, 0, 125]
aa = sum(list)
list1 = [0, 63, 2, 300, 294, 231, 233, 300, 14, 21, 0, 6, 300, 29, 300, 20, 0, 131, 56, 0, 157]
bbb = sum(list1)


list3 = [0, 63, 2, 300, 300, 300, 300, 300, 24, 32, 0, 9, 300, 38, 300, 28, 1, 146, 59, 0, 191]
ccc = sum(list3)
list4 = [0, 65, 2, 300, 300, 300, 300, 300, 24, 32, 0, 9, 300, 38, 300, 28, 1, 150, 60, 0, 194]
ddd = sum(list4)


label_dict = {"未分类": 0, "娱乐": 1, "艺术": 2, "体育": 3, "收藏": 4, "时政": 5, "时尚": 6, "社会": 7,
                                          "亲子": 8, "汽车": 9, "女性": 10, "旅游": 11, "科技": 12, "军事": 13, "教育": 14, "健康": 15,
                                          "航空": 16, "国内": 17, "国际": 18, "传媒": 19, "财经": 20}
aa = "未分类"
bb = " "
if aa in label_dict:
    print(aa)

if bb in label_dict:
    print(bb)
    print(11)

aa = [1,2,3]
ccc = [random.randint(1, 10) for i in range(len(aa))]


import numpy as np
aa =  np.load("vectorForWords.npy")
bb = aa.copy()
aa[0] = 0

np.save("vectorForWords_zero0.npy",aa)