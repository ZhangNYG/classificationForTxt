import os
import os.path
from jpype import *
import hdfs
import numpy as np


HADOOP_IP_PORT = "http://10.1.0.41:50070"
HADOOP_PATH = "/user/cdh/guojie/News_CategoryDataSet/"


############################ Jpype local Variable Define ###########################
# Define the directory where the jar package is located (define the lib)
jarpath = os.path.join(os.path.abspath('.'), '/home/cdh/guojie/')
# Define the jar package name
jarname = 'TextVectExpression-0.0.1-SNAPSHOT.jar'
# Define the class file path
classpath = 'com.neusoft.sts.textVector.TextVectExpression'

startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=%s" % (jarpath + jarname))

JDClass = JClass(classpath)

javaInstance = JDClass()
###############################  Referenced Function  ##############################

# JpypeJavaApi Function: Calling  TextVectExpression-0.0.1-SNAPSHOT.jar through jpype,return list<float>
# def JpypeJavaApi(text):
#     ################################ Jpype Java API ###############################
#
#     # Start the JAVA Virtual Machine：[ startJVM(jvmpath, *args) ]
#     # startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=%s" % (jarpath + jarname))
#
#     # Java println() object: Use to println(log) == [jprint(log)]
#     # jprint = java.lang.System.out.println
#
#     # JDClass = JClass(classpath)
#
#     # Call java constructor
#     # javaInstance = JDClass()
#     # print(text)
#     # TextVectExpression-0.0.1-SNAPSHOT.jar
#     result = javaInstance.run(text)
#     # for index in result:
#     # 	jprint('shape[i]: ' + str(index))
#     return result

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
                if len(data_tmp[-1]) <= 40 :
                    print(data_tmp)
                    result = javaInstance.run(data_tmp[0]) #JpypeJavaApi(data_tmp[0])
                    result_list = list(result)
                    result_np = np.array(result,dtype=np.float32)
                    print(type(result_np),len(result_np),result_np)
                    print("list:   ",type(result_list),len(result_list), result_list)
                    # data.append(data_tmp[-1])
                    print(data_tmp[0])
                    print(result)
                    print(type(result))
                    print(data_tmp[-1])
        print('counter: ',counter) #9829
    # x_data =
    # y_data =
    # return x_data,y_data



client = hdfs.Client(HADOOP_IP_PORT, root="/", timeout=500, session=False)
fileList = client.list(HADOOP_PATH)
for file_loop in fileList:  # 每个数据集中有一批数据
    get_data(client, HADOOP_PATH + file_loop)
