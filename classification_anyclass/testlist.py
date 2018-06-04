import numpy as np
input = [1,2]
# API calls will begin at the apply() method, with the request body passed as 'input'
# For more details, TODO 这里后期加上说明！！！！
def apply(input):
    np_input = np.array(input)
    np_input *= 2
    result_list = np_input.tolist()

    return "hello {}".format(result_list)
aa = apply(input)
print(aa)