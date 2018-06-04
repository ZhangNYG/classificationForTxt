 part-m-00004
Traceback (most recent call last):
  File "change_lose_best_save_model_samping.py", line 228, in <module>
    get_data(client, HADOOP_PATH + file_loop)
  File "change_lose_best_save_model_samping.py", line 98, in get_data
    feed_dict={x_input: result_batch_np, y_lable: label_np_batch})
  File "/home/cdh/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 889, in run
    run_metadata_ptr)
  File "/home/cdh/anaconda3/envs/tensorflow/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 1096, in _run
    % (np_val.shape, subfeed_t.name, str(subfeed_t.get_shape())))
ValueError: Cannot feed value of shape (0,) for Tensor 'Placeholder:0', which has shape '(?, 128)'
(tensorflow) [cdh@dev3 tuning_parameters]$ 
(tensorflow) [cdh@dev3 tuning_parameters]$ python change_lose_best_save_model_samping.py 
2018-04-30 10:43:19.529489: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX

