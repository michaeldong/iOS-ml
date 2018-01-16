import tfcoreml as tf_converter

tf_converter.convert(tf_model_path = 'XXXX.pb',
                     mlmodel_path = 'YYYYYY.mlmodel',
                     output_feature_names = ['output:0'],
                     is_bgr = True,
                     input_name_shape_dict = {'input:0' : [1, 416, 416, 3]},
                     image_input_names = ['input:0'],
                     image_scale = 1 / 255.0)
