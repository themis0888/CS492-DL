# -*- coding: utf-8 -*- 
import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

def custom_model_fn(features, labels, mode):
    """Model function for PA2 Extra2"""

    input_layer = tf.reshape(features["x"], [-1, 32, 32, 3])

    # Convolutional Layer #1
    W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 64], stddev=5e-2))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(input_layer, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

    # 첫번째 Pooling layer
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 두번째 convolutional layer -- 32개의 특징들(feature)을 64개의 특징들(feature)로 맵핑(maping)한다.
    W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 64], stddev=5e-2))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

    # 두번째 pooling layer.
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 세번째 convolutional layer
    W_conv3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=5e-2))
    b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)

    # 네번째 convolutional layer
    W_conv4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
    b_conv4 = tf.Variable(tf.constant(0.1, shape=[128])) 
    h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4)

    # 다섯번째 convolutional layer
    W_conv5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
    b_conv5 = tf.Variable(tf.constant(0.1, shape=[128]))
    h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5, strides=[1, 1, 1, 1], padding='SAME') + b_conv5)

    # Fully Connected Layer 1 -- 2번의 downsampling 이후에, 우리의 32x32 이미지는 8x8x128 특징맵(feature map)이 된다.
    # 이를 384개의 특징들로 맵핑(maping)한다.
    W_fc1 = tf.Variable(tf.truncated_normal(shape=[8 * 8 * 128, 384], stddev=5e-2))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[384]))

    h_conv5_flat = tf.reshape(h_conv5, [-1, 8*8*128])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)

    # Dropout - 모델의 복잡도를 컨트롤한다. 특징들의 co-adaptation을 방지한다.
    h_fc1_drop = tf.nn.dropout(h_fc1, 0.5) 

    # 384개의 특징들(feature)을 10개의 클래스-airplane, automobile, bird...-로 맵핑(maping)한다.
    W_fc2 = tf.Variable(tf.truncated_normal(shape=[384, 10], stddev=5e-2))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
    logits = tf.matmul(h_fc1_drop,W_fc2) + b_fc2

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # In predictions, return the prediction value, do not modify
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Select your loss and optimizer from tensorflow API
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits) # Refer to tf.losses

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.001) # Refer to tf.train
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


if __name__ == '__main__':
    # Write your dataset path
    dataset_train = np.load('../extra2-train.npy')
    dataset_eval =  np.load('../extra2-valid.npy')
    test_data =  np.load('../extra2-test_img.npy')

    f_dim = dataset_train.shape[1] - 1
    train_data = dataset_train[:,:f_dim].astype(np.float32)
    train_labels = dataset_train[:,f_dim].astype(np.int32)
    eval_data = dataset_eval[:,:f_dim].astype(np.float32)
    eval_labels = dataset_eval[:,f_dim].astype(np.int32)
    test_data = test_data.astype(np.float32)

    # Save model and checkpoint
    mnist_classifier = tf.estimator.Estimator(model_fn=custom_model_fn, model_dir="./model")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # Train the model. You can train your model with specific batch size and epoches
    train_input = tf.estimator.inputs.numpy_input_fn(x={"x": train_data},
        y=train_labels, batch_size=100, num_epochs=None, shuffle=True)
    mnist_classifier.train(input_fn=train_input, steps=2000, hooks=[logging_hook])

    # Eval the model. You can evaluate your trained model with validation data
    eval_input = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data},
        y=eval_labels, num_epochs=1, shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input)


    ## ----------- Do not modify!!! ------------ ##
    # Predict the test dataset
    pred_input = tf.estimator.inputs.numpy_input_fn(x={"x": test_data}, shuffle=False)
    pred_results = mnist_classifier.predict(input_fn=pred_input)
    result = np.asarray([x.values()[1] for x in list(pred_results)])
    ## ----------------------------------------- ##

    np.save('extra2_your_studentID.npy', result)
