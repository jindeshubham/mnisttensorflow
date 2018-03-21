import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

learning_rate = 0.00001
epochs = 10
batch_size = 128

# number of samples to calculate validation and accuracy
# decrease this if you're running out of memory
test_valid_size = 256

# network Parameters
n_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.75  # dropout (probability to keep units)


weights = {"wc1":tf.Variable(tf.truncated_normal([5,5,1,32])),
           "wc2":tf.Variable(tf.truncated_normal([5,5,32,64]),
                             ),
           "wd1":tf.Variable(tf.truncated_normal([7*7*32,1024])),
           "wd2":tf.Variable(tf.truncated_normal([1024,n_classes]))}

biases = {"wb1":tf.zeros(32),
          "wb2":tf.zeros(64),
          "wd1":tf.zeros(1024),
          "wd2":tf.zeros(n_classes)}

conv_1 = tf.nn.conv2d(input,weights['wc1'],strides=2,padding="SAME")
conv_1 = tf.nn.bias_add(conv_1,biases["wb1"])
conv_1 = tf.nn.max_pool(conv_1,ksize=2,strides=2,padding="SAME")
conv_2 = tf.nn.conv2d(conv_1,weights['wc2'],strides=2,padding="SAME")
conv_2 = tf.nn.bias_add(conv_2,biases["wb2"])
conv_2 = tf.nn.max_pool(conv_2,ksize=2,strides=2,padding="SAME")
fc1 = tf.reshape(conv_2,[-1,weights["wd1"].get_shape().as_list()[0]])
fc1 = tf.add(tf.matmul(fc1,weights['wd1']),biases['wd1'])
fc1 = tf.nn.relu(fc1)
fc1 = tf.nn.dropout(fc1,dropout)
out = tf.add(tf.matmul(fc1,weights['wd2']),biases['wd2'])

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# Model

# Define loss and optimizer
cost = tf.reduce_mean(\
    tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
    .minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf. global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        for batch in range(mnist.train.num_examples//batch_size):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={
                x: batch_x,
                y: batch_y,
                keep_prob: dropout})

            # Calculate batch loss and accuracy
            loss = sess.run(cost, feed_dict={
                x: batch_x,
                y: batch_y,
                keep_prob: 1.})
            valid_acc = sess.run(accuracy, feed_dict={
                x: mnist.validation.images[:test_valid_size],
                y: mnist.validation.labels[:test_valid_size],
                keep_prob: 1.})

            print('Epoch {:>2}, Batch {:>3} -'
                  'Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(
                epoch + 1,
                batch + 1,
                loss,
                valid_acc))

    # Calculate Test Accuracy
    test_acc = sess.run(accuracy, feed_dict={
        x: mnist.test.images[:test_valid_size],
        y: mnist.test.labels[:test_valid_size],
        keep_prob: 1.})
    print('Testing Accuracy: {}'.format(test_acc))

