#Roger Wang - ryw3 - COMP576

from scipy import misc
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib as mp


# --------------------------------------------------
# setup

def weight_variable(shape):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''
    
    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''
    
    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    bias_constant = 0.0
    
    return tf.Variable(tf.constant(bias_constant, shape=shape))


def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''
    
    # IMPLEMENT YOUR CONV2D HERE
    
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''
    
    # IMPLEMENT YOUR MAX_POOL_2X2 HERE
    
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


ntrain = 1000 # training images per class
ntest = 100 # testing images per class
nclass = 10 # number of classes
imsize = 28
nchannels = 1
batchsize = 128

Train = np.zeros((ntrain * nclass, imsize, imsize, nchannels))
Test = np.zeros((ntest * nclass, imsize, imsize, nchannels))
LTrain = np.zeros((ntrain * nclass, nclass))
LTest = np.zeros((ntest * nclass, nclass))

itrain = -1
itest = -1
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = 'CIFAR10/Train/%d/Image%05d.png' % (iclass, isample)
        im = misc.imread(path)  # 28 by 28
        im = im.astype(float) / 255
        itrain += 1
        Train[itrain, :, :, 0] = im
        LTrain[itrain, iclass] = 1  # 1-hot lable
    for isample in range(0, ntest):
        path = 'CIFAR10/Test/%d/Image%05d.png' % (iclass, isample)
        im = misc.imread(path)  # 28 by 28
        im = im.astype(float) / 255
        itest += 1
        Test[itest, :, :, 0] = im
        LTest[itest, iclass] = 1  # 1-hot lable

sess = tf.InteractiveSession()

tf_data = tf.placeholder(tf.float32, shape=[None, imsize, imsize, nchannels]) # tf variable for the data, remember shape is [None, width, height, numberOfChannels]
tf_labels = tf.placeholder(tf.float32, shape=[None, nclass]) # tf variable for labels


# --------------------------------------------------
# model
# create your model


def sum_stats(x_list, names):
    """
    takes a list of variables and writes them as needed in tensorboard
    """
    for i in range(len(x_list)):
        x = x_list[i]
        var = names[i]
        with tf.name_scope(var):
            tf.summary.histogram('histogram', x)
            tf.summary.scalar('max', tf.reduce_max(x))
            tf.summary.scalar('min', tf.reduce_min(x))
            tf.summary.scalar('mean', tf.reduce_mean(x))


keep_prob = tf.placeholder(tf.float32)

#Conv layer 1 with 5x5 kernel, 32 filter maps, followed by ReLU
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(tf_data, W_conv1) + b_conv1)




#Max layer subsampling by 2
pool1 = max_pool_2x2(h_conv1)

# #drop random set of activations to prevent overfitting
pool1 = tf.nn.dropout(pool1, keep_prob)


#Conv layer 2 with 5x5 kernel, 64 filter maps, followed by ReLU
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(pool1, W_conv2) + b_conv2)

#Max layer subsampling by 2
pool2 = max_pool_2x2(h_conv2)

# #drop random set of activations to prevent overfitting
pool2 = tf.nn.dropout(pool2, keep_prob)

# Fully Connected layer with 7x7x64 input and output 1024

#initialize weights/biases
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

#apply to 2nd pool
pool2 = tf.reshape(pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(pool2, W_fc1) + b_fc1)

# #drop random set of activations to prevent overfitting
# h_fc1 = tf.nn.dropout(h_fc1, keep_prob) #this caused a drop is overall accuracy


# Fully Connected layer with 1024 input and output 10

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
model_result = tf.matmul(h_fc1, W_fc2) + b_fc2

vars = [W_conv1, b_conv1, h_conv1, pool1, W_conv2, b_conv2, h_conv2, pool2, W_fc1, b_fc1, h_fc1, W_fc2, b_fc2, model_result]
var_names = ["W_conv1", "b_conv1", "h_conv1", "pool1", "W_conv2", "b_conv2", "h_conv2", "pool2", "W_fc1", "b_fc1", "h_fc1", "W_fc2", "b_fc2", "iter_result"]
sum_stats(vars, var_names)

# --------------------------------------------------
# loss
# set up the loss, optimization, evaluation, and accuracy
learning_rate = 1e-4

#softmax regression layer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_labels, logits=model_result))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
tf.summary.scalar("cross_entropy", cross_entropy)

p = tf.equal(tf.math.argmax(model_result, 1), tf.arg_max(tf_labels, 1))
accuracy = tf.reduce_mean(tf.cast(p, tf.float32))
tf.summary.scalar('training accuracy', accuracy)


# --------------------------------------------------
# optimization

sess.run(tf.global_variables_initializer())
batch_xs = np.zeros([batchsize, imsize, imsize, nchannels]) # setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
batch_ys = np.zeros([batchsize, nclass]) # setup as [batchsize, the how many classes]

nsamples = ntrain * nclass

iteration_count = 5000

# Tensorboard stuff
summary = tf.summary.merge_all()
saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter('./results/', sess.graph)

test_acc_list = []
train_acc_list = []
loss_list = []
epochs = []

for i in range(iteration_count):  # try a small iteration size once it works then continue
    perm = np.arange(nsamples)
    np.random.shuffle(perm)
    for j in range(batchsize):
        batch_xs[j, :, :, :] = Train[perm[j], :, :, :]
        batch_ys[j, :] = LTrain[perm[j], :]

    if i % 100 == 0:
    # calculate train accuracy and print it
        summary_out, loss_print, accuracy_print = sess.run([summary, cross_entropy, accuracy], feed_dict={tf_data: batch_xs,
                                                                   tf_labels: batch_ys, keep_prob: 1.0})

        summary_writer.add_summary(summary_out, i)
        summary_writer.flush()
    
        test_acc = accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0})
    
        print("Iter " + str(i) + ", Loss = " + str(loss_print), ", Training Accuracy = " + str(accuracy_print))
        print("test accuracy %g" % test_acc)
        
        epochs.append(i)
        test_acc_list.append(test_acc)
        train_acc_list.append(accuracy_print)
        loss_list .append(loss_print)
        
    optimizer.run(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.85})  # dropout only during training
    
# --------------------------------------------------
# test
print("Final test accuracy: %g" % accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0}))
print("test finished")


# Plot train/test accuracy and training loss
plt.title("Cifar Accuracy")
plt.grid()
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.plot(epochs, train_acc_list, label='Training Accuracy')
plt.plot(epochs, test_acc_list, label='Test Accuracy')
plt.legend(loc='upper right')
plt.show()

plt.title("Training Loss")
plt.grid()
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.plot(epochs, loss_list, "k", label=' Training Loss')
plt.legend(loc='upper right')
plt.show()


# Visualize layer 1 weights
for i in range(32):
    plt.subplot(4, 8, i+1)
    plt.imshow(W_conv1.eval()[:, :, 0, i], cmap='gray')
    plt.title(str(i+1))
    plt.axis('off')
plt.show()


sess.close()