#Roger Wang - ryw3 - COMP576

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) #call mnist function

learningRate = .001
trainingIters = 100001
batchSize = 100
displayStep = 20 #in batch sizes

nInput = 28 #we want the input to take the 28 pixels
nSteps = 28 #every 28
nHidden = 32 #number of neurons for the RNN
nClasses = 10 #this is MNIST so you know

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

weights = {
	'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
}

biases = {
	'out': tf.Variable(tf.random_normal([nClasses]))
}

def RNN(x, weights, biases):
	# configuring so you can get it as needed for the 28 pixels
	x = tf.transpose(x, [1, 0, 2])
	x = tf.reshape(x, [-1, nInput])
	x = tf.split(x, nSteps, 0)  # configuring so you can get it as needed for the 28 pixels

	lstmCell = rnn_cell.BasicRNNCell(nHidden)  #find which lstm to use in the documentation

	outputs, states = tf.contrib.rnn.static_rnn(lstmCell, x, dtype=tf.float32)#for the rnn where to get the output and hidden state

	return tf.matmul(outputs[-1], weights['out'])+ biases['out']

pred = RNN(x, weights, biases)

#optimization
#create the cost, optimization, evaluation, and accuracy
#for the cost softmax_cross_entropy_with_logits seems really good
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)

correctPred = tf.equal(tf.math.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

init = tf.initialize_all_variables()


epochs = []
train_acc_list = []
test_acc_list = []
loss_list = []

with tf.Session() as sess:
	sess.run(init)
	step = 1
	testData = mnist.test.images.reshape((-1, nSteps, nInput))
	testLabel = mnist.test.labels
	
	
	while step* batchSize < trainingIters:
		batchX, batchY = mnist.train.next_batch(batchSize)#mnist has a way to get the next batch
		batchX = batchX.reshape((batchSize, nSteps, nInput))

		sess.run(optimizer, feed_dict={x: batchX, y: batchY})

		if step % displayStep == 0:
			acc = accuracy.eval(feed_dict={x: batchX, y: batchY})
			loss =cost.eval(feed_dict={x: batchX, y: batchY})
			print("Iter " + str(step*batchSize) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
			
			
			loss_list.append(loss)
			train_acc_list.append(acc)
			epochs.append(step*batchSize)
			test_acc_list.append(sess.run(accuracy, feed_dict={x: testData, y: testLabel}))
		step +=1
	print('Optimization finished')

	print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: testData, y: testLabel}))
	
	
# Plot train/test accuracy and training loss
plt.title("RNN Accuracy/Loss")
plt.grid()
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.plot(epochs, train_acc_list, label='Training Accuracy')
plt.plot(epochs, test_acc_list, label='Test Accuracy')
plt.legend(loc='upper right')
ax2 = plt.twinx()
ax2.plot(epochs, loss_list, 'k', label='Loss')
ax2.legend(loc='upper left')
ax2.set_ylabel('Loss')

plt.show()