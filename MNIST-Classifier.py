

# Edited by Zachariah Albers
# Added two hidden neural network layers
# Uses util to convert weights into image-convertable arrays


# Documentation at https://www.tensorflow.org/get_started/mnist/beginners

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image

import argparse
import sys
import numpy
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import utils

FLAGS = None

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
hidden1_units = 600
hidden2_units = 400
NUM_CLASSES = 10


def main(_):
	# Import data
	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

	# Create the model
	x = tf.placeholder(tf.float32, [None, 784])

	# Hidden 1
	W2 = tf.get_variable('W2', [IMAGE_PIXELS, hidden1_units], tf.float32, tf.contrib.layers.xavier_initializer())
	b2 = tf.get_variable('b2', [hidden1_units], tf.float32, tf.zeros_initializer())
	hidden1 = tf.nn.relu(tf.matmul(x, W2) + b2)

	# Hidden 2
	W3 = tf.get_variable('W3', [hidden1_units, hidden2_units], tf.float32, tf.contrib.layers.xavier_initializer())
	b3 = tf.get_variable('b3', [hidden2_units], tf.float32, tf.zeros_initializer())
	hidden2 = tf.nn.relu(tf.matmul(hidden1, W3) + b3)

	# Linear
	final_W = tf.get_variable('final_W', [hidden2_units, NUM_CLASSES], tf.float32, tf.contrib.layers.xavier_initializer())
	final_b = tf.get_variable('final_b', [NUM_CLASSES], tf.float32, tf.zeros_initializer())
	logits = tf.nn.relu(tf.matmul(hidden2, final_W) + final_b)


  	# Define loss and optimizer
	y_ = tf.placeholder(tf.float32, [None, 10])


	# So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
	# outputs of 'y', and then average across the batch.
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))

	train_step = tf.train.GradientDescentOptimizer(0.4).minimize(cross_entropy)

	correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
	# Train
	for iter in range(10000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		_, loss, accr = sess.run([train_step, cross_entropy, accuracy], feed_dict={x: batch_xs, y_: batch_ys})
		if iter % 100 == 0:
		    tf.logging.info('Training iter: %d, loss: %.5f, accuracy: %.5f' %
		                    (iter, loss, accr))


	# Test trained model
	tf.logging.info('Test accuracy: %.5f' %
	                (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))


	temp_images = numpy.transpose(sess.run(W2))

	file_y = utils.tile_raster_images(temp_images[:100], (28, 28), (10, 10))
	im = Image.fromarray(file_y)
	im.show()
	# im.save("img2.png","PNG")



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='./data',
	                    help='Directory for storing input data')
	FLAGS, unparsed = parser.parse_known_args()
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
