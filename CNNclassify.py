import sys
import tensorflow as tf
from keras.datasets import cifar10
import numpy as np
import cv2
import os
import re


#TRAINING FUNCTION:
def train():
	# Import data (Samples, 32, 32, 3)
	(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
	yTrain = np.squeeze(yTrain)
	yTest = np.squeeze(yTest)


	# Create the model
	x = tf.placeholder(tf.float32, [None, 32, 32, 3], name = 'input_x')
	y_ = tf.placeholder(tf.int64, [None], name = 'output_y')

	# Variables
	keep_prob = 0.7
	conv1_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 32], mean=0, stddev=0.08))
	conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 64], mean=0, stddev=0.08))
	conv3_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.08))
	W1 = tf.Variable(tf.random_normal([1152, 1500], stddev=0.03))
	b1 = tf.Variable(tf.random_normal([1500], stddev=0.03))
	W2 = tf.Variable(tf.random_normal([1500,500], stddev=0.03))
	b2 = tf.Variable(tf.random_normal([500], stddev=0.03))
	W3 = tf.Variable(tf.random_normal([500, 500], stddev=0.03))
	b3 = tf.Variable(tf.random_normal([500], stddev=0.03))
	W4 = tf.Variable(tf.random_normal([500, 10], stddev=0.03))
	b4 = tf.Variable(tf.random_normal([10], stddev=0.03))
	
	#Conv/Pooling Layers
	conv1 = tf.nn.conv2d(x, conv1_filter, strides = [1, 1, 1, 1], padding='SAME')
	conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
	conv1_bn = tf.layers.batch_normalization(conv1_pool)
	conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides = [1, 1, 1, 1], padding='SAME')
	conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
	conv2_bn = tf.layers.batch_normalization(conv2_pool)
	conv3 = tf.nn.conv2d(conv2_bn, conv3_filter, strides = [1, 1, 1, 1], padding='VALID')
	conv3_pool = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
	conv3_bn = tf.layers.batch_normalization(conv3_pool)

	#Flatten
	flattened = tf.contrib.layers.flatten(conv3_bn)
	print(np.shape(flattened))

	# Hiddien In/Mid/Output layers
	hidden_in = tf.nn.relu(tf.matmul(flattened, W1) + b1)
	hidden_in = tf.nn.dropout(hidden_in, keep_prob)
	hidden_in = tf.layers.batch_normalization(hidden_in)
	hidden_mid = tf.nn.relu(tf.matmul(hidden_in, W2) + b2)
	hidden_mid = tf.nn.dropout(hidden_mid, keep_prob)
	hidden_mid = tf.layers.batch_normalization(hidden_mid)
	hidden_out = tf.nn.relu(tf.matmul(hidden_mid, W3) + b3)
	hidden_out = tf.nn.dropout(hidden_out, keep_prob)
	hidden_out = tf.layers.batch_normalization(hidden_out)

	# Final Output
	y = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W4), b4))
	
	#Define Learning Rate
	lri=-4.5
	lr = 10**lri

	# Define loss and optimizer
	cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(y_, 10), logits=y))
	train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

	sess = tf.InteractiveSession()
	saver = tf.train.Saver()
	tf.global_variables_initializer().run()
	count=0
	# Training loop
	ittermax=240000
	for step in range(ittermax):
		count+=1
		s = np.arange(xTrain.shape[0])
		np.random.shuffle(s)
		xTr = xTrain[s]
		yTr = yTrain[s]
		batch_xs = xTr[:100]
		batch_ys = yTr[:100]
		loss, step = sess.run([cross_entropy, train_step], feed_dict={x: batch_xs, y_: batch_ys})

		# Output progress at 5% increments
		if int(count) % (ittermax/20) == 0:
			percent_completeion = int(count)/(ittermax/100)
			print("percent completion:",percent_completeion)
			print("loss:",loss)
			correct_prediction = tf.equal(tf.argmax(y, 1), y_)
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			print("Accuracy:")
			print(sess.run(accuracy, feed_dict={x: xTest, y_: yTest}))

	# Save model
	save=input("Would you like to save the model?(Y/N)")
	if save=="Y" or save=="y":
		saver.save(sess, 'model', global_step=step)
		
		# Move model files to appropriate folders
		wd=os.getcwd()
		checkpoint_file='checkpoint'
		model_index='model.index'
		for file in os.listdir(wd):
			if re.match('model.data',file):
				model_data=file
		try:
			os.rename(checkpoint_file, 'model/'+checkpoint_file)
		except:
			print('checkpoint file not found!')
		try:
			os.rename(model_index, 'model/'+model_index)
		except:
			print('model_index file not found!')
		try:
			os.rename(model_data, 'model/'+model_data)
		except:
			print('model_data file not found!')
		try:
			os.rename(model.data, 'model/'+model.meta)
		except:
			print('model.meta file not found!')






def test():

	try:
		input_image=str(sys.argv[2])
		# Check if user input_image is a real file before proceeding
		if os.path.isfile(input_image):

			# Restore graph from saved model
			tf.reset_default_graph()
			imported_meta = tf.train.import_meta_graph("model/model.meta")
			graph = tf.get_default_graph()
			saver = tf.train.Saver()

			# Format input image
			xIn = cv2.imread(input_image)
			xIn = np.reshape(xIn, (1, 32, 32, 3))


		# Restore the Variables
			x = tf.placeholder(tf.float32, [None, 32, 32, 3])
			y_ = tf.placeholder(tf.int64, [None])
			conv1_filter = graph.get_tensor_by_name("Variable:0")
			conv2_filter = graph.get_tensor_by_name("Variable_1:0")
			conv3_filter = graph.get_tensor_by_name("Variable_2:0")
			W1 = graph.get_tensor_by_name("Variable_3:0")
			b1 = graph.get_tensor_by_name("Variable_4:0")
			W2 = graph.get_tensor_by_name("Variable_5:0")
			b2 = graph.get_tensor_by_name("Variable_6:0")
			W3 = graph.get_tensor_by_name("Variable_7:0")
			b3 = graph.get_tensor_by_name("Variable_8:0")
			W4 = graph.get_tensor_by_name("Variable_9:0")
			b4 = graph.get_tensor_by_name("Variable_10:0")

			# Restore the model [I realize there must be an easier way to do this]
			conv1 = tf.nn.conv2d(x, conv1_filter, strides = [1, 1, 1, 1], padding='SAME')
			conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
			conv2 = tf.nn.conv2d(conv1_pool, conv2_filter, strides = [1, 1, 1, 1], padding='SAME')
			conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
			conv3 = tf.nn.conv2d(conv2_pool, conv3_filter, strides = [1, 1, 1, 1], padding='VALID')
			conv3_pool = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
			flattened = tf.contrib.layers.flatten(conv3_pool)
			hidden_in = tf.nn.relu(tf.matmul(flattened, W1) + b1)
			hidden_mid = tf.nn.relu(tf.matmul(hidden_in, W2) + b2)
			hidden_out = tf.nn.relu(tf.matmul(hidden_mid, W3) + b3)

			y = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W4), b4))

			# Make the prediction
			with tf.Session() as sess:
				imported_meta.restore(sess, tf.train.latest_checkpoint('model/'))
				conv_image=(conv1)
				output=(tf.argmax(y, 1))
				guess=sess.run(output, feed_dict={x: xIn})
				image_array=sess.run(conv_image, feed_dict={x: xIn})
#				print('image array = ')
#				print(np.shape(image_array))
				# Change values of all pixels to be between 0 to 255
				for i in range(32):
					# Take max and min values for each of the 32 images
					max_val = float(image_array[0][0][0][i])
					min_val = float(image_array[0][0][0][i])
					for j in range(32):
						for k in range(32):
							if max_val < image_array[0][j][k][i]:
								max_val = float(image_array[0][j][k][i])
							if min_val > image_array[0][j][k][i]:
								min_val = float(image_array[0][j][k][i])
					# Normalize the images between 0 to 255
					for j in range(32):
						for k in range(32):
							image_array[0][j][k][i]=((image_array[0][j][k][i]-min_val)/(max_val-min_val))*255

				# Write the output of the image
				image_name='CONVrslt.png'
				image_array=np.reshape(image_array,(1,1024,32))
				cv2.imwrite(image_name, image_array[0][:][:][:])
				print('Results of Convolution saved as '+image_name)

			# Output Guess	
				if guess == [0]:
					print("Wow seriously??!! Fact: This model never guessed PLANE during all my testing. Never, not once! But for some reason it thinks whatever image you provided looks like a plane")
				if guess == [1]:
					print("That looks like a 2008 Hyundai Sonata GLX. Nah just kidding, its a CAR")
				if guess == [2]:
					print("Is it thanksgiving already? coz i see a turkey! (BIRD)")
				if guess == [3]:
					print("Thats a cute cat!")
				if guess == [4]:
					print("Dear DEER, please don't cross the road when I drive. Thanks")
				if guess == [5]:
					print("HotDOG?")
				if guess == [6]:
					print("Did you know that FROGs can breathe through their skin, completely bypassing their lungs?")
				if guess == [7]:
					print("Was that picture taken near HORSEbarn hill?")
				if guess == [8]:
					print("Free 2 day SHIPping!")
				if guess == [9]:
					print("That looks like a TRUCK to me!")
		else:
			print('file',input,'could not be found in the working directory')



	except:

		# This means that no arg3 was provided

		print('To test a specific image, use the following syntax: python test image.png')

		(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
		yTrain = np.squeeze(yTrain)
		yTest = np.squeeze(yTest)

		# Restopre graph from saved model
		tf.reset_default_graph()
		imported_meta = tf.train.import_meta_graph("model/model.meta")
		graph = tf.get_default_graph()
		saver = tf.train.Saver()

		# Restore the Variables
		x = tf.placeholder(tf.float32, [None, 32, 32, 3])
		y_ = tf.placeholder(tf.int64, [None])
		conv1_filter = graph.get_tensor_by_name("Variable:0")
		conv2_filter = graph.get_tensor_by_name("Variable_1:0")
		conv3_filter = graph.get_tensor_by_name("Variable_2:0")
		W1 = graph.get_tensor_by_name("Variable_3:0")
		b1 = graph.get_tensor_by_name("Variable_4:0")
		W2 = graph.get_tensor_by_name("Variable_5:0")
		b2 = graph.get_tensor_by_name("Variable_6:0")
		W3 = graph.get_tensor_by_name("Variable_7:0")
		b3 = graph.get_tensor_by_name("Variable_8:0")
		W4 = graph.get_tensor_by_name("Variable_9:0")
		b4 = graph.get_tensor_by_name("Variable_10:0")

		# Restore the model [I realize there must be an easier way to do this]
		conv1 = tf.nn.conv2d(x, conv1_filter, strides = [1, 1, 1, 1], padding='SAME')
		conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
		conv2 = tf.nn.conv2d(conv1_pool, conv2_filter, strides = [1, 1, 1, 1], padding='SAME')
		conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
		conv3 = tf.nn.conv2d(conv2_pool, conv3_filter, strides = [1, 1, 1, 1], padding='VALID')
		conv3_pool = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
		flattened = tf.contrib.layers.flatten(conv3_pool)
		hidden_in = tf.nn.relu(tf.matmul(flattened, W1) + b1)
		hidden_mid = tf.nn.relu(tf.matmul(hidden_in, W2) + b2)
		hidden_out = tf.nn.relu(tf.matmul(hidden_mid, W3) + b3)

		y = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W4), b4))

		# Output Accuracy
		with tf.Session() as sess:
			imported_meta.restore(sess, tf.train.latest_checkpoint('model/'))
			correct_prediction = tf.equal(tf.argmax(y, 1), y_)
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			print("Accuracy:")
			print(sess.run(accuracy, feed_dict={x: xTest, y_: yTest}))






def main():
	# Go to training loop
	if str(sys.argv[1])=='train':
		train()
	# Go to test loop
	elif str(sys.argv[1])=='test':
		test()
	#Incorrect input syntax
	else:
		print("Incorrect input argument")

main()