# Import libraries
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from test_packs import get_gradient_convolution, get_gradient_biases 
from graph import Graph
import time
import pickle
import sys
#import matplotlib.pyplot as plt

def get_mnist_data():
    mnist = tf.keras.datasets.mnist
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = np.array(train_X).reshape(len(train_X), 784)
    # Prepend the column of 1s for bias
    N, M = train_X.shape
    all_X = np.ones((N, M))
    all_X[:, :] = train_X

    num_labels = len(np.unique(train_y))
    train_y_eye = np.eye(num_labels)[train_y]  # One liner trick!
    test_y_eye = np.eye(num_labels)[test_y]  # One liner trick!
    # a,b,c,d = train_test_split(all_X, all_Y, test_size=0.00, random_state=0)
    #return (all_X, all_X, all_Y, all_Y)
    return train_X,train_y_eye,test_X,test_y_eye

train_x,train_y,test_x,test_y = get_mnist_data()   
train_x = train_x.reshape(-1, 28, 28, 1)
test_x =  test_x.reshape(-1, 28, 28, 1)
training_iters = 2000
learning_rate = 0.00001 
batch_size = 4000
n_classes = 10
x_size = train_x.shape[1]
y_size = train_y.shape[1]  # Number of outcomes (3 iris flowers)
# Symbols
X = tf.placeholder("float", shape=[None, 28, 28, 1], name="X")
y = tf.placeholder("float", shape=[None, y_size], name="Y")

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x) 

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')


def conv_net(x, weights, biases):  

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.

    convs = []
    convs.append(conv1)
    for i in range(len(weights.keys()) - 3):
        #conv = conv2d(convs[i], weights['wc' + str(i+2)], biases['bc' + str(i+2)]) 
        #conv = maxpool2d(conv, k=2) 
        convs.append( maxpool2d(conv2d(convs[i], weights['wc' + str(i+2)], biases['bc' + str(i+2)]), k=2))
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    last_conv = convs.pop()

   # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    # conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    # conv2 = maxpool2d(conv2, k=2)

    # conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    # conv3 = maxpool2d(conv3, k=2)
    
    # conv7 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    # conv7 = maxpool2d(conv3, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(last_conv, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term. 
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    print(last_conv)
    return out
weights = get_gradient_convolution(sys.argv[2])
biases = get_gradient_biases(sys.argv[2])
pred = conv_net(X, weights, biases)
print(pred)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

#calculate accuracy across all the given images and average them out. 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init) 
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    tempos = []
    #fig, ax = plt.subplots()
    #summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    iter_time = time.time()
    for i in range(training_iters):
        for batch in range(len(train_x)//batch_size):
            
            batch_x = train_x[batch*batch_size:min((batch+1)*batch_size,len(train_x))]
            batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]    
            # Run optimization op (backprop).
            # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={X: batch_x,
                                                              y: batch_y})
            predict,loss, acc = sess.run([pred,cost, accuracy], feed_dict={X: batch_x,
                                                              y: batch_y})
        #print("predict = ")
        
        print("Iter " + str(i) + ", Loss= " + \
                    "{:.6f}".format(loss) + ", Training Accuracy= " + \
                    "{:.5f}".format(acc))
        time_passed = time.time() - iter_time
        print("tempo atual: " + str(time_passed) )
        print("Batch Finished!")
        

        # Calculate accuracy for all 10000 mnist test images
        test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={X: test_x,y : test_y})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        tempos.append(time_passed)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:","{:.5f}".format(test_acc))
        if(time_passed >= 460):
            break
    # summary_writer.close()
    # plt.plot(tempos, test_accuracy, '-', lw=2)
        with open('./graphs/gradient.pckl', 'wb') as save_graph_file:
            save_graph = Graph(tempos,test_loss)
            pickle.dump(save_graph,save_graph_file)
            print('salvei em: ./graphs/gradient.pckl')
    # plt.grid(True)
    # plt.show()
