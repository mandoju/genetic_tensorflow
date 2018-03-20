import numpy as np
import pandas as pd
from subprocess import check_output
from sklearn.model_selection import train_test_split
from random import randrange
import tensorflow as tf
from multiprocessing.dummy import Pool as ThreadPool


def input_fn(df, labels):
    feature_cols = {k: tf.constant(df[k].values, shape=[df[k].size, 1]) for k in columns}
    label = tf.constant(labels.values, shape=[labels.size, 1])
    return feature_cols, label


def input_predict(df):
    feature_cols = {k: tf.constant(df[k].values, shape=[df[k].size, 1]) for k in columns}
    return feature_cols


class ObjetoResultado(object):
  def __init__(self, dnn, evaluation):
     self.dnn = dnn
     self.evaluation = evaluation

def classifier_outputs(i):

    dnn = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                     hidden_units=[randrange(29) + 1, randrange(29) + 1,
                                                                   randrange(29) + 1], n_classes=3)

    dnn.fit(input_fn=lambda: input_fn(X_train, y_train), steps=1000)

    #dnn.evaluate(input_fn=lambda: input_fn(X_test, y_test), steps=1)
    return ObjetoResultado(dnn , dnn.evaluate)

#print(check_output(["ls", "./input"]).decode("utf8"))

iris = pd.read_csv(".\input\Iris.csv")
iris.head()

print(iris.shape)

iris.iloc[:, 1:4] = iris.iloc[:, 1:4].astype(np.float32)
iris["Species"] = iris["Species"].map({"Iris-setosa": 0, "Iris-virginica": 1, "Iris-versicolor": 2})

X_train, X_test, y_train, y_test = train_test_split(iris.iloc[:, 1:5], iris["Species"], test_size=0.33, random_state=42)

columns = iris.columns[1:5]

# print(columns)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.values)

feature_columns = [tf.contrib.layers.real_valued_column(k) for k in columns]

classifier = []
ev = []
inputs = [1,2,3,4]
pool = ThreadPool(4)
results = pool.map(classifier_outputs, inputs)
pool.close()
pool.join()


for i in range(5):

    print(" o resultado foi: " + str(results[i].evaluation["accuracy"]))
    #print("o valor " + str(i) + " tem a accuracy: " + str(ev[i]['accuracy']))

print("máximo é:" + str(max(evaluation['accuracy'] for evaluation in ev)))

    # print(pred)

    # print(list(pred))
