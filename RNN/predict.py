import os
import numpy as np
import tensorflow as tf
import argparse

def load_data(dirname):
    if dirname[-1] != '/':
        dirname=dirname+'/'
    listfile=os.listdir(dirname)
    X = []
    Y = []
    for text in listfile:
        textname = dirname + text
        numbers=[]
        with open(textname, mode = 'r') as t:
            numbers = [float(num) for num in t.read().split()]
            for i in range(len(numbers),25200):
                numbers.extend([0.000])
        landmark_frame=[]
        row=0
        for i in range(0,70):
            landmark_frame.extend(numbers[row:row+84])
            row += 84
        landmark_frame=np.array(landmark_frame)
        landmark_frame=landmark_frame.reshape(-1,84)
        X.append(np.array(landmark_frame))
        Y.append(text)
    X = np.array(X)
    Y = np.array(Y)
    # print(Y)
    x_train = X
    x_train=np.array(x_train)
    return x_train,Y

def load_label():
    listfile=[]
    with open("label.txt",mode='r') as l:
        listfile=[i for i in l.read().split()]
    label = {}
    count = 1
    for l in listfile:
        if "_" in l:
            continue
        label[l] = count
        count += 1
    return label
    
def main(input_data_path, output_data_path):
    comp = 'bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
    mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu'

    cmd = 'GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_cpu \
    --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt'

    output_dir = output_data_path
    x_test, Y = load_data(output_dir)
    new_model = tf.keras.models.load_model('model.h5')
    # new_model.summary()

    labels=load_label()
    xhat = x_test
    print(xhat)

    yhat = new_model.predict(xhat)

    print(yhat)
    predictions = np.array([np.argmax(pred) for pred in yhat])
    rev_labels = dict(zip(list(labels.values()), list(labels.keys())))
    s=0
    filel = np.array(Y)
    txtpath=output_data_path+"result.txt" 
    with open(txtpath, "w") as f:
        for i in predictions:
            f.write(Y[s])
            f.write(" ")
            f.write(rev_labels[i])
            f.write("\n")
            s+=1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Sign language with Mediapipe')
    parser.add_argument("--input_data_path",help=" ")
    parser.add_argument("--output_data_path",help=" ")
    args = parser.parse_args()
    input_data_path = args.input_data_path
    output_data_path = args.output_data_path
    main(input_data_path,output_data_path)
