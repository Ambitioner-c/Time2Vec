import numpy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
import os
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler

from keras import backend as K
import csv
from pandas.core.frame import DataFrame


__scaler = MinMaxScaler(feature_range=(0, 1))


def create_dataset(dataset, look_back):
    # 这里的look_back与timestep相同
    data_x, data_y = [], []
    for j in range(len(dataset) - look_back - 1):
        a = dataset[j: (j + look_back)]
        data_x.append(a)
        data_y.append(dataset[j + look_back])
    return numpy.array(data_x), numpy.array(data_y)


def get_data_list(pathname='../Data/all.csv'):
    id_list = []
    data_list = []
    with open(pathname, 'r') as f:
        for j in csv.reader(f):
            id_list.append(j[0])
            data_list.append(j[1:])

    return id_list, data_list


def get_dataset(data):

    dataset = DataFrame(data)
    # dataframe = pd.read_csv(pathname, usecols=[1], engine='python', skipfooter=3)
    # dataset = dataframe.values
    # 将int变为float
    dataset = dataset.astype('float32')

    # 归一化
    dataset = __scaler.fit_transform(dataset)

    train_size = int(len(dataset) * 0.65)
    train_list = dataset[:train_size]
    test_list = dataset[train_size:]

    # 训练数据太少，look_back不能过大
    # 3, 5, 7, 9
    look_back = 7
    train_x, train_y = create_dataset(train_list, look_back)
    test_x, test_y = create_dataset(test_list, look_back)

    train_x = numpy.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    test_x = numpy.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

    return dataset, train_x, train_y, test_x, test_y


def lstm():
    # creat and fit th LSTM network
    model = Sequential()
    model.add(LSTM(32, input_shape=(None, 1)))
    model.add(Dense(1))
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


def prediction(_id, model, train_x, train_y, test_x, test_y):
    # make predictions
    train_predict = model.predict(train_x)
    test_predict = model.predict(test_x)

    # 反归一化
    train_predict = __scaler.inverse_transform(train_predict)
    train_y = __scaler.inverse_transform(train_y)
    test_predict = __scaler.inverse_transform(test_predict)
    test_y = __scaler.inverse_transform(test_y)

    plt.plot(train_y)
    plt.plot(train_predict[1:])
    plt.savefig('../Image/train_' + _id + '.jpg')
    # plt.show()
    plt.plot(test_y)
    plt.plot(test_predict[1:])
    plt.savefig('../Image/test_' + _id + '.jpg')
    # plt.show()


def get_output(_id, model, dataset):
    # Dense
    model.summary()
    # dense_output = model(input=model.input, output=model.get_layer('dense').output).predict(train_x)
    # print(dense_output)
    get_dense_output = K.function([model.layers[0].input], [model.get_layer('dense').output])
    permute_dense_output = get_dense_output([dataset])[0]
    output = []
    for j in permute_dense_output:
        output.append(j[0])
    with open('../Data/result.csv', 'a') as f:
        writer = csv.writer(f)
        out_list = [_id]
        out_list.extend(output)
        writer.writerow(out_list)


def main(pathname):
    _model = load_model('../Model/Test.h5')
    # _model = lstm()
    _id_list, _data_list = get_data_list(pathname)
    for j in range(len(_data_list)):
        _dataset = None
        _train_x = None
        _train_y = None
        _test_x = None
        _test_y = None
        _dataset, _train_x, _train_y, _test_x, _test_y = get_dataset(_data_list[j])
        # _model.fit(_train_x, _train_y, epochs=100, batch_size=1, verbose=2)

        # prediction(_id_list[j], _model, _train_x, _train_y, _test_x, _test_y)
        get_output(_id_list[j], _model, _dataset)

    _model.save('../Data/Test.h5')


if __name__ == '__main__':
    _pathname = '../Data/all.csv'
    main(_pathname)
