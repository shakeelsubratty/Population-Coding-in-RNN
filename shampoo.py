from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import matplotlib as mlp

mlp.use('TkAgg')
from matplotlib import pyplot
import numpy
import pop_coding

# pop coding parameters
number_of_neurons = 100
sigma = 0.05
range_start = -1.0
range_end = 1.0


# date-time parsing function for loading the dataset
def parser(x):
    return datetime.strptime('190' + x, '%Y-%m')


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# load dataset
series = read_csv('shampoo_sales/shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True,
                  date_parser=parser)

# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)

# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

number_of_features = len(supervised_values[0, :])
number_of_samples = len(supervised_values[:, 0])

# apply population coding to dataset
[supervised_coded, min_max_scaler] = pop_coding.code_dataframe(supervised, number_of_neurons, sigma, range_start,
                                                               range_end)

supervised_coded_values = supervised_coded.values

number_of_coded_features = len(supervised_coded_values[0, :])

# split data into train and test-sets

train_coded = supervised_coded_values[0:-12]
test_coded = supervised_coded_values[-12:]


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:number_of_coded_features - number_of_neurons], train[:,
                                                                     number_of_coded_features - number_of_neurons:number_of_coded_features]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    print(X.shape)
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(number_of_neurons))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        print(i)
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    # print(yhat)
    return yhat


# fit the model
lstm_model = fit_lstm(train_coded, 1, 100, 4)

print(number_of_coded_features - number_of_neurons)
# forecast the entire training dataset to build up state for forecasting
train_coded_reshaped = train_coded[:, 0:number_of_coded_features - number_of_neurons].reshape(len(train_coded), 1,
                                                                                              number_of_coded_features - number_of_neurons)

lstm_model.predict(train_coded_reshaped, batch_size=1)

# walk-forward validation on the test data
predictions = list()
for i in range(len(test_coded)):
    # make one-step forecast
    # print(test_coded)
    X, y = test_coded[i, 0:number_of_coded_features - number_of_neurons], test_coded[i,
                                                                          number_of_coded_features - number_of_neurons:number_of_coded_features]
    yhat = forecast_lstm(lstm_model, 1, X)

    yhat = pop_coding.decode_prediction(yhat, number_of_neurons, range_start, range_end)
    print(yhat)
    a = numpy.zeros([1, 2])
    a[:, 1] = yhat
    rescaled_decoded_prediction = min_max_scaler.inverse_transform(a)[:, 1]

    # invert differencing
    yhat = inverse_difference(raw_values, rescaled_decoded_prediction[0], len(test_coded) + 1 - i)
    # store forecast
    predictions.append(yhat)
    # expected = raw_values[len(train) + i + 1]
    expected = raw_values[len(train_coded) + i + 1]
    # print('Month=%d, Predicted=%f, Expected=%f' % (i + 1, yhat[0], expected))
    print("Predicted: ", yhat, " Expected: ", expected)

# report performance
rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
pyplot.plot(raw_values[-12:])
pyplot.plot(predictions)
pyplot.show()
