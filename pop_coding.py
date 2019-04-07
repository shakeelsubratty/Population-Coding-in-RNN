import numpy
import pandas
from sklearn import preprocessing
import matplotlib as mlp

mlp.use('TkAgg')
from matplotlib import pyplot


def code(x, r, sigma):
    sigma = max(1e-6, sigma)
    exponent = 2

    z = numpy.zeros(len(r), dtype='float')

    # if x is not a number, return z as array of 0s

    z = z + numpy.exp(-(0.5 / sigma ** exponent) * (x-r) ** exponent)
    return z


def decode(c, r):
    v = numpy.sum(c * r) / numpy.sum(c)
    return v


def code_dataframe(input_dataframe, number_of_neurons, sigma, range_start, range_end):
    # load dataset
    input_dataset = input_dataframe.values
    print(input_dataset.shape)
    # scale data
    min_max_scaler = preprocessing.MinMaxScaler()
    dataset_scaled = min_max_scaler.fit_transform(input_dataset)
    dataframe = pandas.DataFrame(dataset_scaled, columns=input_dataframe.columns)

       # print(dataframe)
    # print(dataframe.T.values)
    # print(pandas.DataFrame(numpy.cov(dataframe.T.values)))

    # numpy.savetxt("houseprice_regression/housing_dataset_scaled.csv", dataframe, delimiter=",")

    dataset = dataframe.values
    print(len(dataset[0, :]))
    number_of_features = len(dataset[0, :])

    # split into input (X) and output (Y)
    X = dataset[:, 0:number_of_features - 1 ]
    Y = dataset[:, number_of_features - 1]

    s = numpy.linspace(range_start, range_end, num=number_of_neurons)

    coded_datasets = {}

    # Coding X
    for i in range(len(X[0])):
        coded_datasets[i] = pandas.DataFrame(
            columns=list(range(i * number_of_neurons, (i + 1) * number_of_neurons)))
        x = X[:, i]
        for j in range(len(x)):
            code_for_x = code(x[j], s, sigma)
            coded_datasets[i].loc[j] = code_for_x

    # Coding Y
    coded_datasets[len(coded_datasets)] = pandas.DataFrame(
        columns=list(range((number_of_features - 1) * number_of_neurons, number_of_features * number_of_neurons)))
    for j in range(len(Y)):
        code_for_y = code(Y[j], s, sigma)
        coded_datasets[len(coded_datasets) - 1].loc[j] = code_for_y

    final_dataframe = coded_datasets[0]

    for i in range(1, len(coded_datasets)):
        final_dataframe = final_dataframe.join(coded_datasets[i])

    print(final_dataframe)
    # plot_dataset(final_dataframe, 0, True)

    # numpy.savetxt("houseprice_regression/housing_target_prices_coded.csv", coded_datasets[len(coded_datasets) - 1], delimiter=",")

    return [final_dataframe, min_max_scaler]


def decode_prediction(input_prediction, number_of_neurons, range_start, range_end):
    s = numpy.linspace(range_start, range_end, num=number_of_neurons)
    output = []

    for i in range(len(input_prediction)):
        decoded_value = decode(input_prediction[i], s)
        output.append(decoded_value)
    return output


def plot_dataset(dataset, y, logy):
    dataset.T.plot(y=y, logy=logy)
    pyplot.show()
