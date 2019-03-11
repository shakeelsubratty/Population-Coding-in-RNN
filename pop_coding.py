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

    z = z + numpy.exp(-(0.5 / sigma ** exponent) * (x - r) ** exponent)
    return z


def decode(c, r):
    v = numpy.sum(c * r) / numpy.sum(c)
    return v


def code_dataframe(input_dataframe):
    # load dataset
    # dataframe = pandas.read_csv("housing.csv", delim_whitespace=True, header=None)
    input_dataset = input_dataframe.values

    # scale data
    min_max_scaler = preprocessing.MinMaxScaler()
    dataset_scaled = min_max_scaler.fit_transform(input_dataset)
    dataframe = pandas.DataFrame(dataset_scaled, columns=input_dataframe.columns)

    numpy.savetxt("housing_dataset_scaled.csv", dataframe, delimiter=",")

    # print(dataframe)
    #
    # print(pandas.DataFrame(min_max_scaler.inverse_transform(dataset_scaled), columns=input_dataframe.columns))

    dataset = dataframe.values

    # split into input (X) and output (Y)
    X = dataset[:, 0:13]
    Y = dataset[:, 13]

    # print(pandas.DataFrame(X))
    # print(pandas.DataFrame(Y))

    number_of_neurons = 10

    s = numpy.linspace(0.0, 1.0, num=number_of_neurons)
    sigma_main = 0.2

    coded_datasets = {}

    # Coding X
    for i in range(len(X[0])):
        coded_datasets[i] = pandas.DataFrame(
            columns=list(range(i * number_of_neurons, (i + 1) * number_of_neurons)))
        x = X[:, i]
        for j in range(len(x)):
            code_for_x = code(x[j], s, sigma_main)
            coded_datasets[i].loc[j] = code_for_x

    # Coding Y
    coded_datasets[len(coded_datasets)] = pandas.DataFrame(
        columns=list(range(13 * number_of_neurons, 14 * number_of_neurons)))
    for j in range(len(Y)):
        code_for_y = code(Y[j], s, sigma_main)
        coded_datasets[len(coded_datasets) - 1].loc[j] = code_for_y

    final_dataframe = coded_datasets[0]

    for i in range(1, len(coded_datasets)):
        final_dataframe = final_dataframe.join(coded_datasets[i])

    print(final_dataframe)
    plot_dataset(final_dataframe, 0, True)

    numpy.savetxt("housing_target_prices_coded.csv", coded_datasets[len(coded_datasets) - 1], delimiter=",")

    return [final_dataframe, min_max_scaler]


def decode_prediction(input_prediction):
    number_of_neurons = 10
    s = numpy.linspace(0.0, 1.0, num=number_of_neurons)
    output = []

    for i in range(len(input_prediction)):
        decoded_value = decode(input_prediction[i], s)
        # print(decoded_value)
        output.append(decoded_value)
        # print(output)
    return output


def plot_dataset(dataset, y, logy):
    dataset.T.plot(y=y, logy=logy)
    pyplot.show()
