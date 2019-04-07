import numpy
import pandas
import pop_coding
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import time
import matplotlib as mlp

mlp.use('TkAgg')
from matplotlib import pyplot

number_of_neurons = 100
sigma = 0.2
range_start = 0.0
range_end = 1.0

# load dataset
dataframe = pandas.read_csv("houseprice_regression/housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values

number_of_features = len(dataset[0, :])

data_scaler = pop_coding.code_dataframe(dataframe, number_of_neurons, sigma, range_start, range_end)
coded_dataframe = data_scaler[0]
min_max_scaler = data_scaler[1]

coded_dataset = coded_dataframe.values
number_of_coded_features = len(coded_dataset[0, :])


# k = 11 when number_of_samples = 506
def k_fold_cv_baseline(k, input_dataset, estimator):
    number_of_samples = len(input_dataset[:, 0])
    bucket_size = int(number_of_samples / k)
    errors = list()
    for i in range(k):
        dataset_validate = input_dataset[i * bucket_size:(i + 1) * bucket_size, :]
        dataset_train_0 = input_dataset[0:i * bucket_size, :]
        dataset_train_1 = input_dataset[(i + 1) * bucket_size:number_of_samples, :]
        dataset_train = numpy.concatenate((dataset_train_0, dataset_train_1))

        X_uncoded_train = dataset_train[:, 0:number_of_features - 1]
        Y_uncoded_train = dataset_train[:, number_of_features - 1]

        estimator.fit(X_uncoded_train, Y_uncoded_train)

        X_uncoded_validate = dataset_validate[:, 0:number_of_features - 1]
        Y_uncoded_validate = dataset_validate[:, number_of_features - 1]

        prediction = estimator_uncoded.predict(X_uncoded_validate)
        error = mean_squared_error(Y_uncoded_validate, prediction)
        print(error)
        errors.append(error)

    return errors


# k = 11 when number_of_samples = 506
def k_fold_cv_population_coded(k, input_dataset, estimator):
    number_of_samples = len(input_dataset[:, 0])
    bucket_size = int(number_of_samples / k)
    errors = list()
    for i in range(k):
        coded_dataset_validate = input_dataset[i * bucket_size:(i + 1) * bucket_size, :]
        coded_dataset_train_0 = input_dataset[0:i * bucket_size, :]
        coded_dataset_train_1 = input_dataset[(i + 1) * bucket_size:number_of_samples, :]
        coded_dataset_train = numpy.concatenate((coded_dataset_train_0, coded_dataset_train_1))

        number_of_samples_train = len(coded_dataset_train[:, 0])
        number_of_samples_validate = len(coded_dataset_validate[:, 0])

        # split training data into input (X) and output (Y)
        X_train = coded_dataset_train[:, 0:number_of_coded_features - number_of_neurons]
        Y_train = coded_dataset_train[:, number_of_coded_features - number_of_neurons:number_of_coded_features]

        # split validation data into input (X) and output (Y)
        X_validate = coded_dataset_validate[:, 0:number_of_coded_features - number_of_neurons]
        Y_validate = coded_dataset_validate[:, number_of_coded_features - number_of_neurons:number_of_coded_features]

        estimator.fit(X_train, Y_train)
        prediction = estimator.predict(X_validate)

        decoded_prediction = numpy.array(
            pop_coding.decode_prediction(prediction, number_of_neurons, range_start, range_end))

        a = numpy.zeros([number_of_samples_validate, number_of_features])
        a[:, number_of_features - 1] = decoded_prediction
        rescaled_decoded_prediction = min_max_scaler.inverse_transform(a)[:, number_of_features - 1]

        # numpy.savetxt("houseprice_regression/housing_prediction_coded.csv", prediction, delimiter=",")
        # numpy.savetxt("houseprice_regression/housing_prediction_decoded.csv", decoded_prediction, delimiter=",")
        # numpy.savetxt("houseprice_regression/housing_prediction_decoded_rescaled.csv", rescaled_decoded_prediction,
        #               delimiter=",")

        print(rescaled_decoded_prediction)

        # Error between rescaled prediction and real values

        Y_validate_decoded = dataset[i * bucket_size:(i + 1) * bucket_size, number_of_features - 1]

        error = mean_squared_error(Y_validate_decoded,
                                   rescaled_decoded_prediction)

        print(error)
        errors.append(error)

    return errors


def baseline_model():
    model = Sequential()
    model.add(
        Dense(number_of_features - 1, input_dim=number_of_features - 1, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# define population_coding model
def population_coding_model():
    # create model
    model = Sequential()
    model.add(
        Dense(number_of_coded_features - number_of_neurons, input_dim=number_of_coded_features - number_of_neurons,
              kernel_initializer='normal', activation='relu'))
    model.add(Dense(number_of_neurons, kernel_initializer='normal'))
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


estimator = KerasRegressor(build_fn=population_coding_model, epochs=100, batch_size=5, verbose=1)

start_time = int(round(time.time()*1000))
population_error = k_fold_cv_population_coded(k=11, input_dataset=coded_dataset, estimator=estimator)
end_time = int(round(time.time()*1000))
print("Time taken in milliseconds:")
print(end_time - start_time)
# estimator_uncoded = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=1)
#
# baseline_error = k_fold_cv_baseline(k=11, input_dataset=dataset, estimator=estimator_uncoded)

results = pandas.DataFrame()
# results['baseline'] = baseline_error
results['population_coding'] = population_error
print(results.describe())
results.boxplot()
pyplot.show()
