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

number_of_neurons = 10
sigma = 0.2
range_start = 0.0
range_end = 1.0

# load dataset
dataframe = pandas.read_csv("mpg/mpg_data.csv", delim_whitespace=True, header=None)
dataset = dataframe.values

print(dataframe)
print(dataset)

number_of_features = len(dataset[0, :])

data_scaler = pop_coding.code_dataframe(dataframe, number_of_neurons, sigma, range_start, range_end)
coded_dataframe = data_scaler[0]
min_max_scaler = data_scaler[1]
coded_dataset = coded_dataframe.values

number_of_coded_features = len(coded_dataset[0, :])
number_of_samples = len(coded_dataset[:, 0])

# split into input (X) and output (Y)
X = coded_dataset[:, 0:number_of_coded_features - number_of_neurons]
Y = coded_dataset[:, number_of_coded_features - number_of_neurons:number_of_coded_features]


def baseline_model():
    model = Sequential()
    model.add(Dense(number_of_features-1, input_dim=number_of_features-1, kernel_initializer='normal', activation='relu'))
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


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=population_coding_model, epochs=100, batch_size=5, verbose=1)

# kfold = KFold(n_splits=2, random_state=seed)
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

estimator.fit(X, Y)
prediction = estimator.predict(X)

decoded_prediction = numpy.array(pop_coding.decode_prediction(prediction, number_of_neurons, range_start, range_end))

a = numpy.zeros([number_of_samples, number_of_features])
a[:, number_of_features - 1] = decoded_prediction
rescaled_decoded_prediction = min_max_scaler.inverse_transform(a)[:, number_of_features - 1]

# numpy.savetxt("mgp/mgp_prediction_coded.csv", prediction, delimiter=",")
# numpy.savetxt("mgp/mgp_predection_decoded.csv", decoded_prediction, delimiter=",")
# numpy.savetxt("mgp/mgp_prediction_decoded_rescaled.csv", rescaled_decoded_prediction,
#               delimiter=",")

# Error between rescaled prediction and real values
print(rescaled_decoded_prediction)
print(mean_squared_error(dataset[:, number_of_features - 1], rescaled_decoded_prediction))
#
# X_uncoded = dataset[:, 0:number_of_features-1]
# Y_uncoded = dataset[:, number_of_features-1]
#
# estimator_uncoded = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=1)
#
# estimator_uncoded.fit(X_uncoded, Y_uncoded)
# prediction = estimator_uncoded.predict(X_uncoded)
#
# print(mean_squared_error(dataset[:, number_of_features - 1], prediction))