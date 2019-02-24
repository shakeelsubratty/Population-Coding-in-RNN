import numpy
import pandas
import pop_coding
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# load dataset
dataframe = pandas.read_csv("housing.csv", delim_whitespace=True, header=None)
coded_dataframe = pop_coding.code_dataframe(dataframe)
dataset = coded_dataframe.values

# split into input (X) and output (Y)
X = dataset[:, 0:130]
Y = dataset[:, 130:140]


# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(130, input_dim=130, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal'))
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
# estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)

estimator = KerasRegressor(build_fn=baseline_model, epochs=10, batch_size=5, verbose=1)
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=10, batch_size=5, verbose=1)))
# pipeline = Pipeline(estimators)

# kfold = KFold(n_splits=2, random_state=seed)
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

print(len(X))
print(len(Y))

estimator.fit(X, Y)
prediction = estimator.predict(X)
print(prediction)

numpy.savetxt("housing_prediction.csv", prediction, delimiter=",")