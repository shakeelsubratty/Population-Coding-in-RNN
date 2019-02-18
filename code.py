import numpy
import pandas


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


# load dataset
dataframe = pandas.read_csv("housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values

# split into input (X) and output (Y)
X = dataset[:, 0:13]
Y = dataset[:, 13]

s = numpy.linspace(0.0, 1.0, num=1000)
sigma_main = 0.0015

coded_dataset = pandas.DataFrame(columns=list(range(0,1000)))

print(coded_dataset)

X_0 = X[:, 0]

for i in range(len(X_0)):
    code_for_x = code(X_0[i], s, sigma_main)
    coded_dataset.loc[i] = code_for_x


print(coded_dataset)
# x = X_0[0]
# print(x)


#
# code_for_x = code(x,s,sigma_main)
#
# print(code_for_x)
#
# coded_dataset.loc[0] = code_for_x
#
# print(coded_dataset)
#
# x_decoded = decode(code_for_x, s)
#
# print(x_decoded)
