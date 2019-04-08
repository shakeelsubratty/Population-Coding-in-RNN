# Small script to generate the plots used in Figures 2.3, 2.4 and 2.5

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
import matplotlib as mlp

mlp.use('TkAgg')
from matplotlib import pyplot
import numpy
import pop_coding
import scipy.stats as stats
import math

sigma = 0.81
reso = 100

for i in range(-20, 20):
    mu = i
    x = numpy.linspace(mu - 3*sigma, mu + 3*sigma, reso)
    pyplot.plot(x, stats.norm.pdf(x, mu, sigma))

value = 3
range_start = -10
range_end = 10

s = numpy.linspace(-20 - 3*sigma , 20 + 3*sigma, 40)
a = pop_coding.code(value, s, sigma)
noise = numpy.random.normal(0.05, value/100, 40)
a = a + noise
pyplot.plot(s,a,'bx')
pyplot.show()