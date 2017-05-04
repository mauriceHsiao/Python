# coding:utf-8
import matplotlib.pyplot as plt
import random,math
from probability import inverse_normal_cdf
from collections import Counter

print "---單維分群畫圖長條圖---"
def bucketize(point, bucket_size):
    """floor the point to the next lower multiple of bucket_size"""
    return bucket_size * math.floor(point / bucket_size)

def make_histogram(points, bucket_size):
    """buckets the points and counts how many in each bucket"""
    return Counter(bucketize(point, bucket_size) for point in points)

def plot_histogram(points, bucket_size, title=""):
    histogram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
    plt.title(title)
    plt.show()

def compare_two_distributions():
    random.seed(0)
    uniform = [random.randrange(-100,101) for _ in range(200)]
    normal = [57 * inverse_normal_cdf(random.random())
              for _ in range(200)]
    # 新增一筆資料
    data1 = [30,18,50,80,10,60,19,64,25,78,12,47,60,13,90,100]
    # 資料畫圖
    plot_histogram(uniform, 10, "Uniform Histogram")
    plot_histogram(normal, 10, "Normal Histogram")
    plot_histogram(data1, 10, "My Data")

#compare_two_distributions()

print "---雙維畫圖---"
from linear_algebra import shape,get_column

def random_normal():
    """returns a random draw from a standard normal distribution"""
    return inverse_normal_cdf(random.random())

xs = [random_normal() for _ in range(1000)]
ys1 = [ x + random_normal() / 2 for x in xs] # 調整x變數讓圖形變化
ys2 = [-x+10 + random_normal() / 2 for x in xs]

def scatter():
    plt.scatter(xs, ys1, marker='.', color='black', label='ys1')
    plt.scatter(xs, ys2, marker='.', color='gray',  label='ys2')
    plt.xlabel('xs')
    plt.ylabel('ys')
    plt.legend(loc=9)
    plt.show()

#scatter()

print "---多維度圖表---"
def make_scatterplot_matrix():
    # first, generate some random data
    num_points = 100
    def random_row():
        row = [None, None, None, None, None, None]
        row[0] = random_normal()
        row[1] = -5 * row[0] + random_normal()
        row[2] = row[0] + row[1] + 5 * random_normal()
        row[3] = 6 if row[2] > -2 else 0
        row[4] = 5 * row[0] + random_normal()
        row[5] = 0
        return row
    random.seed(0)
    data = [random_row()
            for _ in range(num_points)]
    # then plot it
    _, num_columns = shape(data)
    fig, ax = plt.subplots(num_columns, num_columns)
    for i in range(num_columns):
        for j in range(num_columns):
            # scatter column_j on the x-axis vs column_i on the y-axis
            if i != j:
                ax[i][j].scatter(get_column(data, j), get_column(data, i))
            # unless i == j, in which case show the series name
            else:
                ax[i][j].annotate("series " + str(i), (0.5, 0.5),
                                  xycoords='axes fraction',
                                  ha="center", va="center")
            # then hide axis labels except left and bottom charts
            if i < num_columns - 1:
                ax[i][j].xaxis.set_visible(False)
            if j > 0:
                ax[i][j].yaxis.set_visible(False)
    # fix the bottom right and top left axis labels, which are wrong because
    # their charts only have text in them
    ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
    ax[0][0].set_ylim(ax[0][1].get_ylim())
    plt.show()

#make_scatterplot_matrix()

print "---資料檢查---"
import csv,dateutil
data = []
def try_or_none(f):
    """wraps f to return None if f raises an exception
    assumes f takes only one input"""
    def f_or_none(x):
        try: return f(x)
        except: return None
    return f_or_none

def parse_row(input_row, parsers):
    return [try_or_none(parser)(value) if parser is not None else value
            for value, parser in zip(input_row, parsers)]

def parse_rows_with(reader, parsers):
    """wrap a reader to apply the parsers to each of its rows"""
    for row in reader:
        yield parse_row(row, parsers)

with open("comma_delimited_stock_prices.csv", "rb") as f:
    reader = csv.reader(f)
    for line in parse_rows_with(reader, [dateutil.parser.parse, None, float]):
        # 有問題的資料就不要加至data
        if any(x is None for x in line):
            pass
        else:
            data.append(line)
print data

# 把不符合的值印出來
for row in data:
    if any(x is None for x in row):
        print row


print "---資料量測單位的統一---"
from linear_algebra import  make_matrix
from statistics import  standard_deviation, mean

def scale(data_matrix):
    num_rows, num_cols = shape(data_matrix)
    means = [mean(get_column(data_matrix,j))
             for j in range(num_cols)]
    stdevs = [standard_deviation(get_column(data_matrix,j))
              for j in range(num_cols)]
    return means, stdevs

def rescale(data_matrix):
    """rescales the input data so that each column
    has mean 0 and standard deviation 1
    ignores columns with no deviation"""
    means, stdevs = scale(data_matrix)
    def rescaled(i, j):
        if stdevs[j] > 0:
            return (data_matrix[i][j] - means[j]) / stdevs[j]
        else:
            return data_matrix[i][j]
    num_rows, num_cols = shape(data_matrix)
    return make_matrix(num_rows, num_cols, rescaled)

data = [[1, 20, 2],
        [1, 30, 3],
        [1, 40, 4]]
print "original: ", data
print "scale: ", scale(data)
print "rescaled: ", rescale(data)