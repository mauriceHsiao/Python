# coding=utf-8
import dateutil.parser
import csv,re

print "-----4/12台灣股市資料檢查型別-----"
data = []
def try_or_none(f):
    """wraps f to return None if f raises an exception
    assumes f takes only one input"""
    def f_or_none(x):
        try:
            return f(x)
        except:
            return None
    return f_or_none

def parse_row(input_row, parsers):
    return [try_or_none(parser)(value) if parser is not None else value
            for value, parser in zip(input_row, parsers)]

def parse_rows_with(reader, parsers):
    """wrap a reader to apply the parsers to each of its rows"""
    for row in reader:
        yield parse_row(row, parsers)

# 2017/04/12 交易量排行
# http://2330.tw/
with open("test.csv", "rb") as f:
    reader = csv.reader(f)
    for line in parse_rows_with(reader, [str, float, float, str,float]):
        data.append(line)

for row in data:
    if any(x is None for x in row):
        print row


print "-----正規表達式檢查類型-----"
# . 代表任意字元
# + 代表1個以上
# \w 代表任意字(除了符號、空白)
# \d 代表任意數字

data2 = []
with open("test2.csv", "rb") as f:
    reader = csv.reader(f)
    for x in reader:
        data2.append(x)

re1 = r'\d+\.\d{2}'
re2 = r'\d+%'
for i in range(len(data2)):
    print data2[i]
    #print re.findall(re1, str(data2[i][2]))
    # if re.findall(re1, str(data2[i][2])) == []:
    #     print "type error",str(data2[i][2])
    # if re.findall(re1, str(data2[i][3])) == []:
    #     print "type error",str(data2[i][3])


print "-----330-412微軟股市最高收盤價與交易量-----"
def try_parse_field(field_name, value, parser_dict):
    """try to parse value using the appropriate function from parser_dict"""
    parser = parser_dict.get(field_name) # None if no such entry
    if parser is not None:
        return try_or_none(parser)(value)
    else:
        return value

def parse_dict(input_dict, parser_dict):
    return { field_name : try_parse_field(field_name, value, parser_dict)
             for field_name, value in input_dict.iteritems() }
# 微軟股價 3/30 - 4/12
# https://finance.yahoo.com/quote/MSFT/history?p=MSFT
with open("stocks.csv", "rb") as f:
    reader = csv.DictReader(f, delimiter=",")
    data = [parse_dict(row, {'date': dateutil.parser.parse,
                             'Volume': float})
            for row in reader]

max_msft_price = max(row["AdjClose"]
                     for row in data
                     )
max_msft_Volume = max(row["Volume"]
                     for row in data
                     )

print "max Microsoft Corporation AdjClose price", max_msft_price
print "max Microsoft Corporation Volume", max_msft_Volume


print "-----PCA Example-----"
# PCA example
import numpy as np
import matplotlib.pyplot as plt

N = 500
xTrue = np.linspace(0, 1000, N)
yTrue = 2 * xTrue
xData = xTrue + np.random.normal(0, 100, N)
yData = yTrue + np.random.normal(0, 100, N)
xData = np.reshape(xData, (N, 1))
yData = np.reshape(yData, (N, 1))
data = np.hstack((xData, yData))

mu = data.mean(axis=0)
data = data - mu
eigenvectors, eigenvalues, V = np.linalg.svd(data.T, full_matrices=False)
projected_data = np.dot(data, eigenvectors)
sigma = projected_data.std(axis=0).mean()
print(eigenvectors) # 特徵向量

fig, ax = plt.subplots()
ax.scatter(xData, yData)
for axis in eigenvectors:
    start, end = mu, mu + sigma * axis
    ax.annotate(
        '', xy=end, xycoords='data',
        xytext=start, textcoords='data',
        arrowprops=dict(facecolor='red', width=2.0))
ax.set_aspect('equal')
#plt.show()