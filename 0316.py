# coding=utf-8

# Set應用，查詢快、不重複
print "---Set應用---"
s = set()
s.add(1)
s.add(2)
s.add(2)
x = len(s)
y = 2 in s
z = 3 in s
print "x = " + str(x) # 2
print "y = " + str(y) # True
print "z = " + str(z) # False

stopwords_list = ["a","an","at"] + ["yet", "you"] + ["yet", "you"]
# list 可用來比對每一個值
print "zip in stopwords_list = "+str(zip in stopwords_list)
stopwords_set = set(stopwords_list)
# Set 速度較快，若要測驗裡面有沒有，用set比較快
print "zip in stopwords_set = "+str(zip in stopwords_set)

print "---list與Set個數不同---"
item_list = [1, 2, 3, 1, 2, 3]
num_items = len(item_list) # 6
item_set = set(item_list) # {1, 2, 3}
num_distinct_items = len(item_set) # 3
distinct_item_list = list(item_set) # [1, 2, 3]
print "num_items = "+str(num_items) # 用list個數為6個
print "item_set = "+str(item_set)
print "num_distinct_items = " + str(num_distinct_items) # Set個數為3個，因為set不重復
print "distinct_item_list = " + str(distinct_item_list)

# Control Flow
print "---判斷機偶數，if寫法---"
x=2
parity = "even" if x % 2 == 0 else "odd"
print "x =" +str(x)+", and parity = "+parity

# while。倘若說網路連線、email等等，建議不要用，會占用CPU
# 假如跑資料迴圈可以，因為會在記憶體中執行
print "---while迴圈---"
x = 0
while x < 10:
    print x, "is less than 10"
    x += 1

print "---for迴圈---"
for x in range(10):
    print x, "is less than 10"

# sorted 或 sort基本上依樣，但sorted(x) x內容不會改變，x.sort()才會改變
print "---sorted排序---"
x = [4,1,2,3]
print x
y = sorted(x) # is [1,2,3,4], x is unchanged
x.sort() # now x is [1,2,3,4]
print y

#
print "---list應用---"
even_numbers = [x for x in range(5) if x % 2 == 0] # [0, 2, 4]
squares = [x * x for x in range(5)] # [0, 1, 4, 9, 16]
even_squares = [x * x for x in even_numbers] # [0, 4, 16]
print even_numbers,",",squares,",",even_squares

square_dict = { x : x * x for x in range(5) } # { 0:0, 1:1, 2:4, 3:9, 4:16 }
square_set = { x * x for x in [1, -1] } # { 1 }
print square_dict,",",square_set

# Generators 產生器
print "---Generators and Iterators---"
import time
def lazy_range(n):
    i = 0
    while i < n:
        time.sleep(i/10)
        yield i
        i += 1
for i in lazy_range(10):
    print (i)

# 亂數應用
print "---random亂數---"
import random
four_uniform_randoms = [random.random() for _ in range(4)]
print four_uniform_randoms

# 用日期給予seed值，讓亂數每次跑的都不一樣
print "---亂數seed種子---"
import datetime
random.seed(datetime.datetime.now())
print random.random()
random.seed(10)
print random.random()

# 用randrange固定亂數的範圍
print "---randrange固定亂數---"
print random.randrange(10)

# 用shuffle讓他跑出不重複的亂數值
print "---shuffle不重複的亂數值---"
up_to_ten = range(5,11)
random.shuffle(up_to_ten)
print up_to_ten

print "---choice---"
my_lunch = random.choice(["7-11","KFC","Macdonald"])
print my_lunch

print "---rand取多個---"
lottery_numbers = range(60)
winning_numbers = random.sample(lottery_numbers, 6)
print winning_numbers

print "---取多個，但會重複---"
four_with_replacement = [random.choice(range(10))
                            for _ in range(4)]
print four_with_replacement

print "---Regular Expressions正規表達式---"
import re
print all([ # all of these are true, because
            not re.match("a", "cat"), # * 'cat' doesn't start with 'a'
            re.search("a", "cat"), # * 'cat' has an 'a' in it
            not re.search("c", "dog"), # * 'dog' doesn't have a 'c' in it
            3 == len(re.split("[ab]", "carbs")), # a,b做切割['c','r','s']
            "R-D-" == re.sub("[0-9]", "*", "R2D2") # 做取代
            ]) # prints True


print "--------------------CH03------------------------"
print  "---Line Chart---"
from matplotlib import pyplot as plt
years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]
# create a line chart, years on x-axis, gdp on y-axis
plt.plot(years, gdp, color='green', marker='o', linestyle='solid') # or'-'or'dashed'
# add a title
plt.title("Nominal GDP")
# add a label to the y-axis
plt.ylabel("Billions of $")
plt.show()

print "---Bar Chart---"
movies = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
num_oscars = [5, 11, 3, 8, 10]
# bars are by default width 0.8, so we'll add 0.1 to the left coordinates
# so that each bar is centered
xs = [i for i, _ in enumerate(movies)]
# plot bars with left x-coordinates [xs], heights [num_oscars]
plt.bar(xs, num_oscars)
plt.ylabel("# of Academy Awards")
plt.title("My Favorite Movies")
# label x-axis with movie names at bar centers
plt.xticks([i for i, _ in enumerate(movies)], movies)
plt.show()

print "---Bar Chart---"
from collections import Counter
grades = [83,95,91,87,70,0,85,82,100,67,73,77,0]
decile = lambda grade: grade // 10 * 10
histogram = Counter(decile(grade) for grade in grades)
plt.bar([x for x in histogram.keys()], # 偏移 4 ，因此要砍掉
histogram.values(), # give each bar its correct height
8) # give each bar a width of 8
plt.axis([-5, 105, 0, 5]) # x-axis from -5 to 105,
# y-axis from 0 to 5
plt.xticks([10 * i for i in range(11)]) # x-axis labels at 0, 10, ..., 100
plt.xlabel("Decile")
plt.ylabel("# of Students")
plt.title("Distribution of Exam 1 Grades")
plt.show()

print "---Bar Chart---"
mentions = [500, 505]
years = [2013, 2014]
plt.bar(years, mentions, 0.8)
plt.xticks(years)
plt.ylabel("# of times I heard someone say 'data science'")
# if you don't do this, matplotlib will label the x-axis 0, 1
# and then add a +2.013e3 off in the corner (bad matplotlib!)
plt.ticklabel_format(useOffset=False)
# misleading y-axis only shows the part above 500
plt.axis([2012.5,2014.5,499,506]) # 刻度要設小一點才看得比較明顯
plt.title("Look at the 'Huge' Increase!")
plt.show()

print "---Line Chart---"
variance = [1, 2, 4, 8, 16, 32, 64, 128, 256]
bias_squared = [256, 128, 64, 32, 16, 8, 4, 2, 1]
total_error = [x + y for x, y in zip(variance, bias_squared)]
xs = [i for i, _ in enumerate(variance)]
# we can make multiple calls to plt.plot
# to show multiple series on the same chart
plt.plot(xs, variance, 'g-', label='variance') # green solid line
plt.plot(xs, bias_squared, 'r-.', label='bias^2') # red dot-dashed line
plt.plot(xs, total_error, 'b:', label='total error') # blue dotted line
# because we've assigned labels to each series
# we can get a legend for free
# loc=9 means "top center"
plt.legend(loc=9)
plt.xlabel("model complexity")
plt.title("The Bias-Variance Tradeoff")
plt.show()

print "---Scatterplots散點圖---"
friends = [ 70, 65, 72, 63, 71, 64, 60, 64, 67]
minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
plt.scatter(friends, minutes)
# label each point
for label, friend_count, minute_count in zip(labels, friends, minutes):
    plt.annotate(label,
                xy=(friend_count, minute_count), # put the label with its point
                xytext=(5, -5), # but slightly offset
                textcoords='offset points')
plt.title("Daily Minutes vs. Number of Friends")
plt.xlabel("# of friends")
plt.ylabel("daily minutes spent on the site")
#plt.axis('equal') # 轉橫式
plt.show()

