# coding=utf-8
from __future__ import division
from collections import Counter
from collections import defaultdict


salaries_and_tenures = [(83000, 8.7), (88000, 8.1),
(48000, 0.7), (76000, 6),
(69000, 6.5), (76000, 7.5),
(60000, 2.5), (83000, 10),
(48000, 1.9), (63000, 4.2),(100000,2.5)]

# 從salaries_and_tenures中取出salary與tenure
print "---salary_by_tenure---"
salary_by_tenure = defaultdict(list)
for salary, tenure in salaries_and_tenures:
    salary_by_tenure[tenure].append(salary)
print salary_by_tenure

# 從salaries_and_tenures中取出薪資與筆數，並算出平均
print "---average_salary_by_tenure---"
average_salary_by_tenure = {
tenure : sum(salaries) / len(salaries)
for tenure, salaries in salary_by_tenure.items()
}
print average_salary_by_tenure

# 將資料做分群，小於2，小於5，大於5
# 用迴圈去跑這兩個門檻值(2,5)
print "---tenure_bucket---"
def tenure_bucket(tenure):
    if tenure < 2:
        return "less than two"
    elif tenure < 5:
        return "between two and five"
    else:
        return "more than five"
print tenure_bucket(5.5)


# 把資料讀出來並做分群
print "---salary_by_tenure_bucket---"
salary_by_tenure_bucket = defaultdict(list)
for salary, tenure in salaries_and_tenures:
    bucket = tenure_bucket(tenure)
    salary_by_tenure_bucket[bucket].append(salary) # {tennure_bucket:[salary]}
print salary_by_tenure_bucket


# 將剛剛分群後的資料取出來做平均
print "---average_salary_by_bucket---"
average_salary_by_bucket = {
tenure_bucket : sum(salaries) / len(salaries)
for tenure_bucket, salaries in salary_by_tenure_bucket.iteritems()
}
print average_salary_by_bucket

interests = [
(0, "Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"),
(0, "Spark"), (0, "Storm"), (0, "Cassandra"),
(1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBase"),
(1, "Postgres"), (2, "Python"), (2, "scikit-learn"), (2, "scipy"),
(2, "numpy"), (2, "statsmodels"), (2, "pandas"), (3, "R"), (3, "Python"),
(3, "statistics"), (3, "regression"), (3, "probability"),
(4, "machine learning"), (4, "regression"), (4, "decision trees"),
(4, "libsvm"), (5, "Python"), (5, "R"), (5, "Java"), (5, "C++"),
(5, "Haskell"), (5, "programming languages"), (6, "statistics"),
(6, "probability"), (6, "mathematics"), (6, "theory"),
(7, "machine learning"), (7, "scikit-learn"), (7, "Mahout"),
(7, "neural networks"), (8, "neural networks"), (8, "deep learning"),
(8, "Big Data"), (8, "artificial intelligence"), (9, "Hadoop"),
(9, "Java"), (9, "MapReduce"), (9, "Big Data")
]

# 把interests指派成user,interest，再轉成英文字小寫在做split()切割
print "---words_and_counts---"
words_and_counts = Counter(word
                            for user, interest in interests
                            for word in interest.lower().split())
# 將剛剛words_and_counts的key,value取出來，並計算次數
for word, count in words_and_counts.most_common():
    if count > 1:
        print word, count

print "----------------------------第二章----------------------------"

# 計算1-5與1-5做相加
print "---計算相加---"
for i in [1, 2, 3, 4, 5]:
    for j in [1, 2, 3, 4, 5]:
        print i, " + ", j," = ",(i + j)
print "done looping"

# 正規表達式
# https://dotblogs.com.tw/johnny/2010/01/25/13301

# 加入division，可以改變預設計算方式
print "---division or not---"
print "division =",5/2 # 2.5
print "no division =",5//2 # 2 (default)

# 宣告函式，並呼叫
print "---declare def---"
def double(x):
    return x * 2

def apply_to_one(f):
    return f(1)
my_double = double # 呼叫前面的函式
x = apply_to_one(my_double) # 2
print x

# 建立lamda並做計算
print "---lamda---"
y = apply_to_one(lambda x: x + 4) # 5
print y

# 建立def並帶參數給他
print "---def帶參數---"
def my_print(message="my default message"):
    print message
my_print('Hello')

# 呼叫def給予參數，沒給參數已預設為主
print "---呼叫def並給予參數---"
def subtract(a=0, b=0):
    return a - b
print subtract(10, 5) # 5
print subtract(0, 5) # -5
print subtract(b=5) # -5

print "---String 處理---"
single_quoted_string = 'data science'
double_quoted_string = "data science"
# tab
tab_string = "\t"
print len(tab_string) # 1

# 呈現\ and t
not_tab_string = r"\t" # represents the characters '\' and 't'
print len(not_tab_string) # is 2

# 註解
multi_line_string = """This is the first line.
and this is the second line
and this is the third line"""

# 例外處理，當跑出例外時print "cannot divide by zero"
print "---try catch例外處理---"
try:
    print 0 / 0
except ZeroDivisionError:
    print "cannot divide by zero"

# list應用
print "---list---"
x = range(10)
zero = x[0]
print zero
one = x[1]
print one
nine = x[-1]
print nine
eight = x[-2]
print eight
x[0] = -1
print x[0]

# in應用
print "---in---"
print 1 in [1, 2, 3] # True
print 0 in [1, 2, 3] # False

# tuple應用
print "---tuple---"
my_list = [1, 2]
my_tuple = (1, 2)
other_tuple = 3, 4
my_list[1] = 3

# try except應用
try:
    my_tuple[1] = 3
except TypeError:
    print "cannot modify a tuple"




