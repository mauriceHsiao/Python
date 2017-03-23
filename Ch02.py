# coding=utf-8
from __future__ import division

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

