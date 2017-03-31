#coding: utf-8
from __future__ import division
import math
import matplotlib.pyplot as plt
A = [[1, 2, 3], [4, 5, 6]]
B = [[1, 2], [3, 4], [5, 6]]

print "---矩陣---"
def shape(A):
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols
def get_row(A, i):
    return A[i]
def get_column(A, j):
    return [A_i[j] for A_i in A]

print shape(A)
print get_row(A,0)
print get_column(A,1)

print "---矩陣---"
def make_matrix(num_rows, num_cols, entry_fn):
    return [[entry_fn(i, j) for j in range(num_cols)]
            for i in range(num_rows)]
def myfun(i,j):
    return  i * j
print make_matrix(3,5,myfun)

print "---對角矩陣---"
def is_diagonal(i, j):
    return 1 if i == j else 0
identity_matrix = make_matrix(5, 5, is_diagonal)
print identity_matrix

print "---5號朋友有哪些---"
friendships = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0], # user 0
               [1, 0, 1, 1, 0, 0, 0, 0, 0, 0], # user 1
               [1, 1, 0, 1, 0, 0, 0, 0, 0, 0], # user 2
               [0, 1, 1, 0, 1, 0, 0, 0, 0, 0], # user 3
               [0, 0, 0, 1, 0, 1, 0, 0, 0, 0], # user 4
               [0, 0, 0, 0, 1, 0, 1, 1, 0, 0], # user 5
               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0], # user 6
               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0], # user 7
               [0, 0, 0, 0, 0, 0, 1, 1, 0, 1], # user 8
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]] # user 9
friends_of_five = [i # only need
                    for i, is_friend in enumerate(friendships[5]) # to look at
                    if is_friend] # one row
print friends_of_five

print "---兩矩陣相加---"
A1 = [[1, 2, 3], [4, 5, 6]]
B2 = [[7,8,9],[10,11,12]]
def matrix_add(A, B):
    if shape(A) != shape(B):
        raise ArithmeticError("cannot add matrices with different shapes")
    num_rows, num_cols = shape(A)
    def entry_fn(i, j):
        return A[i][j] + B[i][j]
    return make_matrix(num_rows, num_cols, entry_fn)
print matrix_add(A1,B2)

print "---兩矩陣繪圖---"
def dot(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))
def scalar_multiply(c, v):
    return [c * v_i for v_i in v]
def make_graph_dot_product_as_vector_projection(plt):
    v = [2, 1]
    w = [math.sqrt(.25), math.sqrt(.75)]
    c = dot(v, w)
    vonw = scalar_multiply(c, w)
    o = [0, 0]
    plt.arrow(0, 0, v[0], v[1],
              width=0.002, head_width=.1, length_includes_head=True)
    plt.annotate("v", v, xytext=[v[0] + 0.1, v[1]])
    plt.arrow(0, 0, w[0], w[1],
              width=0.002, head_width=.1, length_includes_head=True)
    plt.annotate("w", w, xytext=[w[0] - 0.1, w[1]])
    plt.arrow(0, 0, vonw[0], vonw[1], length_includes_head=True)
    plt.annotate(u"(vâ¢w)w", vonw, xytext=[vonw[0] - 0.1, vonw[1] + 0.1])
    plt.arrow(v[0], v[1], vonw[0] - v[0], vonw[1] - v[1],
              linestyle='dotted', length_includes_head=True)
    plt.scatter(*zip(v, w, o), marker='.')
    plt.axis('equal')
    plt.show()
print make_graph_dot_product_as_vector_projection(plt)


print "---ch04課堂報告---"
print "---矩陣相加---"
x = [1,2,3]
y = [4,5,6]
print x+y

import numpy as np
a = np.array([1, 2, 3])
b = np.array([2, 4, 6])
print a+b

print("---計算向量長度---")
import numpy as np
a=np.array([1,3,2])
b=np.array([-2,1,-1])
la=np.sqrt(a.dot(a))
lb=np.sqrt(b.dot(b))
print (la,lb)

print("---計算cos---")
cos_angle=a.dot(b)/(la*lb)
print (cos_angle)

print("---計算夾角(單位為π)---")
angle=np.arccos(cos_angle)
print (angle)

angle2=angle*360/2/np.pi
print("----轉換單位為角度----")
print (angle2)

# 第三個範例
print("----乘法運算----")
print("----矩陣相乘----")
import numpy as np

a = np.array([[3, 4], [2, 3]])
b = np.array([[1, 2], [3, 4]])
c = np.mat([[3, 4], [2, 3]])
d = np.mat([[1, 2], [3, 4]])
e = np.dot(a, b)
f = np.dot(c, d)
print (a * b)
print (c * d)
print (e)
print (f)

# 第四個範例
print "---亂數產生---"
import numpy as np
a = np.random.randint(1, 10, (3, 5))
print (a)

# 第五個範例
print "---計算矩陣行列式---"
from numpy import *
a = mat([[1, 2, -1], [3, 0, 1], [4, 2, 1]])
print linalg.det(a)

# 第六個範例
print "---矩陣畫圖---"
import numpy as np
from matplotlib import pyplot
x = np.arange(0, 10, 0.1)
y = np.sin(x)
pyplot.plot(x, y)
pyplot.show()