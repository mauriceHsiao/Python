# coding=utf-8
from __future__ import division
import math, random
from linear_algebra import scalar_multiply

print "---斜率&倒數---"
def sum_of_squares(v):
    """computes the sum of squared elements in v"""
    return sum(v_i ** 2 for v_i in v)

# 兩點間斜率
def difference_quotient(f, x, h):
    return (f(x + h) - f(x)) / h

# 兩點斜率算倒數
def plot_estimated_derivative():

    def square(x):
        return x * x

    def derivative(x):
        return 2 * x

    # 倒數
    derivative_estimate = lambda x: difference_quotient(square, x, h=0.00001)

    # plot to show they're basically the same
    import matplotlib.pyplot as plt
    x = range(-10,10)
    plt.plot(x, map(derivative, x), 'rx')           # red  x
    plt.plot(x, map(derivative_estimate, x), 'b+')  # blue +
    plt.show()

plot_estimated_derivative()

print "---斜率---"
# linear_algebra
def vector_subtract(v, w):
    """subtracts two vectors componentwise"""
    return [v_i - w_i for v_i, w_i in zip(v,w)]
def squared_distance(v, w):
    return sum_of_squares(vector_subtract(v, w))
def distance(v, w):
   return math.sqrt(squared_distance(v, w))

# 找尋最低點
def step(v, direction, step_size):
    """move step_size in the direction from v"""
    return [v_i + step_size * direction_i
            for v_i, direction_i in zip(v, direction)]

def sum_of_squares_gradient(v):
    return [2 * v_i for v_i in v]

v = [random.randint(-10, 10) for i in range(3)]
print "sum of squares & start point :",sum_of_squares(v),v
tolerance = 0.0000001

# 每次都往下找，值到點與點之間差距很小
while True:
    # print v, sum_of_squares(v)
    gradient = sum_of_squares_gradient(v)  # compute the gradient at v
    next_v = step(v, gradient, -0.01)  # take a negative gradient step
    #print str(sum_of_squares(next_v)) + str(next_v)
    if distance(next_v, v) < tolerance:  # stop if we're converging
        break
    v = next_v  # continue if we're not

print "minimum v", v
print "minimum value", sum_of_squares(v)


print "---v最小值方法二---"
# 用陣列給他step值，讓他一開始跳比較大，之後再慢慢縮小
def safe(f):
    """define a new function that wraps f and return it"""
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')         # this means "infinity" in Python
    return safe_f

def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    """use gradient descent to find theta that minimizes target function"""
    no = 0
    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

    theta = theta_0  # set theta to initial value
    target_fn = safe(target_fn)  # safe version of target_fn
    value = target_fn(theta)  # value we're minimizing

    while True:
        gradient = gradient_fn(theta)
        next_thetas = [step(theta, gradient, -step_size)
                       for step_size in step_sizes]

        # choose the one that minimizes the error function
        next_theta = min(next_thetas, key=target_fn)
        next_value = target_fn(next_theta)
        print  str(no)+"  "+str(sum_of_squares(next_theta))+str(next_theta)
        no += 1
        # stop if we're "converging"
        if abs(value - next_value) < tolerance:
            return theta
        else:
            theta, value = next_theta, next_value

v = [random.randint(-10, 10) for i in range(3)]
v = minimize_batch(sum_of_squares, sum_of_squares_gradient, v)

print "minimum v", v
print "minimum value", sum_of_squares(v)


# 未完待續...
# def in_random_order(data):
#     """generator that returns the elements of data in random order"""
#     indexes = [i for i, _ in enumerate(data)]  # create a list of indexes
#     random.shuffle(indexes)                    # shuffle them
#     for i in indexes:                          # return the data in that order
#         yield data[i]
# def minimize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
#     data = zip(x, y)
#     theta = theta_0  # initial guess
#     alpha = alpha_0  # initial step size
#     min_theta, min_value = None, float("inf")  # the minimum so far
#     iterations_with_no_improvement = 0
#
#     # if we ever go 100 iterations with no improvement, stop
#     while iterations_with_no_improvement < 100:
#         value = sum(target_fn(x_i, y_i, theta) for x_i, y_i in data)
#
#         if value < min_value:
#             # if we've found a new minimum, remember it
#             # and go back to the original step size
#             min_theta, min_value = theta, value
#             iterations_with_no_improvement = 0
#             alpha = alpha_0
#         else:
#             # otherwise we're not improving, so try shrinking the step size
#             iterations_with_no_improvement += 1
#             alpha *= 0.9
#
#         # and take a gradient step for each of the data points
#         for x_i, y_i in in_random_order(data):
#             gradient_i = gradient_fn(x_i, y_i, theta)
#             theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))
#
#     return min_theta
#
# v1=[-6,-4,-2,0,2,4,6]
# v2=[-9,-4,-2,0,2,5,7]
# theta = [2,3]
# v = [random.randint(-10, 10) for i in range(3)]
# v = minimize_stochastic(sum_of_squares, sum_of_squares_gradient, v1,v2,theta)
#
