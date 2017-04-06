# coding=utf-8
from __future__ import division
from collections import Counter
import math, random

print "---課本範例---"
both_girls = 0
older_girl = 0
either_girl = 0

def random_kid():
    return random.choice(["boy", "girl"])

random.seed(0)
for _ in range(10000):
    younger = random_kid()
    older = random_kid()
    if older == "girl":
        older_girl += 1
    if older == "girl" and younger == "girl":
        both_girls += 1
    if older == "girl" or younger == "girl":
        either_girl += 1

print "P(both | older):", both_girls / older_girl      # 0.514 ~ 1/2
print "P(both | either): ", both_girls / either_girl   # 0.342 ~ 1/3

print "---修改範例---"
a1=0
a2=0
aboth=0
n=100000
def random_ball():
    return random.choice(["B", "Y"])

random.seed(2)
for _ in range(n):
    get1 = random_ball()
    get2 = random_ball()
    if get1 == "B":
        a1 += 1
    if get1 == "B" and get2 == "B":
        aboth += 1
    if get2 == "B":
        a2 += 1

print "---第一次發生與都發生機率相乘看有沒有獨立---"
print "P(aboth):", aboth / n
print "P(get1): ", a1 / n
print "P(get2): ", a2 / n
print "P(get1,get2): ", a1*a2 / (n*n)
print "P(get1|get2) = p(aboth)/p(get2): ", (aboth / n) / (a2 / n)
print "p(get1|get2)/p(get2) = p(get1)p(get2)/p((get2) = p(get1) : ",a1 / n

print "---常態機率分配---"
from matplotlib import pyplot as plt
def normal_pdf(x, mu=0, sigma=1):
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (sqrt_two_pi * sigma))

# sigma越小，越集中，sigma:變異數，mu:平均值
# 機率分配
def plot_normal_pdfs(plt):
    xs = [x / 10.0 for x in range(-50, 50)]
    plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
    plt.plot(xs,[normal_pdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
    plt.plot(xs,[normal_pdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
    plt.plot(xs,[normal_pdf(x,mu=-1)   for x in xs],'-.',label='mu=-1,sigma=1')
    plt.legend()
    plt.show()
plot_normal_pdfs(plt)

print "---累積分配---"
def normal_cdf(x, mu=0,sigma=1):
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

def plot_normal_cdfs(plt):
    xs = [x / 10.0 for x in range(-50, 50)]
    plt.plot(xs,[normal_cdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
    plt.plot(xs,[normal_cdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
    plt.plot(xs,[normal_cdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
    plt.plot(xs,[normal_cdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
    plt.legend(loc=4) # bottom right
    plt.show()
plot_normal_cdfs(plt)

print "---假設檢定---"
# Z值
def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
    """find approximate inverse using binary search"""
    # if not standard, compute standard and rescale
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)
    low_z, low_p = -10.0, 0  # normal_cdf(-10) is (very close to) 0
    hi_z, hi_p = 10.0, 1  # normal_cdf(10)  is (very close to) 1
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2  # consider the midpoint
        mid_p = normal_cdf(mid_z)  # and the cdf's value there
        if mid_p < p:
            # midpoint is still too low, search above it
            low_z, low_p = mid_z, mid_p
        elif mid_p > p:
            # midpoint is still too high, search below it
            hi_z, hi_p = mid_z, mid_p
        else:
            break
    return mid_z
# 平均數,變異數
def normal_approximation_to_binomial(n, p):
    """finds mu and sigma corresponding to a Binomial(n, p)"""
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma
# 右尾
def normal_upper_bound(probability, mu=0, sigma=1):
    """returns the z for which P(Z <= z) = probability"""
    return inverse_normal_cdf(probability, mu, sigma)
# 左尾
def normal_lower_bound(probability, mu=0, sigma=1):
    """returns the z for which P(Z >= z) = probability"""
    return inverse_normal_cdf(1 - probability, mu, sigma)
# 雙尾
def normal_two_sided_bounds(probability, mu=0, sigma=1):
    """returns the symmetric (about the mean) bounds
    that contain the specified probability"""
    tail_probability = (1 - probability) / 2
    # upper bound should have tail_probability above it
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)
    # lower bound should have tail_probability below it
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)
    return lower_bound, upper_bound
print "---0.5，95%信賴區間---"
mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
print "mu_0", mu_0
print "sigma_0", sigma_0
print "normal_two_sided_bounds(0.95, mu_0, sigma_0)", normal_two_sided_bounds(0.95, mu_0, sigma_0)
print "---0.47，99%信賴區間---"
mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.47)
print "mu_0", mu_0
print "sigma_0", sigma_0
print "normal_two_sided_bounds(0.95, mu_0, sigma_0)", normal_two_sided_bounds(0.99, mu_0, sigma_0)

