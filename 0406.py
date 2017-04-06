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

