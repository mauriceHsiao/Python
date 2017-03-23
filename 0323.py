# coding=utf-8

# 建立新的類別，自定義類別方法
print "---物件導向---"
class Set:
     # these are the member functions
     # every one takes a first parameter "self" (another convention)
     # that refers to the particular Set object being used
     # 類別初始化
    def __init__(self, values=None):
        """This is the constructor.
         It gets called when you create a new Set.
         You would use it like
         s1 = Set() # empty set
        s2 = Set([1,2,2,3]) # initialize with values"""
        self.dict = {}  # each instance of Set has its own dict property
        # which is what we'll use to track memberships
        if values is not None:
            for value in values:
                self.add(value)
    # 方法
    def __repr__(self):
        """this is the string representation of a Set object
        if you type it at the Python prompt or pass it to str()"""
        return "str(Set): " + str(self.dict.keys())
    # we'll represent membership by being a key in self.dict with value True
    def add(self, value):
        self.dict[value] = True
    # value is in the Set if it's a key in the dictionary
    def contains(self, value):
        return value in self.dict
    def remove(self, value):
        del self.dict[value]

s = Set([1,2,3])
s.add(4)
print s.contains(4) # True
s.remove(3)
print s.contains(3) # False
print str(s)

# 繼承上面那個類別，不會動到原本的類別
print "---類別繼承---"
class Set1(Set):
    def __repr__(self):
        """this is the string representation of a Set object
        if you type it at the Python prompt or pass it to str()"""
        return "My data in Set: " + str(self.dict.keys())
s2=Set1([1,2,3,4])
s2.add(10)
print "str(Set1): "+str(s2)

# function
print "---function用法---"
from functools import partial
def exp(base, power):
    return base ** power
square_of = partial(exp, power=2) # 呼叫exp function
print square_of(3)


print "---呼叫function的方法---"
def double(x):
    return 2 * x
xs = [1, 2, 3, 4]
twice_xs = [double(x) for x in xs] # 第一種
print "doube1: " + str(twice_xs)
twice_xs = map(double, xs) # 第二種
print "doube2: " + str(twice_xs)
list_doubler = partial(map, double) # 第三種
twice_xs = list_doubler(xs) # 再把xs放入
print "doube3: " + str(twice_xs)

# filter 過濾，把不要的過濾掉
# reduce 一個一個拿出來比對
print "---filter & reduce---"
def multiply(x, y): return x * y
products = map(multiply, [1, 2], [4, 5]) # [1 * 4, 2 * 5] = [4, 10]
def is_even(x):
    """True if x is even, False if x is odd"""
    return x % 2 == 0
x_evens = [x for x in xs if is_even(x)] # [2, 4]
x_evens = filter(is_even, xs) # same as above
print x_evens
x_product = reduce(multiply, xs) # = 1 * 2 * 3 * 4 = 24
print x_product

# 枚舉，把資料列舉出來 tuples (index, element)
print "---enumerate---"
documents = ["aa","bb","cc"]
for i, document in enumerate(documents):
    print (i, document) # (0, 'aa')(1, 'bb')(2, 'cc')

# zip 合併
print "---zip合併---"
list1 = ['a', 'b', 'c']
list2 = [1, 2, 3]
list3 = zip(list1, list2)
print list3 # [('a', 1), ('b', 2), ('c', 3)]

letters, numbers = zip(*list3)
print "letters: " + str(letters)
print "numbers: " + str(numbers)

# args & kwargs
# args沒有名稱的參數
# kwargs有參數名稱的參數
print "---args,kwargs---"
def magic(*args, **kwargs):
    print "unnamed args:", args
    print "keyword args:", kwargs
magic(1, 2, 4, 5, 6, 7, key="word", key2="word2",id="123") # 有參數與沒參數
