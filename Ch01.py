# coding=utf-8
from __future__ import division
from collections import Counter # not loaded by default
from collections import defaultdict

users = [
{ "id": 0, "name": "Hero" },
{ "id": 1, "name": "Dunn" },
{ "id": 2, "name": "Sue" },
{ "id": 3, "name": "Chi" },
{ "id": 4, "name": "Thor" },
{ "id": 5, "name": "Clive" },
{ "id": 6, "name": "Hicks" },
{ "id": 7, "name": "Devin" },
{ "id": 8, "name": "Kate" },
{ "id": 9, "name": "Klein" }
]

friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
(4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

# 將users的資料取出來
for user in users:
    user["friends"] = []

# 將friendships取出來後分成兩類
for i, j in friendships:
    # this works because users[i] is the user whose id is i
    users[i]["friends"].append(users[j]) # add i as a friend of j
    users[j]["friends"].append(users[i]) # add j as a friend of i

# 計算其user['friend']的個數
print "---計算朋友個數---"
def number_of_friends(user):
    """how many friends does _user_ have?"""
    return len(user["friends"]) # length of friend_ids list
total_connections = sum(number_of_friends(user) for user in users) # 24
print ("total_connections = " + str(total_connections))

print "---平均有幾個朋友---"
num_users = len(users)
avg_connections = total_connections / num_users
print ("avg_connections = " + str(avg_connections))

# 建立list存放user_id,number_of_friends
print "---每個id有幾個朋友---"
num_friends_by_id = [(user["id"], number_of_friends(user)) for user in users]
print ("num_friends_by_id = " + str(num_friends_by_id))

# 依照id做朋友的排序
print "---做排序---"
sorted_num_friends_by_id = sorted(num_friends_by_id, # get it sorted
                           key=lambda (user_id, num_friends): num_friends, # by num_friends
                           reverse=True) # largest to smallest
print ("sorted_num_friends_by_id = " + str(sorted_num_friends_by_id))

# 朋友的朋友的關係
print "---哪些人有哪些朋友---"
def friends_of_friend_ids_bad(user):
    # "foaf" is short for "friend of a friend"
    return [foaf["id"]
            for friend in user["friends"] # for each of user's friends
            for foaf in friend["friends"]] # get each of _their_ friends

print [friend["id"] for friend in users[0]["friends"]] # [1, 2]
print [friend["id"] for friend in users[1]["friends"]] # [0, 2, 3]
print [friend["id"] for friend in users[2]["friends"]] # [0, 1, 3]
print friends_of_friend_ids_bad(users[0])

# 朋友的朋友
print "---朋友的朋友---"
def not_the_same(user, other_user):
    """two users are not the same if they have different ids"""
    return user["id"] != other_user["id"]

def not_friends(user, other_user):
    """other_user is not a friend if he's not in user["friends"];
    that is, if he's not_the_same as all the people in user["friends"]"""
    return all(not_the_same(friend, other_user)
               for friend in user["friends"])

def friends_of_friend_ids(user):
    return Counter(foaf["id"]
                   for friend in user["friends"] # for each of my friends
                   for foaf in friend["friends"] # count *their* friends
                   if not_the_same(user, foaf) # who aren't me
                   and not_friends(user, foaf)) # and aren't my friends
print friends_of_friend_ids(users[3]) # Counter({0: 2, 5: 1})

# 相同的興趣
print "---哪些人有相同的興趣---"
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

def data_scientists_who_like(target_interest):
    return [user_id
            for user_id, user_interest in interests
            if user_interest == target_interest]
print data_scientists_who_like("Python")

# 將user id 與 興趣 取出來並加入user_ids_by_interest
print "---那些人跟誰有相同的興趣---"
user_ids_by_interest = defaultdict(list)
for user_id, interest in interests:
    user_ids_by_interest[interest].append(user_id)

# 將user id 與 興趣 取出來並加入interests_by_user_id
interests_by_user_id = defaultdict(list)
for user_id, interest in interests:
    interests_by_user_id[user_id].append(interest)

# 那些人跟誰有相同的興趣
def most_common_interests_with(user):
    return Counter(interested_user_id
                   for interest in interests_by_user_id[user["id"]]
                   for interested_user_id in user_ids_by_interest[interest]
                   if interested_user_id != user["id"])
print most_common_interests_with(users[3])


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

