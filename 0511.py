#coding:utf-8
from __future__ import division


# TP,TN,FP,FN 範例
print "---TP,TN,FP,FN---"
def accuracy(tp, fp, fn, tn):
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct / total

def precision(tp, fp, fn, tn):
    return tp / (tp + fp)

def recall(tp, fp, fn, tn):
    return tp / (tp + fn)

def f1_score(tp, fp, fn, tn):
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)
    return 2 * p * r / (p + r)

print "accuracy(70, 4930, 13930, 981070)", accuracy(70, 4930, 13930, 981070)
print "precision(70, 4930, 13930, 981070)", precision(70, 4930, 13930, 981070)
print "recall(70, 4930, 13930, 981070)", recall(70, 4930, 13930, 981070)
print "f1_score(70, 4930, 13930, 981070)", f1_score(70, 4930, 13930, 981070)

print "---座標與程式語言的分群---"
from collections import Counter
from linear_algebra import distance

def raw_majority_vote(labels):
    votes = Counter(labels)
    winner, _ = votes.most_common(1)[0]
    return winner

def majority_vote(labels):
    """assumes that labels are ordered from nearest to farthest"""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count
                       for count in vote_counts.values()
                       if count == winner_count])
    if num_winners == 1:
        return winner  # unique winner, so return it
    else:
        return majority_vote(labels[:-1])  # try again without the farthest

def knn_classify(k, labeled_points, new_point):
    """each labeled point should be a pair (point, label)"""
    # order the labeled points from nearest to farthest
    by_distance = sorted(labeled_points,
                         key=lambda (point, _): distance(point, new_point))
    # find the labels for the k closest
    k_nearest_labels = [label for _, label in by_distance[:k]]
    # and let them vote
    return majority_vote(k_nearest_labels)

# Data Set
cities = [(-86.75, 33.5666666666667, 'Python'), (-88.25, 30.6833333333333, 'Python'),
          (-112.016666666667, 33.4333333333333, 'Java'), (-110.933333333333, 32.1166666666667, 'Java'),
          (-92.2333333333333, 34.7333333333333, 'R'), (-121.95, 37.7, 'R'), (-118.15, 33.8166666666667, 'Python'),
          (-118.233333333333, 34.05, 'Java'), (-122.316666666667, 37.8166666666667, 'R'), (-117.6, 34.05, 'Python'),
          (-116.533333333333, 33.8166666666667, 'Python'), (-121.5, 38.5166666666667, 'R'),
          (-117.166666666667, 32.7333333333333, 'R'), (-122.383333333333, 37.6166666666667, 'R'),
          (-121.933333333333, 37.3666666666667, 'R'), (-122.016666666667, 36.9833333333333, 'Python'),
          (-104.716666666667, 38.8166666666667, 'Python'), (-104.866666666667, 39.75, 'Python'),
          (-72.65, 41.7333333333333, 'R'), (-75.6, 39.6666666666667, 'Python'), (-77.0333333333333, 38.85, 'Python'),
          (-80.2666666666667, 25.8, 'Java'), (-81.3833333333333, 28.55, 'Java'),
          (-82.5333333333333, 27.9666666666667, 'Java'), (-84.4333333333333, 33.65, 'Python'),
          (-116.216666666667, 43.5666666666667, 'Python'), (-87.75, 41.7833333333333, 'Java'),
          (-86.2833333333333, 39.7333333333333, 'Java'), (-93.65, 41.5333333333333, 'Java'),
          (-97.4166666666667, 37.65, 'Java'), (-85.7333333333333, 38.1833333333333, 'Python'),
          (-90.25, 29.9833333333333, 'Java'), (-70.3166666666667, 43.65, 'R'),
          (-76.6666666666667, 39.1833333333333, 'R'), (-71.0333333333333, 42.3666666666667, 'R'),
          (-72.5333333333333, 42.2, 'R'), (-83.0166666666667, 42.4166666666667, 'Python'),
          (-84.6, 42.7833333333333, 'Python'), (-93.2166666666667, 44.8833333333333, 'Python'),
          (-90.0833333333333, 32.3166666666667, 'Java'), (-94.5833333333333, 39.1166666666667, 'Java'),
          (-90.3833333333333, 38.75, 'Python'), (-108.533333333333, 45.8, 'Python'), (-95.9, 41.3, 'Python'),
          (-115.166666666667, 36.0833333333333, 'Java'), (-71.4333333333333, 42.9333333333333, 'R'),
          (-74.1666666666667, 40.7, 'R'), (-106.616666666667, 35.05, 'Python'),
          (-78.7333333333333, 42.9333333333333, 'R'), (-73.9666666666667, 40.7833333333333, 'R'),
          (-80.9333333333333, 35.2166666666667, 'Python'), (-78.7833333333333, 35.8666666666667, 'Python'),
          (-100.75, 46.7666666666667, 'Java'), (-84.5166666666667, 39.15, 'Java'), (-81.85, 41.4, 'Java'),
          (-82.8833333333333, 40, 'Java'), (-97.6, 35.4, 'Python'), (-122.666666666667, 45.5333333333333, 'Python'),
          (-75.25, 39.8833333333333, 'Python'), (-80.2166666666667, 40.5, 'Python'),
          (-71.4333333333333, 41.7333333333333, 'R'), (-81.1166666666667, 33.95, 'R'),
          (-96.7333333333333, 43.5666666666667, 'Python'), (-90, 35.05, 'R'),
          (-86.6833333333333, 36.1166666666667, 'R'), (-97.7, 30.3, 'Python'), (-96.85, 32.85, 'Java'),
          (-95.35, 29.9666666666667, 'Java'), (-98.4666666666667, 29.5333333333333, 'Java'),
          (-111.966666666667, 40.7666666666667, 'Python'), (-73.15, 44.4666666666667, 'R'),
          (-77.3333333333333, 37.5, 'Python'), (-122.3, 47.5333333333333, 'Python'),
          (-89.3333333333333, 43.1333333333333, 'R'), (-104.816666666667, 41.15, 'Java')]
cities = [([longitude, latitude], language) for longitude, latitude, language in cities]

# 讓附近的哪幾個點與其他城市做排序，判斷離我最近的K個城市的語言是什麼
# 再去預測該城市的語言是什麼，在驗證是否正確，正確就+1
# try several different values for k
for k in [1,2,3,4,5,6,7,8,9]:
    num_correct = 0
    for location, actual_language in cities:
        other_cities = [other_city
                        for other_city in cities
                        if other_city != (location, actual_language)]
        predicted_language = knn_classify(k, other_cities, location)
        if predicted_language == actual_language:
            num_correct += 1
    print k, "neighbor[s]:", num_correct, "correct out of", len(cities)


import matplotlib.pyplot as plt
import plot_state_borders as plt1

def plot_state_borders(plt, color='0.8'):
    pass

def plot_cities():

    # key is language, value is pair (longitudes, latitudes)
    plots = { "Java" : ([], []), "Python" : ([], []), "R" : ([], []) }

    # we want each language to have a different marker and color
    markers = { "Java" : "o", "Python" : "s", "R" : "^" }
    colors  = { "Java" : "r", "Python" : "b", "R" : "g" }

    for (longitude, latitude), language in cities:
        plots[language][0].append(longitude)
        plots[language][1].append(latitude)

    # create a scatter series for each language
    for language, (x, y) in plots.iteritems():
        plt.scatter(x, y, color=colors[language], marker=markers[language],
                          label=language, zorder=10)

        # assume we have a function that does this
    plt1.plot_state_borders(plt,color='0.8')
    plt.legend(loc=0)          # let matplotlib choose the location
    plt.axis([-130,-60,20,55]) # set the axes
    plt.title("Favorite Programming Languages")
    plt.show()
#plot_cities()


import random
from statistics import mean
def random_point(dim):
    return [random.random() for _ in range(dim)]

def random_distances(dim, num_pairs):
    return [distance(random_point(dim), random_point(dim))
            for _ in range(num_pairs)]

dimensions = range(1, 101, 5)
avg_distances = []
min_distances = []
random.seed(0)
for dim in dimensions:
    distances = random_distances(dim, 10000)  # 10,000 random pairs
    avg_distances.append(mean(distances))     # track the average
    min_distances.append(min(distances))      # track the minimum
    print dim, min(distances), mean(distances), min(distances) / mean(distances)


print "---貝式分類法，信件差異---"
from machine_learning import split_data
from collections import Counter, defaultdict
import re,glob,math

def tokenize(message):
    message = message.lower()                       # convert to lowercase
    all_words = re.findall("[a-z0-9']+", message)   # extract the words
    return set(all_words)                           # remove duplicates

def count_words(training_set):
    """training set consists of pairs (message, is_spam)"""
    counts = defaultdict(lambda: [0, 0])
    for message, is_spam in training_set:
        for word in tokenize(message):
            counts[word][0 if is_spam else 1] += 1
    return counts

def word_probabilities(counts, total_spams, total_non_spams, k=0.5):
    """turn the word_counts into a list of triplets
    w, p(w | spam) and p(w | ~spam)"""
    return [(w,
             (spam + k) / (total_spams + 2 * k),
             (non_spam + k) / (total_non_spams + 2 * k))
             for w, (spam, non_spam) in counts.iteritems()]

def get_subject_data(path):
    data = []
    # regex for stripping out the leading "Subject:" and any spaces after it
    subject_regex = re.compile(r"^Subject:\s+")
    # glob.glob returns every filename that matches the wildcarded path
    for fn in glob.glob(path):
        is_spam = "ham" not in fn
        with open(fn, 'r') as file:
            for line in file:
                if line.startswith("Subject:"):
                    subject = subject_regex.sub("", line).strip()
                    data.append((subject, is_spam))
    return data


def spam_probability(word_probs, message):
    message_words = tokenize(message)
    log_prob_if_spam = log_prob_if_not_spam = 0.0
    for word, prob_if_spam, prob_if_not_spam in word_probs:
        # for each word in the message,
        # add the log probability of seeing it
        if word in message_words:
            log_prob_if_spam += math.log(prob_if_spam)
            log_prob_if_not_spam += math.log(prob_if_not_spam)
        # for each word that's not in the message
        # add the log probability of _not_ seeing it
        else:
            log_prob_if_spam += math.log(1.0 - prob_if_spam)
            log_prob_if_not_spam += math.log(1.0 - prob_if_not_spam)
    prob_if_spam = math.exp(log_prob_if_spam)
    prob_if_not_spam = math.exp(log_prob_if_not_spam)
    return prob_if_spam / (prob_if_spam + prob_if_not_spam)

class NaiveBayesClassifier:
    def __init__(self, k=0.5):
        self.k = k
        self.word_probs = []

    def train(self, training_set):
        # count spam and non-spam messages
        num_spams = len([is_spam
                         for message, is_spam in training_set
                         if is_spam])
        num_non_spams = len(training_set) - num_spams
        # run training data through our "pipeline"
        word_counts = count_words(training_set)
        self.word_probs = word_probabilities(word_counts,
                                             num_spams,
                                             num_non_spams,
                                             self.k)

    def classify(self, message):
        return spam_probability(self.word_probs, message)

def p_spam_given_word(word_prob):
    word, prob_if_spam, prob_if_not_spam = word_prob
    return prob_if_spam / (prob_if_spam + prob_if_not_spam)

def train_and_test_model(path):
    data = get_subject_data(path)
    random.seed(0)      # just so you get the same answers as me
    train_data, test_data = split_data(data, 0.75)
    classifier = NaiveBayesClassifier()
    classifier.train(train_data)
    classified = [(subject, is_spam, classifier.classify(subject))
              for subject, is_spam in test_data]
    counts = Counter((is_spam, spam_probability > 0.5) # (actual, predicted)
                     for _, is_spam, spam_probability in classified)
    print counts

    classified.sort(key=lambda row: row[2])
    spammiest_hams = filter(lambda row: not row[1], classified)[-5:]
    hammiest_spams = filter(lambda row: row[1], classified)[:5]
    print "spammiest_hams", spammiest_hams
    print "hammiest_spams", hammiest_spams

    words = sorted(classifier.word_probs, key=p_spam_given_word)

    spammiest_words = words[-5:]
    hammiest_words = words[:5]
    print "spammiest_words", spammiest_words
    print "hammiest_words", hammiest_words
train_and_test_model(r"spam\*\*")


print "---simple linear regression---"
from statistics import mean, correlation, standard_deviation, de_mean
from gradient_descent import minimize_stochastic

def predict(alpha, beta, x_i):
    return beta * x_i + alpha

def error(alpha, beta, x_i, y_i):
    return y_i - predict(alpha, beta, x_i)

def sum_of_squared_errors(alpha, beta, x, y):
    return sum(error(alpha, beta, x_i, y_i) ** 2
               for x_i, y_i in zip(x, y))

def least_squares_fit(x, y):
    """given training values for x and y,
    find the least-squares values of alpha and beta"""
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta

def total_sum_of_squares(y):
    """the total squared variation of y_i's from their mean"""
    return sum(v ** 2 for v in de_mean(y))

def r_squared(alpha, beta, x, y):
    """the fraction of variation in y captured by the model, which equals
    1 - the fraction of variation in y not captured by the model"""

    return 1.0 - (sum_of_squared_errors(alpha, beta, x, y) /
                  total_sum_of_squares(y))

def squared_error(x_i, y_i, theta):
    alpha, beta = theta
    return error(alpha, beta, x_i, y_i) ** 2

def squared_error_gradient(x_i, y_i, theta):
    alpha, beta = theta
    return [-2 * error(alpha, beta, x_i, y_i),  # alpha partial derivative
            -2 * error(alpha, beta, x_i, y_i) * x_i]  # beta partial derivative

num_friends_good = [49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
daily_minutes_good = [68.77,51.25,52.08,38.36,44.54,57.13,51.4,41.42,31.22,34.76,54.01,38.79,47.59,49.1,27.66,41.03,36.73,48.65,28.12,46.62,35.57,32.98,35,26.07,23.77,39.73,40.57,31.65,31.21,36.32,20.45,21.93,26.02,27.34,23.49,46.94,30.5,33.8,24.23,21.4,27.94,32.24,40.57,25.07,19.42,22.39,18.42,46.96,23.72,26.41,26.97,36.76,40.32,35.02,29.47,30.2,31,38.11,38.18,36.31,21.03,30.86,36.07,28.66,29.08,37.28,15.28,24.17,22.31,30.17,25.53,19.85,35.37,44.6,17.23,13.47,26.33,35.02,32.09,24.81,19.33,28.77,24.26,31.98,25.73,24.86,16.28,34.51,15.23,39.72,40.8,26.06,35.76,34.76,16.13,44.04,18.03,19.65,32.62,35.59,39.43,14.18,35.24,40.13,41.82,35.45,36.07,43.67,24.61,20.9,21.9,18.79,27.61,27.21,26.61,29.77,20.59,27.53,13.82,33.2,25,33.1,36.65,18.63,14.87,22.2,36.81,25.53,24.62,26.25,18.21,28.08,19.42,29.79,32.8,35.99,28.32,27.79,35.88,29.06,36.28,14.1,36.63,37.49,26.9,18.58,38.48,24.48,18.95,33.55,14.24,29.04,32.51,25.63,22.22,19,32.73,15.16,13.9,27.2,32.01,29.27,33,13.74,20.42,27.32,18.23,35.35,28.48,9.08,24.62,20.12,35.26,19.92,31.02,16.49,12.16,30.7,31.22,34.65,13.13,27.51,33.2,31.57,14.1,33.42,17.44,10.12,24.42,9.82,23.39,30.93,15.03,21.67,31.09,33.29,22.61,26.89,23.48,8.38,27.81,32.35,23.84]

alpha, beta = least_squares_fit(num_friends_good, daily_minutes_good)
print "alpha", alpha
print "beta", beta
print "predict: num_friends_good=49,daily_minutes_good=" + str(predict(alpha,beta,49))
print "r-squared", r_squared(alpha, beta, num_friends_good, daily_minutes_good)

print "gradient descent:"
# choose random value to start
random.seed(0)
theta = [random.random(), random.random()]
alpha, beta = minimize_stochastic(squared_error,
                                  squared_error_gradient,
                                  num_friends_good,
                                  daily_minutes_good,
                                  theta,
                                  0.0001)
print "alpha", alpha
print "beta", beta
print "predict: num_friends_good=49,daily_minutes_good=" + str(predict(alpha, beta, 49))
print "r-squared", r_squared(alpha, beta, num_friends_good, daily_minutes_good)