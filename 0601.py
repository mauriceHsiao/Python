#coding:utf-8
import math, random
from linear_algebra import squared_distance, vector_mean, distance

# KMean class
class KMeans:
    """performs k-means clustering"""
    def __init__(self, k):
        self.k = k  # number of clusters
        self.means = None  # means of clusters

    def classify(self, input):
        """return the index of the cluster closest to the input"""
        return min(range(self.k),
                   key=lambda i: squared_distance(input, self.means[i]))

    def train(self, inputs):
        self.means = random.sample(inputs, self.k)
        assignments = None
        while True:
            # Find new assignments
            new_assignments = map(self.classify, inputs)
            # If no assignments have changed, we're done.
            if assignments == new_assignments:
                return
            # Otherwise keep the new assignments,
            assignments = new_assignments
            for i in range(self.k):
                i_points = [p for p, a in zip(inputs, assignments) if a == i]
                # avoid divide-by-zero if i_points is empty
                if i_points:
                    self.means[i] = vector_mean(i_points)



inputs = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],[-49,15],[26,13],[-46,5],[-34,-1],[11,15],[-49,0],[-22,-16],[19,28],[-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]

print "---分成3群---"
# 以...為中心點分群
random.seed(0) # so you get the same results as me
clusterer = KMeans(3)
clusterer.train(inputs)
print "3-means:"
print clusterer.means

print "---分成2群---"
# 以...為中心點分群
random.seed(0)
clusterer = KMeans(2)
clusterer.train(inputs)
print "2-means:"
print clusterer.means

# 分群誤差
def squared_clustering_errors(inputs, k):
    """finds the total squared error from k-means clustering the inputs"""
    clusterer = KMeans(k)
    clusterer.train(inputs)
    means = clusterer.means
    assignments = map(clusterer.classify, inputs)
    return sum(squared_distance(input, means[cluster])
               for input, cluster in zip(inputs, assignments))

# 跑分群，從1群到20群，看他的差異值
print "---跑不同的分群誤差---"
for k in range(1, len(inputs) + 1):
    print k, squared_clustering_errors(inputs, k)

def plot_squared_clustering_errors(plt):
    ks = range(1, len(inputs) + 1)
    errors = [squared_clustering_errors(inputs, k) for k in ks]
    plt.plot(ks, errors)
    plt.xticks(ks)
    plt.xlabel("k")
    plt.ylabel("total squared error")
    plt.show()

print "---分群差異畫圖---"
import matplotlib.pyplot as plt
plot_squared_clustering_errors(plt)

print "---圖片顏色分群---"
import matplotlib.image as mpimg
def recolor_image(input_file, k=5):
    img = mpimg.imread(input_file)
    pixels = [pixel for row in img for pixel in row]
    clusterer = KMeans(k)
    clusterer.train(pixels) # this might take a while

    def recolor(pixel):
        cluster = clusterer.classify(pixel) # index of the closest cluster
        return clusterer.means[cluster]     # mean of the closest cluster

    new_img = [[recolor(pixel) for pixel in row]
               for row in img]
    plt.imshow(new_img)
    plt.axis('off')
    plt.show()

# 5色分群
#recolor_image(r"flower.png")
# 3色分群
#recolor_image(r"flower.png",3)

print "---NLP自然語言處理---"
import matplotlib.pyplot as plt
def plot_resumes(plt):
    data = [ ("big data", 100, 15), ("Hadoop", 95, 25), ("Python", 75, 50),
         ("R", 50, 40), ("machine learning", 80, 20), ("statistics", 20, 60),
         ("data science", 60, 70), ("analytics", 90, 3),
         ("team player", 85, 85), ("dynamic", 2, 90), ("synergies", 70, 0),
         ("actionable insights", 40, 30), ("think out of the box", 45, 10),
         ("self-starter", 30, 50), ("customer focus", 65, 15),
         ("thought leadership", 35, 35)]

    def text_size(total):
        """equals 8 if total is 0, 28 if total is 200"""
        return 8 + total / 200 * 20

    for word, job_popularity, resume_popularity in data:
        plt.text(job_popularity, resume_popularity, word,
                 ha='center', va='center',
                 size=text_size(job_popularity + resume_popularity))
    plt.xlabel("Popularity on Job Postings")
    plt.ylabel("Popularity on Resumes")
    plt.axis([0, 100, 0, 100])
    plt.show()
plot_resumes(plt)


print "---爬網抓文章，用前一個字比對---"
import requests,re
from bs4 import BeautifulSoup
from collections import defaultdict

def fix_unicode(text):
    return text.replace(u"\u2019", "'")

def get_document():
    url = "http://radar.oreilly.com/2010/06/what-is-data-science.html"
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html5lib')
    content = soup.find("div", "article-body")        # find article-body div
    regex = r"[\w']+|[\.]"                            # matches a word or a period
    document = []

    for paragraph in content("p"):
        words = re.findall(regex, fix_unicode(paragraph.text))
        document.extend(words)

    return document

def generate_using_bigrams(transitions):
    current = "."   # this means the next word will start a sentence
    result = []
    while True:
        next_word_candidates = transitions[current]    # bigrams (current, _)
        current = random.choice(next_word_candidates)  # choose one at random
        result.append(current)                         # append it to results
        if current == ".": return " ".join(result)     # if "." we're done

document = get_document()
bigrams = zip(document, document[1:])
transitions = defaultdict(list)

for prev, current in bigrams:
    transitions[prev].append(current)

random.seed(0)
print "bigram sentences"
for i in range(10):
    print i, generate_using_bigrams(transitions)

print "---用前兩個字與後一個字做比對---"
def generate_using_trigrams(starts, trigram_transitions):
    current = random.choice(starts)   # choose a random starting word
    prev = "."                        # and precede it with a '.'
    result = [current]
    while True:
        next_word_candidates = trigram_transitions[(prev, current)]
        next = random.choice(next_word_candidates)
        prev, current = current, next
        result.append(current)
        if current == ".":
            return " ".join(result)

trigrams = zip(document, document[1:], document[2:])
trigram_transitions = defaultdict(list)
starts = []

for prev, current, next in trigrams:
    if prev == ".":              # if the previous "word" was a period
        starts.append(current)   # then this is a start word
    trigram_transitions[(prev, current)].append(next)

print "trigram sentences"
for i in range(10):
    print i, generate_using_trigrams(starts, trigram_transitions)


print "---加入文法增加文章可讀性---"
def is_terminal(token):
    return token[0] != "_"

def expand(grammar, tokens):
    for i, token in enumerate(tokens):
        # ignore terminals
        if is_terminal(token): continue
        # choose a replacement at random
        replacement = random.choice(grammar[token])
        if is_terminal(replacement):
            tokens[i] = replacement
        else:
            tokens = tokens[:i] + replacement.split() + tokens[(i+1):]
        return expand(grammar, tokens)
    # if we get here we had all terminals and are done
    return tokens

def random_y_given_x(x):
    """equally likely to be x + 1, x + 2, ... , x + 6"""
    return x + roll_a_die()

def random_x_given_y(y):
    if y <= 7:
        # if the total is 7 or less, the first die is equally likely to be
        # 1, 2, ..., (total - 1)
        return random.randrange(1, y)
    else:
        # if the total is 7 or more, the first die is equally likely to be
        # (total - 6), (total - 5), ..., 6
        return random.randrange(y - 6, 7)

def gibbs_sample(num_iters=100):
    x, y = 1, 2 # doesn't really matter
    for _ in range(num_iters):
        x = random_x_given_y(y)
        y = random_y_given_x(x)
    return x, y

def roll_a_die():
    return random.choice([1,2,3,4,5,6])

def direct_sample():
    d1 = roll_a_die()
    d2 = roll_a_die()
    return d1, d1 + d2

def generate_sentence(grammar):
    return expand(grammar, ["_S"])

def compare_distributions(num_samples=1000):
    counts = defaultdict(lambda: [0, 0])
    for _ in range(num_samples):
        counts[gibbs_sample()][0] += 1
        counts[direct_sample()][1] += 1
    return counts

# 文法
grammar = {
        "_S"  : ["_NP _VP"],
        "_NP" : ["_N",
                 "_A _NP _P _A _N"],
        "_VP" : ["_V",
                 "_V _NP"],
        "_N"  : ["data science", "Python", "regression"],
        "_A"  : ["big", "linear", "logistic"],
        "_P"  : ["about", "near"],
        "_V"  : ["learns", "trains", "tests", "is"]
    }

print "grammar sentences"
for i in range(10):
    print i, " ".join(generate_sentence(grammar))

print "gibbs sampling"
comparison = compare_distributions()
for roll, (gibbs, direct) in comparison.iteritems():
    print roll, gibbs, direct
