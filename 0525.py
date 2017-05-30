#coding:utf-8
from __future__ import division
from collections import Counter, defaultdict
import math

def entropy(class_probabilities):
    """given a list of class probabilities, compute the entropy"""
    return sum(-p * math.log(p, 2) for p in class_probabilities if p)

def class_probabilities(labels):
    total_count = len(labels)
    return [count / total_count
            for count in Counter(labels).values()]

def data_entropy(labeled_data):
    labels = [label for _, label in labeled_data]
    probabilities = class_probabilities(labels)
    return entropy(probabilities)

def partition_entropy(subsets):
    """find the entropy from this partition of data into subsets"""
    total_count = sum(len(subset) for subset in subsets)

    return sum(data_entropy(subset) * len(subset) / total_count
               for subset in subsets)

def group_by(items, key_fn):
    """returns a defaultdict(list), where each input item 
    is in the list whose key is key_fn(item)"""
    groups = defaultdict(list)
    for item in items:
        key = key_fn(item)
        groups[key].append(item)
    return groups

def partition_by(inputs, attribute):
    """returns a dict of inputs partitioned by the attribute
    each input is a pair (attribute_dict, label)"""
    return group_by(inputs, lambda x: x[0][attribute])

def partition_entropy_by(inputs,attribute):
    """computes the entropy corresponding to the given partition"""
    partitions = partition_by(inputs, attribute)
    return partition_entropy(partitions.values())

inputs = [
        ({'level':'Senior','lang':'Java','tweets':'no','phd':'no'},   False),
        ({'level':'Senior','lang':'Java','tweets':'no','phd':'yes'},  False),
        ({'level':'Mid','lang':'Python','tweets':'no','phd':'no'},     True),
        ({'level':'Junior','lang':'Python','tweets':'no','phd':'no'},  True),
        ({'level':'Junior','lang':'R','tweets':'yes','phd':'no'},      True),
        ({'level':'Junior','lang':'R','tweets':'yes','phd':'yes'},    False),
        ({'level':'Mid','lang':'R','tweets':'yes','phd':'yes'},        True),
        ({'level':'Senior','lang':'Python','tweets':'no','phd':'no'}, False),
        ({'level':'Senior','lang':'R','tweets':'yes','phd':'no'},      True),
        ({'level':'Junior','lang':'Python','tweets':'yes','phd':'no'}, True),
        ({'level':'Senior','lang':'Python','tweets':'yes','phd':'yes'},True),
        ({'level':'Mid','lang':'Python','tweets':'no','phd':'yes'},    True),
        ({'level':'Mid','lang':'Java','tweets':'yes','phd':'no'},      True),
        ({'level':'Junior','lang':'Python','tweets':'no','phd':'yes'},False)
    ]

for key in ['level','lang','tweets','phd']:
    print key, partition_entropy_by(inputs, key)

print "---比較哪level與tweets哪種比較好?---"
print '---level=senior---'
senior_inputs = [(input, label) for input, label in inputs if input["level"] == "Senior"]
for key in ['lang', 'tweets', 'phd']:
    print key, partition_entropy_by(senior_inputs, key)

print '---level=Junior---'
senior_inputs = [(input, label) for input, label in inputs if input["level"] == "Junior"]
for key in ['lang', 'tweets', 'phd']:
    print key, partition_entropy_by(senior_inputs, key)

print '---tweets=yes---'
senior_inputs = [(input, label) for input, label in inputs if input["tweets"] == "yes"]
for key in ['lang', 'tweets', 'phd']:
    print key, partition_entropy_by(senior_inputs, key)

print '---tweets=no---'
senior_inputs = [(input, label) for input, label in inputs if input["tweets"] == "no"]
for key in ['lang', 'tweets', 'phd']:
    print key, partition_entropy_by(senior_inputs, key)

print '---phd=yes---'
senior_inputs = [(input, label) for input, label in inputs if input["phd"] == "yes"]
for key in ['lang', 'tweets', 'phd']:
    print key, partition_entropy_by(senior_inputs, key)

print '---phd=no---'
senior_inputs = [(input, label) for input, label in inputs if input["phd"] == "no"]
for key in ['lang', 'tweets', 'phd']:
    print key, partition_entropy_by(senior_inputs, key)

print "以上用level去分第一層效果最好"

print "---建立決策樹---"
from functools import partial
def classify(tree, input):
    """classify the input using the given decision tree"""
    # if this is a leaf node, return its value
    if tree in [True, False]:
        return tree
    # otherwise find the correct subtree
    attribute, subtree_dict = tree
    subtree_key = input.get(attribute)  # None if input is missing attribute
    if subtree_key not in subtree_dict:  # if no subtree for key,
        subtree_key = None  # we'll use the None subtree
    subtree = subtree_dict[subtree_key]  # choose the appropriate subtree
    return classify(subtree, input)  # and use it to classify the input

def build_tree_id3(inputs, split_candidates=None):
    # if this is our first pass,
    # all keys of the first input are split candidates
    if split_candidates is None:
        split_candidates = inputs[0][0].keys()
    # count Trues and Falses in the inputs
    num_inputs = len(inputs)
    num_trues = len([label for item, label in inputs if label])
    num_falses = num_inputs - num_trues
    if num_trues == 0:  # if only Falses are left
        return False  # return a "False" leaf
    if num_falses == 0:  # if only Trues are left
        return True  # return a "True" leaf
    if not split_candidates:  # if no split candidates left
        return num_trues >= num_falses  # return the majority leaf
    # otherwise, split on the best attribute
    best_attribute = min(split_candidates,
                         key=partial(partition_entropy_by, inputs))
    partitions = partition_by(inputs, best_attribute)
    new_candidates = [a for a in split_candidates
                      if a != best_attribute]
    # recursively build the subtrees
    subtrees = {attribute: build_tree_id3(subset, new_candidates)
                for attribute, subset in partitions.iteritems()}
    subtrees[None] = num_trues > num_falses  # default case
    return (best_attribute, subtrees)

print "---building the tree---"
tree = build_tree_id3(inputs)
print tree

print "Junior / Java / tweets / no phd", classify(tree,
        { "level" : "Junior",
          "lang" : "Java",
          "tweets" : "yes",
          "phd" : "no"} )

print "Junior / Java / tweets / phd", classify(tree,
        { "level" : "Junior",
                 "lang" : "Java",
                 "tweets" : "yes",
                 "phd" : "yes"} )

print "Intern", classify(tree, { "level" : "Intern" } )
print "Senior", classify(tree, { "level" : "Senior" } )


print "---Neural Networks---"
import random
from linear_algebra import dot

def sigmoid(t):
    return 1 / (1 + math.exp(-t))

def neuron_output(weights, inputs):
    return sigmoid(dot(weights, inputs))

def feed_forward(neural_network, input_vector):
    """takes in a neural network (represented as a list of lists of lists of weights)
    and returns the output from forward-propagating the input"""
    outputs = []
    for layer in neural_network:
        input_with_bias = input_vector + [1]             # add a bias input
        output = [neuron_output(neuron, input_with_bias) # compute the output
                  for neuron in layer]                   # for this layer
        outputs.append(output)                           # and remember it
        # the input to the next layer is the output of this one
        input_vector = output
    return outputs

def backpropagate(network, input_vector, target):
    hidden_outputs, outputs = feed_forward(network, input_vector)
    # the output * (1 - output) is from the derivative of sigmoid
    output_deltas = [output * (1 - output) * (output - target[i])
                     for i, output in enumerate(outputs)]
    # adjust weights for output layer (network[-1])
    for i, output_neuron in enumerate(network[-1]):
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            output_neuron[j] -= output_deltas[i] * hidden_output
    # back-propagate errors to hidden layer
    hidden_deltas = [hidden_output * (1 - hidden_output) *
                     dot(output_deltas, [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]
    # adjust weights for hidden layer (network[0])
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i] * input

raw_digits = [
    """11111
       1...1
       1...1
       1...1
       11111""",

    """..1..
       ..1..
       ..1..
       ..1..
       ..1..""",

    """11111
       ....1
       11111
       1....
       11111""",

    """11111
       ....1
       11111
       ....1
       11111""",

    """1...1
       1...1
       11111
       ....1
       ....1""",

    """11111
       1....
       11111
       ....1
       11111""",

    """11111
       1....
       11111
       1...1
       11111""",

    """11111
       ....1
       ....1
       ....1
       ....1""",

    """11111
       1...1
       11111
       1...1
       11111""",

    """11111
       1...1
       11111
       ....1
       11111"""]

def make_digit(raw_digit):
    return [1 if c == '1' else 0
            for row in raw_digit.split("\n")
            for c in row.strip()]

inputs = map(make_digit, raw_digits)

targets = [[1 if i == j else 0 for i in range(10)]
           for j in range(10)]
print "target: ",targets
# 調整參數
random.seed(0)  # to get repeatable results
input_size = 25  # each input is a vector of length 25
num_hidden = 10  # we'll have 5 neurons in the hidden layer
output_size = 10  # we need 10 outputs for each input

# each hidden neuron has one weight per input, plus a bias weight
hidden_layer = [[random.random() for __ in range(input_size + 1)]
                for __ in range(num_hidden)]

# each output neuron has one weight per hidden neuron, plus a bias weight
output_layer = [[random.random() for __ in range(num_hidden + 1)]
                for __ in range(output_size)]

# the network starts out with random weights
network = [hidden_layer, output_layer]

# 10,000 iterations seems enough to converge
for __ in range(10000):
    for input_vector, target_vector in zip(inputs, targets):
        backpropagate(network, input_vector, target_vector)

def predict(input):
    return feed_forward(network, input)[-1]

for i, input in enumerate(inputs):
    outputs = predict(input)
    print i, [round(p, 2) for p in outputs]

print "---3---"
# print """.@@@.
# ...@@
# ..@@.
# ...@@
# .@@@."""
print [round(x, 2) for x in
       predict([0, 1, 1, 1, 0,  # .@@@.
                0, 0, 0, 1, 1,  # ...@@
                0, 0, 1, 1, 0,  # ..@@.
                0, 0, 0, 1, 1,  # ...@@
                0, 1, 1, 1, 0])  # .@@@.
       ]

print "---8---"
# print """.@@@.
# @..@@
# .@@@.
# @..@@
# .@@@."""
print [round(x, 2) for x in
       predict([0, 1, 1, 1, 0,  # .@@@.
                1, 0, 0, 1, 1,  # @..@@
                0, 1, 1, 1, 0,  # .@@@.
                1, 0, 0, 1, 1,  # @..@@
                0, 1, 1, 1, 0])  # .@@@.
       ]

print "---9---"
print [round(x, 2) for x in
       predict([0, 1, 1, 1, 0,  # .@@@.
                1, 0, 0, 1, 1,  # @..@@
                1, 1, 1, 1, 1,  # .@@@.
                0, 0, 0, 1, 1,  # @..@@
                0, 0, 0, 1, 1])  # .@@@.
       ]