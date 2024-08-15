
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
## I want to make my own tree classifier using entropy bc the best feature to pick is the one to classify
# is the one that gives the most information, ie, the one with the highest entropy



# for the iris dataset and every dataset that I will ever see I will need to calculate the probability 
# of the entire dataset entropy, aka, the overall entropy at a node in the decision tree, I have to consider the probabilities
# of all possible classes

class DecisionTreeClassifierJHN:
      #I am making initially for the iris dataset but with minor changes, it can be used for other
      def __init__(self):
            self.iris = load_iris()
            self.data = pd.DataFrame(self.iris.data, columns=self.iris.feature_names)
            self.target = pd.Series(self.iris.target)
      

            

      def calc_entropy(self, p):
            class_count = np.bincount(self.target) # this bincount method counts the occurence of each element
            len_count = len(self.target) #total count
            probabilities = class_count / len_count
            entropy = 0 #it works like an entropy counter, it
            for p in probabilities:
                  if p !=0:
                        entropy -= p * np.log2(p)
                  else:
                        entropy = 0

            #entropy_outcome = calc_entropy(self.data)
            return entropy
             
            #print(f"Entropy of the dataset: {entropy_outcome}")
      
""" 
It aims to build a decision tree by iteratively 
selecting the best attribute to split the data based on information gain.

Each node represents a test on an attribute, 
and each branch represents a possible outcome of the test. 

The ID3 algorithm then selects the feature that provides the most information about the target 
variable. The decision tree is built top-down, starting with the root node, 
which represents the entire dataset. At each node, the ID3 algorithm selects the attribute that 
provides the most information gain about the target variable. 

A dataset with high entropy is a dataset where the data points are evenly distributed 
across the different categories. A dataset with low entropy is a dataset where the data points are concentrated 
in one or a few categories.

"""

# o output eh da forma {Sepal.Length Sepal.Width Petal.Length Petal.Width }

if __name__ == "__main__":
      tree = DecisionTreeClassifierJHN()
      dataset_entropy = tree.calc_entropy(tree.data)
      print(dataset_entropy)
""" 
I have some interesting information of this output.
When I run this, the output of the dataset is 1.584962500721156. But how is that possible if the
literature says that entropy is [0, 1]? The thing is: the entropy varies with the dataset. For example for example
a dataset that has 2 features(or classes) is indeed [0,1]. 
There is a rule that MaxEntropy = log2(n) as n the number of classes """
