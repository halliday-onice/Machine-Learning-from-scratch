
import numpy as np
import pandas as pd
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
                        entropy -= -p * np.log2(p)
                  else:
                        entropy = 0
                  
                  return entropy


tree = DecisionTreeClassifierJHN()
print(tree.iris)

