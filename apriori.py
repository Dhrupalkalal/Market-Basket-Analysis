import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('groceries - groceries.csv')
dataset=dataset.drop('Item(s)',axis=1)

transactions = []
for i in range(0, 9835):
  transactions.append([str(dataset.values[i,j]) for j in range(0, 32)])
  
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.004, min_confidence = 0.4, min_lift = 3, min_length = 2, max_length = 2)


results = list(rules)
results

def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

resultsinDataFrame

resultsinDataFrame.nlargest(n = 15, columns = 'Lift')