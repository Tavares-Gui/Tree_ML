import numpy as np
import pandas as pd
import matplotlib as mtp

from random import random

X = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5]
Y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]

data = list(zip(X, Y))

def giniIndex(data):
    if len(data) == 0:
        return None
    
    p0 = sum([1 for i in data if i[1] == 1]) / len(data),
    p1 = sum([1 for i in data if i[1] == 0]) / len(data)

    return p0 * (1 - p0) + p1 * (1 - p1)
    
def fit(data, depth, minsamples):
    if len(data) == 0:
        return None
    
    qtd0 = 0
    qtd1 = 0
    
    probs = [
        sum([1 for i in data if i[1] == 1]) / len(data),
        sum([1 for i in data if i[1] == 0]) / len(data)
    ]
    
    node = {
        'probs': probs,
        'feature': None,
        'value': None,
        'left': None,
        'right': None
    }
    
    if depth == 0:
        return node
        
    if len(data) < minsamples:
        return node
    
    bestGini = 1000000
    bestX = 0
    
    for i in range(4):
        minX = min(data, lambda row: row[0])[0]
        maxX = max(data, lambda row: row[0])[0]
        
        randX = minX + random() * (maxX - minX)

        XL = [row for row in data if row[0] < randX]
        XR = [row for row in data if row[0] >= randX]
        
        gini = giniIndex(XL) + giniIndex(XR)
        
        if gini < bestGini:
            bestGini = gini
            bestX = randX
    
    feature = 0
    x = bestGini
        
    node['feature'] = feature
    node['value'] = x
    
    for row in data:
        if row[feature] < x:
            XL.append(row)
        else: XR.append(row)
        
    leftNode = fit(XL, depth - 1, minsamples)
    rightNode = fit(XR, depth - 1, minsamples)
    
    node['left'] = leftNode
    node['right'] = rightNode
    
    return node

tree = fit(data, 200, 1)

def predict(tree, x):
    if tree is None:
        return None
    
    if tree['feature'] is None:
        return tree['probs']
        
    if x[tree['feature']] < tree['value']:
        res = predict(tree['left'], x)
        
        if res is None:
            return tree['probs']
        
    return predict(tree['right'], x)

result = tree(tree, [1.7])

print(result)
