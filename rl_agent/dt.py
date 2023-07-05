from sklearn import tree
from matplotlib import pyplot as plt


class Experience:
    def __init__(self, input, output):
        self.input = input
        self.output = output
    
    def __str__(self):
        return str(self.input + self.output)
    

import numpy as np 
import state

    
class DT:
    def __init__(self):
        self.experiences = []
        self.clf = None
        
    def update(self, ):
        if self.experiences: 
            dim = len(self.experiences[0].input)
            inputs = np.empty((0,dim))
            outputs = np.empty((0,1))
            for exp in self.experiences:
                inputs = np.append(inputs, np.array(exp.input)[np.newaxis, :], axis = 0)
                outputs = np.append(outputs, np.array([[exp.output]]), axis = 0)
            clf = tree.DecisionTreeRegressor(max_depth=5)
            clf = clf.fit(inputs, outputs)
            self.clf = clf

    def add_experience(self, input, output):
        new_exp = Experience(input=input, output=output)  
        exists = False
        for exp in self.experiences:
            if exp.input == input and exp.output == output:
                exists = True
        if not exists: 
            self.experiences.append(new_exp)
        
    def predict(self, input):
        if self.clf:
            prediction = self.clf.predict([np.array(input)])[0]
            return prediction
        else: 
            return 0 
        
