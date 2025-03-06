import numpy as np
import random
class Perceptron:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        #Added attributes for multiclass predictions
        self.classes = None
        self.perceptrons_ = []

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.)
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    #fitting method for multiclass perceptrons
    def fit_multiclass(self, X, y):
        self.classes = np.unique(y)

        
        #for each level, create a perceptron that predicts its likelyhood to appear across all observations
        for clas in self.classes:
            #not sure if it would be better to just add by uniform amount everytime
            i = random.randint(1,10)
            y_binary = np.where(y == clas, 1, 0)

            p = Perceptron(self.eta, self.n_iter, self.random_state+i)
            p.fit(X, y_binary)
            self.perceptrons_.append(p)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
    
    def predict_multiclass(self, X):
        net_inputs = np.column_stack([
            p.net_input(X) for p in self.perceptrons_
        ])
        max = np.argmax(net_inputs, axis=1)
        return self.classes[max]