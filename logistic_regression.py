
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

class LogisticRegression:
    def __init__(self, learning_rate=0.02, n_iter=200):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
    
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            output = self.activation(X)
            errors = (y - output)
            self.w_[1:] += self.learning_rate * X.T.dot(errors) / X.shape[0]
            self.w_[0] += self.learning_rate * errors.sum() / errors.shape[0]
            
            cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        return 1.0 / (1.0 + np.exp(-self.net_input(X)))
    
    def predict(self, X):
        return np.where(self.activation(X) >= 0.5, 1, 0)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
    
    def cost(self, X, y):
        return (-(y * np.log(self.activation(X)) + (1 - y) * np.log(1 - self.activation(X)))) / X.shape[0]
    
    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_)), self.cost_)
        plt.ylabel('Cost')
        plt.xlabel('Iterations')
        plt.show()


def train_model(model: LogisticRegression, X: np.ndarray, result: np.ndarray, k: int):
    # normalize X
    std = X.std()
    X = (X - X.mean()) / std


    mapping = {1: '一', 2: '二', 3: '三', 4: '四'}
    y = np.where(result == '第' + mapping[k] + '梯队', 1, 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(y_train)
    model.fit(X_train, y_train)
    #log to file 
    logging.basicConfig(filename='logistic_regression.log', level=logging.INFO)
    logging.info('{}-vs-others classifier'.format(k))
    logging.info('Weights: {}'.format(model.w_))
    logging.info('Original weights: {}'.format(model.w_ * (np.insert(std.values, 0, 1, axis=0))))

    logging.info('y_train: {}'.format(y_train))
    logging.info('Training set score: {}'.format(model.score(X_train, y_train)))

    logging.info('y_test: {}'.format(y_test))
    logging.info('Testing set score: {}\n'.format(model.score(X_test, y_test)))
    
    model.plot_cost()

if __name__ == '__main__':
    # import data from 2565-3-租赁分层.csv
    df = pd.read_csv('./data/2565-3-租赁分层.csv')
    # Remove data from df where '拨备覆盖率(%)' column value is missing
    df = df.dropna(subset=['拨备覆盖率(%)'])


    X = df.iloc[:, 3:]
    result = df['分层结果']

    lg = LogisticRegression(0.2, 200)
    train_model(lg, X, result, 3)







