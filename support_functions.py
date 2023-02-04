import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

def get_iris_data():
    # Setup the grid we will use to visualise the decision boundary
    iris = load_iris()
    x_range = np.linspace(iris.data[:, 0].min(), iris.data[:, 0].max())
    y_range = np.linspace(iris.data[:, 1].min(), iris.data[:, 1].max())
    feature_1, feature_2 = np.meshgrid(x_range, y_range)
    grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T

    # Cast the training data to a pandas dataframe
    features = iris.data[:, :2]
    labels = iris.target
    iris_pd = pd.DataFrame(features).assign(labels=labels).rename(columns={0: 'x', 1: 'y', 'labels': 'class'})
    iris_pd['class'] = iris_pd['class'].astype('str')

    return [grid, iris_pd]