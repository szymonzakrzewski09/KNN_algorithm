import statistics
import pandas as pd
import numpy as np
from pandas import DataFrame

pd.set_option('display.max_columns', None)


class KNN:
    _data = None

    def __init__(self, raw_data: str = 'iris.txt', header: str = 'iris-type.txt'):
        self.raw_data = raw_data
        self.header = header

    def make_df(self):
        self._data = pd.read_table(self.raw_data, names=pd.read_table('iris-type.txt', header=None)[0])
        # normalize the data
        self._data = self._normalize(self._data)

    @staticmethod
    def _normalize(dataset: DataFrame):
        df_norm = (dataset - dataset.min()) / (dataset.max() - dataset.min())
        df_norm[dataset.columns[-1]] = dataset[dataset.columns[-1]]
        return df_norm

    @staticmethod
    def calculate_distances(idx: int, df: DataFrame, metric: str = 'euclidean', p_parameter: int = 3):
        coords = df.iloc[idx, :-1]
        distances = []

        if metric == 'euclidean':
            distances = np.sqrt(np.sum((df - coords) ** 2, axis=1))
        elif metric == 'manhattan':
            distances = np.sum(np.abs(df - coords), axis=1)
        elif metric == 'chebyshev':
            distances = np.max(np.abs(df - coords), axis=1)
        elif metric == 'minkowski':
            distances = np.sum(np.abs(df - coords) ** p_parameter, axis=1) ** (1 / p_parameter)

        distances[idx] = np.inf
        return distances

    def knn(self, k_parameter: int, metric: str = 'euclidean', p_parameter: int = 3):
        predictions_of_all = []

        for data_row in range(len(self._data)):
            nearst_point, predict = [], []
            distances = self.calculate_distances(data_row, self._data, metric, p_parameter)
            index_of_smallest_metric = np.argsort(distances)[:k_parameter]

            nearst_point.extend(index_of_smallest_metric.values)
            for _ in nearst_point:
                predict.append(self._data.iloc[_]['class(1=Setosa,2=Versicolour,3=Virginica)'])

            predictions_of_all.append(statistics.mode(predict))

        return predictions_of_all

    def accuracy(self, predictions: list):
        num_matches = np.sum([predictions[i] == list(self._data.iloc[:, -1])[i] for i in range(len(predictions))])
        return 100 * num_matches / len(predictions)

    def one_vs_rest(self, k_tables: list, metric_tables: list, p_parameter: int = 3):
        for k in k_tables:
            for metric in metric_tables:
                print(
                    f' Dla k = {k} oraz metryki {metric} otrzymujemy accuracy: {self.accuracy(self.knn(k, metric, p_parameter))}')


knn = KNN()

knn.make_df()

knn.one_vs_rest([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['minkowski', 'euclidean'], p_parameter=7)
