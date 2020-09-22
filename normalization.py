from point import Point
from numpy import mean, var


class DummyNormalizer:
    def fit(self, points):
        pass

    def transform(self, points):
        return points


class SumNormalizer:
    def __init__(self):
        self.sum = []

    def fit(self, points):
        all_coordinates = [p.coordinates for p in points]
        self.sum = []
        values = [0] * len(all_coordinates[0])
        for i in range(len(all_coordinates[0])):
            values[i] = sum([abs(all_coordinates[j][i]) for j in range(len(all_coordinates))])
            self.sum.append(values[i])

    def transform(self, points):
        new = []
        for p in points:
            new_coordinates = p.coordinates
            new_coordinates = [(new_coordinates[i] / self.sum[i]) for i in range(len(p.coordinates))]
            new.append(Point(p.name, new_coordinates, p.label))
        return new


class MinMaxNormalizer:
    def __init__(self):
        self.minimum = []
        self.maximum = []

    def fit(self, points):
        all_coordinates = [p.coordinates for p in points]
        self.minimum = [0]*len(all_coordinates[0])
        self.maximum = [0]*len(all_coordinates[0])
        for i in range(len(all_coordinates[0])):
            self.minimum[i] = min((all_coordinates[j][i]) for j in range(len(all_coordinates)))
            self.maximum[i] = max((all_coordinates[j][i]) for j in range(len(all_coordinates)))

    def transform(self, points):
        new = []
        for p in points:
            new_coordinates = p.coordinates
            new_coordinates = [(new_coordinates[i] - self.minimum[i]) / (self.maximum[i] - self.minimum[i])
                               for i in range(len(p.coordinates))]
            new.append(Point(p.name, new_coordinates, p.label))
        return new


class ZNormalizer:
    def __init__(self):
        self.mean_variance_list = []

    def fit(self, points):
        all_coordinates = [p.coordinates for p in points]
        self.mean_variance_list = []
        for i in range(len(all_coordinates[0])):
            values = [x[i] for x in all_coordinates]
            self.mean_variance_list.append([mean(values), var(values, ddof=1)**0.5])

    def transform(self, points):
        new = []
        for p in points:
            new_coordinates = p.coordinates
            new_coordinates = [(new_coordinates[i] - self.mean_variance_list[i][0]) / self.mean_variance_list[i][1]
                               for i in range(len(p.coordinates))]
            new.append(Point(p.name, new_coordinates, p.label))
        return new
