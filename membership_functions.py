import math
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt


class MembershipFunction(ABC):
    def __init__(self, a: float, b: float, y_min: float = 0, y_max: float = 1):
        self.a = a
        self.b = b
        self._y_min = y_min
        self._y_max = y_max

    @property
    def y_min(self) -> float:
        return self._y_min

    @y_min.setter
    def y_min(self, val):
        if val < 0:
            raise ValueError("Min for membership can't be less than 0")
        self._y_min = val

    @property
    def y_max(self) -> float:
        return self._y_max

    @y_max.setter
    def y_max(self, val):
        if val > 1:
            raise ValueError("Max for membership can't be more than 1")
        self.y_max = val

    @abstractmethod
    def calculate_y(self, x: float) -> float:
        pass

    def plot(self, range_x: np.arange = np.arange(10, step=0.1)) -> None:
        figure, axis = plt.subplots()
        data = [self.calculate_y(x) for x in range_x]
        axis.plot(range_x, data)
        plt.show()  # TODO show lines for a, b, ymax, ymin, ...


class Linear(MembershipFunction):
    def __init__(self, a, b):
        super().__init__(a, b)

    def calculate_y(self, x: float) -> float:
        if x < self.a:
            return self.y_min
        if x > self.b:
            return self.y_max
        if self.a < x < self.b:
            return self.y_min + ((self.y_max - self.y_min) / (self.b - self.a)) * (x - self.a)


class Triangle(MembershipFunction):
    def __init__(self, a: float, m: float, b: float):
        super().__init__(a, b)
        self.m = m

    def calculate_y(self, x):
        if (x <= self.a) or (x >= self.b):
            return self.y_min
        if self.a < x < self.m:
            return self.y_min + (self.y_max - self.y_min) / (self.m - self.a) * (x - self.a)
        if self.m <= x < self.b:
            return self.y_max - (self.y_max - self.y_min) / (self.b - self.m) * (x - self.m)


class Trapezoidal(MembershipFunction):
    def __init__(self, a, m1, m2, b):
        super().__init__(a, b)
        self.m1 = m1
        self.m2 = m2

    def calculate_y(self, x) -> float:
        if (x <= self.a) or (x >= self.b):
            return self.y_min
        if self.m1 < x < self.m2:
            return self.y_max
        y_max_minus_min = (self.y_max - self.y_min)
        if self.a < x <= self.m1:
            return self.y_min + (y_max_minus_min / (self.m1 - self.a)) * (x - self.a)
        if self.m2 <= x < self.b:
            return self.y_max - (y_max_minus_min / (self.b - self.m2)) * (x - self.m2)


class S(MembershipFunction):

    def calculate_y(self, x) -> float:
        if x <= self.a:
            return self.y_min
        if x > self.b:
            return self.y_max
        y_max_minus_min = (self.y_max - self.y_min)
        if self.a < x <= (self.a + self.b) / 2:
            return self.y_min + 2 * math.pow((x - self.a) / (self.b - self.a), 2) * y_max_minus_min
        if (self.a + self.b) / 2 < x <= self.b:
            return self.y_min + (1 - 2 * math.pow((self.b - x) / (self.b - self.a), 2)) * y_max_minus_min


class Z(MembershipFunction):
    def __init__(self, a, b):
        super().__init__(a, b)

    def calculate_y(self, x: float) -> float:
        return 1 - S(a=self.a, b=self.b).calculate_y(x)


class Pi(MembershipFunction):
    def __init__(self, a, b, c):  # b is midpoint
        super().__init__(a, b)
        self.c = c

    def calculate_y(self, x: float) -> float:
        if x <= self.b:
            return S(a=self.a, b=self.b).calculate_y(x)
        else:
            return Z(a=self.b, b=self.c).calculate_y(x)


if __name__ == '__main__':
    Linear(a=4, b=6).plot(range_x=np.arange(stop=10, step=0.01))
    Triangle(a=1, b=5, m=3).plot(range_x=np.arange(stop=6, step=0.1))
    Trapezoidal(a=1, b=9, m1=4, m2=6).plot(range_x=np.arange(stop=10, step=0.1))
    S(2, 8).plot(np.arange(10, step=0.0001))
    Z(2, 8).plot(np.arange(10, step=0.0001))
    Pi(2, 5, 8).plot(np.arange(10, step=0.01))
