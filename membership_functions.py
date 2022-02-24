import math
import re
from abc import ABC, abstractmethod
from functools import wraps
from typing import Dict, Type

import matplotlib.pyplot as plt
import numpy as np


class MembershipFunction(ABC):
    def __init__(self, a: float, b: float, y_min: float = 0, y_max: float = 1):
        self.a = a
        self.b = b
        self.y_min = y_min
        self.y_max = y_max

    REQUIRED_INIT_ARGUMENTS = ['a', 'b']

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
        self._y_max = val

    @abstractmethod
    def calculate_y(self, x: float) -> float:
        pass

    def plot(self, *args, details: int = 25) -> None:
        """
        Plots function and its parameters in apropriate range.
        :param details: the amount of evenly spaced x-values computed e.g: details=1 => 10 values for a range of 10
        :param args: start and stop parameter for the range of x values
        """
        start, stop = 0, 10
        if len(args) == 0:
            d = abs(self.b - self.a) / 5
            start = math.floor(self.a - d)
            stop = math.ceil(self.b + d)
        elif len(args) == 1:
            stop = args[0]
        elif len(args) == 2:
            start, stop = args

        x_axis = np.linspace(start, stop, num=(stop - start) * details)
        y_axis = [self.calculate_y(x) for x in x_axis]

        figure, axes = plt.subplots()
        axes.set_ylabel("μ(x)")
        axes.set_xlabel("x")
        axes.plot(x_axis, y_axis, label=f'{type(self).__name__}')
        axes.plot(x_axis, [self.y_max] * len(x_axis), 'g:', label="y_max")
        axes.plot(x_axis, [self.y_min] * len(x_axis), 'r:', label="y_min")
        axes.axvline(x=self.a, ls=':', color='y', label='a')
        axes.axvline(x=self.b, ls=':', color='m', label='m')
        if hasattr(self, 'm'):
            axes.axvline(x=self.m, ls=':', color='c', label='m')
        if hasattr(self, 'm1'):
            axes.axvline(x=self.m1, ls=':', color='b', label='m1')
        if hasattr(self, 'm2'):
            axes.axvline(x=self.m2, ls=':', color='aqua', label='m2')
        axes.legend(loc=0)
        plt.show()


class Linear(MembershipFunction):
    def __init__(self, a, b, **kwargs):
        super().__init__(a, b, **kwargs)

    def calculate_y(self, x: float) -> float:
        if x < self.a:
            return self.y_min
        if x > self.b:
            return self.y_max
        if self.a < x < self.b:
            return self.y_min + ((self.y_max - self.y_min) / (self.b - self.a)) * (x - self.a)


class Triangle(MembershipFunction):
    def __init__(self, a, m: float, b, **kwargs):
        super().__init__(a, b, **kwargs)
        self.m = m

    REQUIRED_INIT_ARGUMENTS = ['a', 'm', 'b']

    def calculate_y(self, x):
        if (x <= self.a) or (x >= self.b):
            return self.y_min
        if self.a < x < self.m:
            return self.y_min + (self.y_max - self.y_min) / (self.m - self.a) * (x - self.a)
        if self.m <= x < self.b:
            return self.y_max - (self.y_max - self.y_min) / (self.b - self.m) * (x - self.m)


class Trapezoidal(MembershipFunction):
    def __init__(self, a, m1: float, m2: float, b, **kwargs):
        super().__init__(a, b, **kwargs)
        self.m1 = m1
        self.m2 = m2

    REQUIRED_INIT_ARGUMENTS = ['a', 'm1', 'm2', 'b']

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
        if self.a < x <= (self.a + self.b) / 2:
            return self.y_min + 2 * math.pow((x - self.a) / (self.b - self.a), 2) * (self.y_max - self.y_min)
        if (self.a + self.b) / 2 < x <= self.b:
            return self.y_min + (1 - 2 * math.pow((self.b - x) / (self.b - self.a), 2)) * (self.y_max - self.y_min)


class Z(MembershipFunction):
    def __init__(self, a, b, **kwargs):
        super().__init__(a, b, **kwargs)

    def calculate_y(self, x: float) -> float:
        return self.y_max + self.y_min - S(a=self.a, b=self.b, y_max=self.y_max, y_min=self.y_min).calculate_y(x)


class Pi(MembershipFunction):
    def __init__(self, a, m: float, b, **kwargs):
        super().__init__(a, b, **kwargs)
        self.m = m

    REQUIRED_INIT_ARGUMENTS = ['a', 'm', 'b']

    def calculate_y(self, x: float) -> float:
        if x <= self.m:
            return S(a=self.a, b=self.m, y_max=self.y_max, y_min=self.y_min).calculate_y(x)
        else:
            return Z(a=self.m, b=self.b, y_max=self.y_max, y_min=self.y_min).calculate_y(x)


def get_init_kwargs_input() -> Dict[str, float]:
    info = 'Additional parameters (default: y_max=1, y_min=0). Write e.g.: y_max: 0.5 \nPress Enter to skip'
    print(info)
    kwargs = {}
    while True:
        _in = str(input('y_max/min:value => '))
        if _in == '':
            break
        arg_colon_val_regex = re.compile('(y_min|y_max)\s*:\s*(\??[0-9]*[.]?[0-9]+)')
        match = arg_colon_val_regex.search(_in)
        if not match:
            print('invalid Input format')
            continue
        groups = arg_colon_val_regex.search(_in).groups()
        kwargs[groups[0]] = float(groups[1])
    return kwargs


def get_init_args_input(f: Type[MembershipFunction]) -> list:
    args = []
    for arg in f.REQUIRED_INIT_ARGUMENTS:
        while True:
            arg_val = get_arg_val_input(arg)
            if arg_val:
                break
        args.append(arg_val)
    return args


def handle_value_error(f, return_on_fail=False, msg: str = "Invalid input"):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            if msg:
                print(msg)
            print(e)
            return return_on_fail

    return wrapper


@handle_value_error
def get_arg_val_input(arg_name: str) -> float:
    return float(input(f'{arg_name}: '))


@handle_value_error
def instantiate_membership_function(F: Type[MembershipFunction], args: list, kwargs: dict) -> MembershipFunction:
    return F(*args, **kwargs)


def main():
    while True:
        functions = {'L': Linear, 'Tri': Triangle, 'Tra': Trapezoidal, 'S': S, 'Z': Z, 'Pi': Pi}
        print(f'Functions: {", ".join(functions)}')
        F = functions.get(input('Choose function:'))
        if not F:
            print("Unrecognized function\nShowing examples...")
            examples()
            continue
        print(f'{F.__name__} - membership function\nEnter parameters:')

        args = get_init_args_input(F)
        kwargs = get_init_kwargs_input()
        f = instantiate_membership_function(F, args, kwargs)
        if not f:
            continue

        f.plot()  # tdo: other stuff to do with f or (gui) [sliders for all parameters,...]


def examples():
    Linear(a=4, b=6, y_max=0.69, y_min=0.2).plot()
    Triangle(a=1, b=5, m=3).plot()
    Trapezoidal(a=1, b=9, m1=4, m2=6).plot()
    S(2, 8, y_min=0.5).plot()
    Z(2, 8).plot()
    Pi(2, 5, 8, y_max=0.69, y_min=0.42).plot()


if __name__ == '__main__':
    main()

