import inspect
import math
import re
from abc import ABC, abstractmethod
from functools import wraps
from typing import Dict, Type, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sympy import simplify


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
        if val < 0 or val >= 1:
            raise ValueError("Min for membership can't be < 0 or >= 1")
        self._y_min = val

    @property
    def y_max(self) -> float:
        return self._y_max

    @y_max.setter
    def y_max(self, val):
        if val > 1 or val <= 0:
            raise ValueError("Max for membership can't be > 1 or <= 0")
        self._y_max = val

    @abstractmethod
    def calculate_y(self, x: float) -> float:
        pass

    def get_function_def(self, print_: bool = True) -> str:
        """
        :returns: the function definition with values inserted into formula and in simplified version
        :param print_ prints the function definition if True
        """
        ifs = self.get_if_statements()
        returns = self.get_return_statements()

        ifs = self.insert_vars(ifs)
        returns = self.insert_vars(returns)

        f_definition = f'{type(self).__name__} := {{\n'
        for if_, return_ in zip(ifs, returns):
            f_definition += f'{if_}: {return_} ==> {simplify(return_).evalf(3)}\n'

        if print_:
            print(f_definition)
        return f_definition

    def get_if_statements(self) -> List[str]:
        """:returns: expression in if statements of self.calculate_y"""
        if_statement_regex = re.compile('if (.+):')
        method_body = inspect.getsource(self.calculate_y)
        return if_statement_regex.findall(method_body)

    def get_return_statements(self) -> List[str]:
        """:returns: expression in return statements of self.calculate_y"""
        return_statement_regex = re.compile('return (.+)')
        method_body = inspect.getsource(self.calculate_y)
        return return_statement_regex.findall(method_body)

    def insert_vars(self, expressions: List[str]) -> list:
        """:returns: List of expressions with all text equal to any instance variable replaced with the corresponding
        \value """
        vars_ = vars(self)
        for var in vars_:
            for i, expr in enumerate(expressions):
                expressions[i] = expr.replace(f"self.{var.lstrip('_')}", str(vars_[var]))
        return expressions

    def plot(self, detail: int = 15 ** 4) -> Figure:
        """
        Plots function and its parameters in apropriate range.
        :param detail: the amount of evenly spaced x-values computed
        """
        difference_a_b = abs(self.b - self.a) / 8
        start = self.a - difference_a_b
        stop = self.b + difference_a_b

        x_axis = np.linspace(math.floor(start), math.ceil(stop), num=detail)
        #           whyy does it only take integers here?
        # computing more numbers than necessary (for small differences between a and b; loosing lots of precision)
        mask = ((x_axis >= start) & (x_axis <= stop))

        x_axis = x_axis[mask]
        y_axis = [self.calculate_y(x) for x in x_axis]

        figure, axes = plt.subplots()
        axes.set_ylabel("μ(x)")
        axes.set_xlabel("x")
        axes.plot(x_axis, y_axis, label=f'{type(self).__name__}')
        axes.plot(x_axis, [self.y_max] * len(x_axis), 'g:', label="y_max")
        axes.plot(x_axis, [self.y_min] * len(x_axis), 'r:', label="y_min")
        axes.axvline(x=self.a, ls=':', color='y', label='a')
        axes.axvline(x=self.b, ls=':', color='m', label='b')
        if hasattr(self, 'm'):
            axes.axvline(x=self.m, ls=':', color='c', label='m')
        if hasattr(self, 'm1'):
            axes.axvline(x=self.m1, ls=':', color='b', label='m1')
        if hasattr(self, 'm2'):
            axes.axvline(x=self.m2, ls=':', color='aqua', label='m2')
        axes.legend(loc=0)
        plt.show(block=False)

        return figure


class Linear(MembershipFunction):
    def __init__(self, a, b, **kwargs):
        super().__init__(a, b, **kwargs)

    def calculate_y(self, x: float) -> float:
        if x <= self.a:
            return self.y_min
        if x >= self.b:
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
        if self.a < x <= self.m1:
            return self.y_min + ((self.y_max - self.y_min) / (self.m1 - self.a)) * (x - self.a)
        if self.m2 <= x < self.b:
            return self.y_max - ((self.y_max - self.y_min) / (self.b - self.m2)) * (x - self.m2)


class S(MembershipFunction):

    def calculate_y(self, x) -> float:
        if x <= self.a:
            return self.y_min
        if x > self.b:
            return self.y_max
        if self.a < x <= (self.a + self.b) / 2:
            return self.y_min + 2 * pow((x - self.a) / (self.b - self.a), 2) * (self.y_max - self.y_min)
        if (self.a + self.b) / 2 < x <= self.b:
            return self.y_min + (1 - 2 * pow((self.b - x) / (self.b - self.a), 2)) * (self.y_max - self.y_min)


class Z(MembershipFunction):
    def __init__(self, a, b, **kwargs):
        super().__init__(a, b, **kwargs)

    def calculate_y(self, x: float) -> float:
        return self.y_max + self.y_min - S(a=self.a, b=self.b, y_max=self.y_max, y_min=self.y_min).calculate_y(x)

    def get_function_def(self, print_: bool = True) -> str:
        s = S(a=self.a, b=self.b, y_max=self.y_max, y_min=self.y_min)
        ifs = self.insert_vars(s.get_if_statements())
        returns = self.insert_vars(s.get_return_statements())
        f_definition = f'{type(self).__name__} := {{\n'
        for if_, return_ in zip(ifs, returns):
            return_ = f'{self.y_max} + {self.y_min} - {return_}'
            f_definition += f'{if_}: {return_} ==> {simplify(return_)}\n'
        if print_:
            print(f_definition)
        return f_definition


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

    def get_function_def(self, print_: bool = True) -> str:
        f_def = f'Pi := {{\n' \
                f'x <= {self.m}:\n' \
                f'{S(a=self.a, b=self.m, y_max=self.y_max, y_min=self.y_min).get_function_def(print_=False)}' \
                f'x > {self.m}:\n' \
                f'{Z(a=self.m, b=self.b, y_max=self.y_max, y_min=self.y_min).get_function_def(print_=False)}'
        if print_:
            print(f_def)
        return f_def


def get_init_kwargs_input() -> Dict[str, float]:
    info = 'Additional parameters (default: y_max=1, y_min=0). Write e.g.: y_max: 0.5 \nPress Enter to skip'
    print(info)
    kwargs = {}
    while True:
        _in = str(input('y_max/min: '))
        print(_in)
        if _in == '':
            break
        arg_colon_val_regex = re.compile('(y_min|y_max)\s*:\s*(\??[0-9]*[.]?[0-9]+)')
        match = arg_colon_val_regex.search(_in)
        if not match:
            print('invalid input format')
            continue
        groups = arg_colon_val_regex.search(_in).groups()
        kwargs[groups[0]] = float(groups[1])
    return kwargs


def get_init_args_input(f: Type[MembershipFunction]) -> list:
    args = []
    for arg in f.REQUIRED_INIT_ARGUMENTS:
        while True:
            arg_val = get_float_input(arg)
            if arg_val:
                break
        args.append(arg_val)
    return args


def handle_value_error(f):
    @wraps(f)
    def wrapper(*args, return_on_fail=False, msg: str = "Invalid input", print_err: bool = True, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            if msg:
                print(msg)
            if print_err:
                print(e)
            return return_on_fail

    return wrapper


@handle_value_error
def get_float_input(input_txt: str) -> float:
    return float(input(f'{input_txt}: '))


@handle_value_error
def instantiate_membership_function(F: Type[MembershipFunction], args: list, kwargs: dict) -> MembershipFunction:
    return F(*args, **kwargs)


def ask_to_calculate_y(f: MembershipFunction) -> bool:
    _in = get_float_input('x', msg='', print_err=False)
    if not _in and isinstance(_in, bool):
        return False
    print(f.calculate_y(_in))
    return True


def main():
    # cli with rich https://youtu.be/4zbehnz-8QU ? and simple gui for sliding params, etc...?
    while True:
        FUNCTIONS = {'L': Linear, 'Tri': Triangle, 'Tra': Trapezoidal, 'S': S, 'Z': Z, 'Pi': Pi}
        print(f'Functions: {", ".join(FUNCTIONS)}')
        F = FUNCTIONS.get(input('Choose function:'))
        if not F:
            _in = input("Unrecognized function\n'e' for examples\nAny key to continue... : ")
            if _in in 'eE' and _in != '':
                examples()
            continue
        print(f'{F.__name__} - membership function\nEnter parameters:')

        args = get_init_args_input(F)
        kwargs = get_init_kwargs_input()
        f: MembershipFunction = instantiate_membership_function(F, args, kwargs)
        if not f:
            continue

        f.get_function_def()
        f.plot()

        print('Calculate specific points:\n(Any key to exit...)')
        while True:
            if not ask_to_calculate_y(f):
                break


def examples():
    Linear(a=4, b=6, y_max=0.69, y_min=0.2).plot()
    Triangle(a=1, b=5, m=3).plot()
    Trapezoidal(a=1, b=9, m1=4, m2=6).plot()
    S(2, 8, y_min=0.5).plot()
    Z(2, 8).plot()
    Pi(2, 5, 8, y_max=0.69, y_min=0.42).plot()


def test():
    # testing if edge cases are accounted for in implementations of calculate_y
    FUNCTIONS = [Linear(a=10, b=40), Triangle(a=10, b=40, m=15), Trapezoidal(a=10, b=40, m1=15, m2=30),
                 S(a=10, b=40), Z(a=10, b=40), Pi(a=10, b=40, m=15)]
    x_test_values = {
        'a': 10,
        'm': 15,
        'm1': 15,
        'm2': 30,
        'b': 40
    }
    for f in FUNCTIONS:
        for x_val in x_test_values:
            if hasattr(f, x_val):
                assert f.calculate_y(x_test_values[x_val]) is not None
    print("Passed!")


if __name__ == '__main__':
    main()
