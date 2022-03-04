from os import environ

from flmf.membership_functions import *


def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"


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


if __name__ == '__main__':
    suppress_qt_warnings()
    main()
