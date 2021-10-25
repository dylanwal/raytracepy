
import warnings


def sig_figs(number, significant_figures: int = 3):
    try:
        return '{:g}'.format(float('{:.{p}g}'.format(number, p=significant_figures)))
    except ValueError:
        warnings.warn("number of significant_figures can't be negative! Using whole number")
        return number


if __name__ == "__main__":
    test_number = 123.52423423
    print(test_number)
    print(sig_figs(test_number, -1))
    print(sig_figs(test_number, 2))
    print(sig_figs(test_number, 3))
    print(sig_figs(test_number, 4))
    print(sig_figs(test_number, 7))
