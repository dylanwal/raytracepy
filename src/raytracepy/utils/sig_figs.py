
def sig_figs(number: float, significant_digit: int = 3) -> float:
    """
    Given a number return a string rounded to the desired significant digits.
    :param number:
    :param significant_digit:
    :return:
    """
    return float('{:.{p}g}'.format(number, p=significant_digit))
