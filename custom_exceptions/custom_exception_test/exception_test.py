# Testing the custom exception
from custom_exceptions.custom_exception import ValueTooSmallError
from custom_exceptions.custom_exception import ValueTooLargeError


class ValueCheck:
    """
    class to check the valid value
    """
    number = 10

    def __init__(self, value: int):
        """

        :param value:
        """
        self.value = value

    def check_value(self):
        """

        :return:
        """
        try:
            if self.value < self.number:
                raise ValueTooSmallError
            elif self.value > self.number:
                raise ValueTooLargeError
            else:
                print("Congratulations! You guessed it correctly.")
        except ValueTooSmallError:
            print("The Value is too small, try again")
            print()
        except ValueTooLargeError:
            print("The Value is too large, try again")
            print()


def main():
    """
    Main Function
    :return:
    """
    i = int(input("Enter a number:"))
    value_check = ValueCheck(i)
    value_check.check_value()


if __name__ == "__main__":
    main()
