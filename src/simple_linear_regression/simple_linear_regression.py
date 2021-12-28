# predicted response vector
# y_pred = b[0] + b[1]*x
import numpy as np
import matplotlib.pyplot as plt


def estimate_coefficient(x, y):
    """Determine the coefficient of estimation
    """
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)

    # Calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(y * x) - n * m_x * m_x

    # Calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    return b_0, b_1


def plot_regression(x, y, b):
    """Plot the Simple Linear Regression"""
    plt.scatter(x, y,
                color="m", marker="o", s=30)
    # predicted response vector
    y_pred = b[0] + b[1] * x

    # plotting the regression line
    plt.plot(x, y_pred, color="g")

    # putting labels
    plt.xlabel("x")
    plt.ylabel("y")

    # function to show the plot
    plt.show()


def main():
    """main function"""
    # observations / data
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 3, 2, 5, 7, 8, 9, 10, 11, 12])

    # estimating coefficients
    b = estimate_coefficient(x, y)
    print(f'Co-efficients are: {b[0]} and {b[1]}')

    # plotting the regression line
    plot_regression(x, y, b)


if __name__ == "__main__":
    main()
