import numpy as np
import scipy as sc
from scipy import stats

np.random.seed(0)
x = np.random.random(20)
y = np.random.random(20)

# --------------------------------------------------------------------------------------------------------------
# estimate slope and intercept with gradient descent
# --------------------------------------------------------------------------------------------------------------
def residual_sum_of_squares(x, y, slope, intercept):
    sum_of_squared_residuals = 0

    for weight in range(len(x)):
        predicted_height = (slope * x[weight]) + intercept
        observed_height = y[weight]
        sum_of_squared_residuals += (observed_height - predicted_height) ** 2
    return sum_of_squared_residuals

def get_d_intercept(x, y, slope, intercept):
    d_intercept = 0
    for weight, height in zip(x, y):
        d_intercept += -2 * (height - (intercept + (slope * weight)))
    return d_intercept

def get_d_slope(x, y, slope, intercept):
    d_slope = 0
    for weight, height in zip(x, y):
        d_slope += -2 * weight * (height - (intercept + (slope * weight)))
    return d_slope

def gradient_descent(x, y, initial_intercept=0, initial_slope=1, max_steps=10000):
    steps = 0
    intercept = initial_intercept
    slope = initial_slope
    learning_rate = 0.01
    epsilon = 10e-10

    orig_guess = residual_sum_of_squares(x, y, slope, intercept)

    while True:
        if steps > max_steps:
            break

        d_slope = get_d_slope(x, y, slope, intercept)
        d_intercept = get_d_intercept(x, y, slope, intercept)
        new_slope = slope - (d_slope * learning_rate)
        new_intercept = intercept - (d_intercept * learning_rate)
        
        slope = new_slope
        intercept = new_intercept
        new_guess = residual_sum_of_squares(x, y, new_slope, new_intercept)

        if abs(orig_guess - new_guess) <= epsilon:
            break
        else:
            orig_guess = new_guess

        steps += 1

    return slope, intercept

slope, intercept = gradient_descent(x, y)
print('Values from gradient_descent:')
print('\tslope: {}'.format(slope))
print('\tintercept: {}'.format(intercept))

# --------------------------------------------------------------------------------------------------------------
# calculate R^2
# --------------------------------------------------------------------------------------------------------------
ss_mean = residual_sum_of_squares(x, y, slope=0, intercept=y.mean()) # sum of squares around the mean
var_mean = ss_mean / x.size

ss_fit = residual_sum_of_squares(x, y, slope, intercept) # sum of squares around the best fit line
var_fit = ss_fit / x.size

r_squared = (var_mean - var_fit) / var_mean

print('\tr_value: {}'.format(r_squared**0.5))
print('\tr_squared: {}'.format(r_squared))

# --------------------------------------------------------------------------------------------------------------
# calculate F
# --------------------------------------------------------------------------------------------------------------
p_fit = 2 # number of parameters for the equation for the fitted line
p_mean = 1  # number of parameters for the equation for the mean line
F = ((ss_mean - ss_fit) / (p_fit - p_mean)) / ((ss_fit) / (x.size - p_fit))
print('\tF: {}'.format(F))

# --------------------------------------------------------------------------------------------------------------
# results from scipy.stats.linregress
# --------------------------------------------------------------------------------------------------------------
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
print('\nValues from scipy.stats.linregress:')
print('\tslope: {}'.format(slope))
print('\tintercept: {}'.format(intercept))
print('\tr_value: {}'.format(r_value))
print('\tr_squared: {}'.format(r_value**2))
print('\tp_value: {}'.format(p_value))
print('\tstd_err: {}'.format(std_err))
