from backtester import simulate_alternative
from Trader import Trader

from scipy.optimize import minimize
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args

import numpy as np

round = 4
day = 3
time = 999000
trader = Trader()
product = 'BANANAS'

# Define the search space for the hyper-parameters
search_space = [Real(0.1, 3.5, name='imbalance_threshold'),
                Integer(low=500, high=10000, dtype=int, name="amount_of_history"),
                # Categorical(['relu', 'tanh', 'sigmoid'], name='activation')
                ]
# search_space = [Real(0.1, 3.5, name='entry_threshold'),
#                 Real(0.1, 3.5, name='exit_threshold'),
#                 Integer(low=500, high=10000, dtype=int, name="amount_of_history"),
#                 # Categorical(['relu', 'tanh', 'sigmoid'], name='activation')
#                 ]
# search_space = [Integer(low=1, high=5, dtype=int, name="std"),
                # Integer(low=500, high=10000, dtype=int, name="present_hist"),
                # Integer(low=500, high=10000, dtype=int, name="past_hist"),
                # Categorical(['relu', 'tanh', 'sigmoid'], name='activation')
                # ]

# search_space = [Integer(low=1, high=100000, dtype=int, name="dist_from_usual_sum")]

counter = 0

# Define the objective function to be minimized
def objective_function(x):
    # Train a model using the hyperparameters in x and return the performance metric
    profit = simulate_alternative(round,
                                  day,
                                  time,
                                  trader,
                                  time,
                                  True,
                                  False,
                                  False,
                                  [],
                                  x[0],
                                  x[1],
                                  # x[2]
                                  )
    print("Iteration:", counter)

    globals()['counter'] += 1

    if product == 'CPC':
        return profit[time]['COCONUTS'] + profit[time]['PINA_COLADAS']
    elif product == 'BDUP':
        return profit[time]['BAGUETTE'] + profit[time]['DIP'] + profit[time]['UKULELE'] + profit[time]['PICNIC_BASKET']
    else:
        return profit[time][product]

# Define the acquisition function to be used
acquisition_function = 'gp_hedge'

if __name__ == '__main__':
    # Initialize the optimizer
    optimizer = gp_minimize(objective_function, search_space, acq_func=acquisition_function, n_random_starts=10, n_calls=50)

    # Print the best set of hyper-parameters found
    print("Best set of hyper-parameters: ", optimizer.x)

    # Print the best objective function value found
    print("Best objective function value: ", optimizer.fun)