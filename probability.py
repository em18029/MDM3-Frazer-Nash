from power import main_power
import numpy as np

def distribution(powers):

    #### Make a max uncertainty and min uncertainty and multiply/add/subtract to make two new lists. Combine for new std & mean.

    mu = np.mean(powers)
    sigma = np.std(powers)
  
    return mu, sigma

def calc_p90(powers):

    mu, sigma = distribution(powers)
    print(mu)
    print(sigma)
    p90 = mu - 1.282*sigma

    return p90

def calc_p50():

    return

if __name__ == '__main__':

    powers = main_power()
    p90 = calc_p90(powers)
    print(p90)