from get_data import main_get_data
import numpy as np

def distribution(powers):

    mu = np.mean(powers)
    sigma = np.std(powers)
    
    return mu, sigma

def uncertainties():

    return

def calc_p90(powers):

    mu, sigma = distribution(powers)
    p90 = mu - 1.282*sigma

    return p90

def calc_p50():

    return

if __name__ == '__main__':

    p90 = calc_p90