import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
from power_code import main_power





def distribution(main_power):

    sigma, mu =  main_power()

    return sigma, mu


def plot_p90(mu, sigma, p90, p50):

    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    #plt.vlines(x=p90, ymin=0, ymax=stats.norm.pdf(x, mu, sigma), colors='green', ls=':', lw=2, label='p90')
    #plt.vlines(x=p75, ymin=0, ymax=stats.norm.pdf(x, mu, sigma), colors='red', ls=':', lw=2, label='p75')
    #plt.vlines(x=mu, ymin=0, ymax=stats.norm.pdf(x, mu, sigma), colors='black', ls=':', lw=2, label='p50')
    
    plt.xlabel("Power generated (GWH)")
    plt.ylabel("Probability density")
    
    plt.grid()
    plt.show()


def calc_p90(mu, sigma):

    sigma = sigma * 1.282
    p90 = mu - sigma
    
    return p90



def calc_p75(mu, sigma):

    sigma = sigma * 0.674
    p75 = mu - sigma
    
    return p75



if __name__ == '__main__':

    sigma, mu = distribution(main_power)

    p90 = calc_p90(mu, sigma)
    print('p90 values are:')
    print(p90)

    p75 = calc_p75(mu, sigma)
    print('p75 values are:')
    print(p75)
          
    plot_p90(mu, sigma, p90, p75)

