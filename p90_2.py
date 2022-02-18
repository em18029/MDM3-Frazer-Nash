import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
from power_code import main_power


def distribution(main_power):
    

    mu, var =  main_power()
    sigma = np.array(var)
    mu = np.array(mu)

    return mu, sigma

    


def plot_p90(mu, sigma):

    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    plt.xlabel("Power generated")
    plt.ylabel("Probability density")
    plt.legend = (["2002", "2003", "2004", "2005", "2006", "2007",
                "2008", "2009", "2010", "2011", "2012", "2013",
                "2014", "2015", "2016"])
    

#    for a, b in zip(mu, sigma):
#        year = 2002
#        plt.plot(x, stats.norm.pdf(x, a, b), label = year)
#        year += 1
#
#    plt.legend() 
#    plt.xlabel("Power generated")
#   plt.ylabel("Probability density")
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

    mu, sigma = distribution(main_power)

    p90 = calc_p90(mu, sigma)
    print('p90 values are:')
    print(p90)

    p75 = calc_p75(mu, sigma)
    print('p75 values are:')
    print(p75)
          
    plot_p90(mu, sigma)

