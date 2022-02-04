from get_data import main_get_data
from example_script import main_industry_standard
import matplotlib.pyplot as plt
import numpy as np
from cmath import pi

def generate_power_curve(df):

    p = 1.225 #kg/m^3
    D = 125 #metres
    min_speed = 3
    max_speed = 25
    #speed = np.linspace(0,20,num=81)
    speed = 56166.245
    Power = ((speed**3)/2)*p*((pi*D**2)/4)
    print(Power)
    #Power = np.where(speed >= min_speed and speed <= max_speed,((speed**3)/2)*p*((pi*D**2)/4),0)
    #plt.plot(speed,Power)
    #plt.show()

def calc_annual_yield(df):
    """
    annual yield = sum of power from wind speed values for a single year
    """
    p = 1.225 #kg/m^3
    D = 125 #metres
    power_df = ((df**3)/2)*p*((pi*D**2)/4)
    year_list = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19']
    ann_yield_list = [np.sum(power_df.loc['2000-01-01 02:00:00':'2000-12-31 23:00:00'])/10**9]
    for year in year_list:
        yearly_power_sum = np.sum(power_df.loc[f'20{year}-01-01 00:00:00':f'20{year}-12-31 23:00:00'])/10**9
        ann_yield_list.append(yearly_power_sum)
    #print(ann_yield_list)

    return ann_yield_list


if __name__ == '__main__':

    # Get datasets as dfs.
    df, df_reanalysis = main_get_data()
    predictions_standard, sect_predictions_standard = main_industry_standard(df, df_reanalysis)
    annual_yield_standard = calc_annual_yield(predictions_standard)
    generate_power_curve(df)
