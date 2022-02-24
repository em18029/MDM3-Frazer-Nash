import matplotlib.pyplot as plt
import numpy as np
from cmath import pi
import pandas as pd


def read_clean_csv():

    predictions_df = pd.read_csv('mcp_file.csv',
                       header=0)
    predictions_df = predictions_df.drop(['Unnamed: 0'], axis = 1)
    predictions_df['datetime'] = pd.to_datetime(predictions_df['datetime'], errors='coerce', yearfirst=True)

    return predictions_df


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

def simple_power(df):

    p = 1.225 #air density kg/m^3
    D = 125 #turbine diameter metres
    max_power = 3000 #Watts
    cut_out_speed = 25 #m/s
    cut_in_speed = 3 #m/s
    efficiency = 0.35


    

    power = ((speeds**3)/2)*p*((pi*D**2)/4)*efficiency

    return power

def convert_to_power(df):

    p = 1.225 #air density kg/m^3
    D = 125 #turbine diameter metres
    max_power = 3000000 #Watts
    cut_out_speed = 25 #m/s
    cut_in_speed = 3 #m/s
    availability = 0.97 #%
    betz_limit = 0.5926
    generator_efficiency = 0.7 #Typical
    
    ### Uncertainties
    drop_in_p = 0.03 # +-0.01  # At typical height of turbine
    generator_efficiency_unc = 0.9 # A factor that reduces it further about electrical eff.

    #print(df)

    for values in df['Wind_Speed']:
        if values > cut_out_speed:
            df = df.replace(values, 0)

    
    # For stability the turbine only cuts back in 1 hour after cut out.
    indexes = df[df['Wind_Speed'] > cut_out_speed].index.tolist()
    for idx in indexes:
        df.iloc[idx+1:'Wind_Speed'] = 0


    for values in df['Wind_Speed']:
        if values < cut_in_speed:
            df = df.replace(values, 0)

    df['Wind_Speed'] = (1/2)*df['Wind_Speed']**3*p*pi*D**2*betz_limit*generator_efficiency*availability
    df.rename(columns={'Wind_Speed': 'Power'}, inplace=True)
    

    #print(df)

    for values in df['Power']:
        if values > max_power:
            df = df.replace(values, max_power)
    
    #print(df)

    return df



def simple_power(df):

    p = 1.225 #air density kg/m^3
    D = 125 #turbine diameter metres
    max_power = 3000 #Watts
    cut_out_speed = 25 #m/s
    cut_in_speed = 3 #m/s
    efficiency = 0.35


    power = ((speeds**3)/2)*p*((pi*D**2)/4)*efficiency

    return power

    

def degradation(yields):

    deg_factor = 0.984 # 1-%

    updated_yields = [P*deg_factor**yields.index(P) for P in yields]

    return updated_yields


def convert_to_gwh(df):

    df['Power'] = df['Power']/1000000000
    return df


def calc_annual_yield(df):
    """
    annual yield = sum of power from wind speed values for a single year
    """
    
    ann_yields = df.groupby(
    [df["datetime"].dt.year])["Power"].sum()
    ann_yields_list = ann_yields.tolist()

    #ann_yield_list = degradation(ann_yield_list)

    return ann_yields_list

def calc_annual_stdev(df):

    ann_variance = df.groupby(
    [df["datetime"].dt.year])["Power"].std()
    ann_stdev_list = ann_variance.tolist()
    #power = df['Power'].values
    #stdev = np.std(power, axis=0)
    #print(ann_stdev_list)

    
    return ann_stdev_list


def stdev(yields):
    mean_yield = np.array(yields)
    stdev = np.std(mean_yield)
    mean_yield = np.mean(mean_yield)

    return stdev, mean_yield



def main_power():

    predictions_df = read_clean_csv()
    power_df = convert_to_power(predictions_df)
    power_df = convert_to_gwh(power_df)
    yields = calc_annual_yield(power_df)
    stdeviation, mean_yield = stdev(yields)
    return stdeviation, mean_yield

if __name__ == '__main__':

    predictions_df = read_clean_csv()
    
    #predictions = list(predictions_df['Wind_Speed'])
    #annual_yield_mcp = calc_annual_yield(predictions)
    power_df = convert_to_power(predictions_df)
    power_df = convert_to_gwh(power_df)
    yields = calc_annual_yield(power_df)
    #print(yields)
    #yields = degradation(yields)
    #power_df.to_csv("power_prediction.csv")
    print('Mean yearly power output (GWH)')
    print(yields)
    stdeviation, mean_yield = stdev(yields)
    print('Standard Deviation in power generated ')
    print(stdeviation)
    print('Mean power output (GWH)')
    print(mean_yield)
    #print(np.mean(yields))
    
