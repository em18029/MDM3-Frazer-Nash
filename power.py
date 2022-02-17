from get_data import main_get_data
from new_model import main_nn
from MCP_script import main_mcp
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

def simple_power(speeds):

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
    max_power = 3000 #Watts
    cut_out_speed = 25 #m/s
    cut_in_speed = 3 #m/s
    availability = 0.97 #%
    betz_limit = 0.5926
    generator_efficiency = 0.7 #Typical
    
    ### Uncertainties
    drop_in_p = 0.03 # +-0.01  # At typical height of turbine
    generator_efficiency_unc = 0.9 # A factor that reduces it further about electrical eff.

    print(df)

    df[df['Wind_Speed'] > cut_out_speed, 'Wind_Speed'] = 0

    print(df)
    
    # For stability the turbine only cuts back in 1 hour after cut out.
    indexes = df[df['Wind_Speed'] > cut_out_speed].index.tolist()
    for idx in indexes:
        df.iloc[idx+1:'Wind_Speed'] = 0

    print(df)

    df[df['Wind_Speed'] < cut_in_speed, 'Wind_Speed'] = 0

    df['Wind_Speed'] = ((df['Wind_Speed']**3)/2)*p*((pi*D**2)/4)*availability*betz_limit*generator_efficiency

    print(df)

    df.rename(columns={'Wind_Speed': 'Power'}, inplace=True)
    df[df['Power'] > max_power, 'Power'] = max_power

    print(df)

    return df

def degradation(yields):

    deg_factor = 0.984 # 1-%

    updated_yields = [P*deg_factor**yields.index(P) for P in yields]

    return updated_yields

def calc_annual_yield(df):
    """
    annual yield = sum of power from wind speed values for a single year
    """
    
    ann_yields = df.groupby(
    [df["datetime"].dt.year])["Power"].mean()
    ann_yield_list = list(ann_yields['Power'])

    ann_yield_list = degradation(ann_yield_list)

    return ann_yield_list


def main_power():

    predictions = main_mcp()
    annual_yield_mcp = calc_annual_yield(predictions)

    return annual_yield_mcp

if __name__ == '__main__':


    trpr, tepr = main_nn()

    print(np.mean(trpr))
    print(len(trpr))

    # Get datasets as dfs.
    #df, df_reanalysis = main_get_data()
    predictions_df = main_mcp()
    print(predictions_df)
    #predictions = list(predictions_df['Wind_Speed'])
    #annual_yield_mcp = calc_annual_yield(predictions)
    power_df = convert_to_power(predictions_df)
    yields = calc_annual_yield(power_df)
    print(np.mean(yields))