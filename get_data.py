from cmath import pi
import pandas as pd
import numpy as np

def make_datetime_index(df):
    # Assume we've got the date and the time, and can smoosh them together successfully.
    df['DateTime'] = df['Date'] + ' ' + df['Time']
    df.index = pd.to_datetime(df['DateTime'], errors='coerce', yearfirst=True)
    return df[['Measured_Windspeed', 'Measured_Direction']]

def num_to_time(num):
    x = f"{num:04}"
    return f"{x[:2]}:{x[2:4]}:00"

def num_to_date(num):
    x = str(num)
    return f"{x[:4]}-{x[4:6]}-{x[6:]}"


def combine_the_short_term_and_long_term_speed_measurements(long_term_df, short_term_df):
    # Match the short and long periods, then do the correlation. Uses the direction from the short-term
    # as the true direction.
    # Abbreviate
    ltdf = long_term_df
    stdf = short_term_df
    # Get inputs to combined dataframe
    ltdf = ltdf['Measured_Windspeed'].dropna()
    # Create a combined dataframe
    comb_df = pd.DataFrame()
    comb_df['lt_speed']=ltdf.loc[ltdf.index.intersection(stdf.index)]
    # Get the short-term speed
    st_speed = stdf['Measured_Windspeed'].groupby(pd.Grouper(freq='H')).mean()
    comb_df['st_speed'] = st_speed
    # Add the direction from the short-term measurements
    direction = stdf['Measured_Direction']
    comb_df['direction']=direction.loc[direction.index.intersection(stdf.index)]
    comb_df = comb_df.dropna()
    #print(comb_df)
    # Combined df has 1 hourly intervals for 1 year 2008. Gives speed from each dataset for the long term and short term. Also direction
    # Is this the uncertainty/error? Used to make that correlation?
    return comb_df

def get_power_from_speed(df):
    k = 0.00013
    Cp = 0.59
    p = 1.225 #kg/m^3
    D = 125 #metres
    min_speed = 3
    max_speed = 15
    df['lt_Power'] = np.where(df['lt_speed'] >= min_speed and df['lt_speed'] <= max_speed,((df['lt_speed']**3)/2)*k*Cp*p*((pi*D**2)/4),0)
    df['st_Power'] = np.where(df['st_speed'] >= min_speed and df['st_speed'] <= min_speed,((df['st_speed']**3)/2)*k*Cp*p*((pi*D**2)/4),0)
    return df

def main_get_data():
    # Load in measured data from 2008.
    measured_input_data = pd.read_excel("measured_wind_speed_and_direction.xlsx", dtype={"Time":str})
    df_meas = make_datetime_index(measured_input_data)['2008']

    # Load in reanalysis data from 2000 to 2020, and give it datetime columns.
    reanalysis = pd.read_excel('reanalysis.xlsx')
    reanalysis['Time'] = reanalysis['Time'].apply(num_to_time)
    reanalysis['Date'] = reanalysis['Date'].apply(num_to_date)
    reanalysis.drop('Unnamed: 0', inplace=True, axis=1)
    df_reanalysis = make_datetime_index(reanalysis)
    #print(df_reanalysis)

    #df = combine_the_short_term_and_long_term_speed_measurements(reanalysis, df_meas)
    df = combine_the_short_term_and_long_term_speed_measurements(df_reanalysis, df_meas)
    #print(df)
    return df, df_reanalysis


if __name__ == '__main__':

    # Load in measured data from 2008.
    measured_input_data = pd.read_excel("measured_wind_speed_and_direction.xlsx", dtype={"Time":str})
    df_meas = make_datetime_index(measured_input_data)['2008']

    # Load in reanalysis data from 2000 to 2020, and give it datetime columns.
    reanalysis = pd.read_excel('reanalysis.xlsx')
    reanalysis['Time'] = reanalysis['Time'].apply(num_to_time)
    reanalysis['Date'] = reanalysis['Date'].apply(num_to_date)
    reanalysis.drop('Unnamed: 0', inplace=True, axis=1)
    df_reanalysis = make_datetime_index(reanalysis)
    #print(df_reanalysis)

    #df = combine_the_short_term_and_long_term_speed_measurements(reanalysis, df_meas)
    df = combine_the_short_term_and_long_term_speed_measurements(df_reanalysis, df_meas)
    #print(df['st_speed'])

    #dfpower = get_power_from_speed(df)
    #print(dfpower)