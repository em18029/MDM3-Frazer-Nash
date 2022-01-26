""" Program that takes wind speed data in 10 minute intervals from 2008 and combined with reanalysis data from
2000 to 2020 in hourly intervals. The input is an excel worksheet and output is a pandas dataframe """

import pandas as pd

def make_datetime_index(df):
    # Assume we've got the date and the time, and can smoosh them together successfully.
    df['datet'] = df['Date'] + ' ' + df['Time']
    df.index = pd.to_datetime(df['datet'], errors='coerce', yearfirst=True)
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
    #print(df)