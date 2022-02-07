import pandas as pd

def make_datetime_index(df):
    # Assume we've got the date and the time, and can smoosh them together successfully.
    df.index = pd.to_datetime(dict(year=df.Year, month=df.Month, day=df.Day, hour=df.Hour, minute=df.Minute))
    df.drop(columns=['Decimal_Day','Day','Month','Year','Hour','Minute','Decimal_Year','Serial_Time','Epoch_Time'])
    return df

def combine_the_short_term_and_long_term_speed_measurements(long_term_df, short_term_df):
    # Match the short and long periods, then do the correlation. Uses the direction from the short-term
    # as the true direction.
    ltdf = long_term_df
    stdf = short_term_df
    # Add the direction from the short-term measurements
    direction = stdf['Wind_Direction']
    stdf = stdf['Wind_Speed']
    ltdf = ltdf['Wind_Speed']
    # Create a combined dataframe
    comb_df = pd.DataFrame()
    comb_df['lt_speed']=ltdf.loc[ltdf.index.intersection(stdf.index)]
    # Get the short-term speed
    st_speed = stdf.groupby(pd.Grouper(freq='H')).mean()
    comb_df['st_speed'] = st_speed
    comb_df['direction']=direction.loc[direction.index.intersection(stdf.index)]
    comb_df = comb_df.dropna()
    return comb_df

def missing_data(df):

    ## Will find a better for missing data. Imputation.

    df.drop(df[df['Temperature'] > 1000].index, inplace=True)

    return df

def main_get_data():
    
    # Load in measured data.
    measured_input_data = pd.read_excel("measurement_data_2015.xlsx")
    measurement = make_datetime_index(measured_input_data)
    measurement = missing_data(measurement)

    # Load in reanalysis data, and give it datetime columns.
    reanalysis_input_data = pd.read_excel('reanalysis_data_2002-2016.xlsx')
    reanalysis = make_datetime_index(reanalysis_input_data)
    reanalysis = missing_data(reanalysis)

    df = combine_the_short_term_and_long_term_speed_measurements(reanalysis, measurement)

    return df, measurement, reanalysis


if __name__ == '__main__':

    # Load in measured data.
    measured_input_data = pd.read_excel("measurement_data_2015.xlsx")
    measurement = make_datetime_index(measured_input_data)
    measurement = missing_data(measurement)

    # Load in reanalysis data, and give it datetime columns.
    reanalysis_input_data = pd.read_excel('reanalysis_data_2002-2016.xlsx')
    reanalysis = make_datetime_index(reanalysis_input_data)
    reanalysis = missing_data(reanalysis)

    df = combine_the_short_term_and_long_term_speed_measurements(reanalysis, measurement)