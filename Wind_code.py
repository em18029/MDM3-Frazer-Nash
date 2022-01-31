"""
This script is an example of simple Model-Correlate-Predict for long-term prediction of wind resource.
It correlates a short term with a long term data source, then used the derived correlation model to predict what the
short-term (measured) data might have looked like for the full 20-year period.
"""

import copy
import pandas as pd
import matplotlib.pyplot as plt
import sys
import csv
import json
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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
    return comb_df


def get_sector_dict(sector_size=30, debug=False, dirstring='m_dir'):
    """Returns dict with queries defining sectors based on compass point wind directions"""
    sector_dict = {}
    sectors_start = 0
    num_sectors = 360 / sector_size
    if 360 % num_sectors != 0:
        raise ValueError('360 should divide exactly into sector size')
    sector_centres = [x*sector_size for x in range(int(num_sectors))]
    for i_centre, centre in enumerate(sector_centres):
        #sector_name = wa.WIND_DIRECTIONS[i_centre]
        sector_right = centre + sector_size / 2
        sector_left = centre - sector_size/2
        if sector_left < 0:
            sector_left = 360 + sector_left
        if sector_right > 360:
            sector_right = 360 - sector_right
        if sector_right > sector_left:
            sector_string = '{dirstring} > {} and {dirstring} <= {}'.format(sector_left, sector_right, dirstring=dirstring)
        elif sector_left > sector_right:
            sector_string = '{dirstring} > {} or {dirstring} <= {}'.format(sector_left, sector_right, dirstring=dirstring)
        else:
            sector_string = 'Invalid sector'
        if debug:
            print(sector_string)
        sector_dict[str(sector_centres[i_centre])] = sector_string
    return(sector_dict)


def get_dict_with_dfs_for_each_sector(df, **kwargs):
    # Split the data into sectors, for a selected vane
    sector_dict = get_sector_dict(**kwargs)
    sectored_dfs = dict()
    for sector, query in sector_dict.items():
        sectored_dfs[sector] = copy.deepcopy(df.query(query))
    # from pprint import pprint
    # pprint(sector_dict)
    return sectored_dfs


def train_a_linear_model(X, y, plot=False):
    # Todo: make the plotting optional (it crashes if plot is false right now)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

    if plot:
        plt.subplots()
        plt.scatter(X_test, y_test, color='black', alpha=0.3, marker='+')
        plt.plot(X_test, y_pred, color='red', linewidth=0.5)

    return regressor, df


def find_linear_model_between_two_speeds(series1, series2, plot=False):
    # Plot the correlation between the shared heights (use the
    # linear model from LNG terminal lifing work)
    X = series1.values.reshape(-1,1)
    y = series2.values.reshape(-1,1)
    regressor1, df = train_a_linear_model(X, y, plot=plot)
    #print(f' The correlation coefficient between s1 and s2 is {s1.corr(s2)}')
    return regressor1, df


def find_sectorial_correlations(comb_df, name=''):
    regressor_dict = dict()
    sect_dict = get_dict_with_dfs_for_each_sector(comb_df, sector_size=30, dirstring='direction')
    corr_coef_df = pd.DataFrame(columns=sect_dict.keys(), index=[0])
    for sector, df in sect_dict.items():
        u1 = df['lt_speed'] # The long-term is x; we know it
        u2 = df['st_speed'] # the 'short term'; e.g. speed at the mast is what we want, so it's Y.
        corr_coeff = u1.corr(u2)
        print(f'Correlation in sector {sector} is {corr_coeff}')
        regressor, obs_v_predicted = find_linear_model_between_two_speeds(u1, u2, plot=True)
        regressor_dict[sector] = regressor
        corr_coef_df[sector] = corr_coeff
        plt.grid()
        plt.title(f'{name} Sector: {sector}')
        plt.xlabel('Long-term wind speed (m/s)')
        plt.ylabel('Short-term wind speed (m/s)')
    return regressor_dict, sect_dict, corr_coef_df


def make_a_long_term_prediction(long_term_df, regressor_dict):
    """The P part of MCP. Long-term df is typically a 20-year set of reanalysis data.
    regressor dict is a dict produced by the function `find_sectorial_correlations`.
    The function returns a timeseries with the predicted speed for all sectors combined."""
    lt_dict = get_dict_with_dfs_for_each_sector(long_term_df, dirstring='Measured_Direction')
    predicted_sect_speed = dict()
    for sector, df in lt_dict.items():
        print(sector)
        df = df.dropna()
        regressor = regressor_dict[sector]
        prediction = regressor.predict(df['Measured_Windspeed'].values.reshape(-1, 1)).flatten()
        predicted_sect_speed[sector] = pd.Series(index=df.index,
                                                 data=prediction)
    lt_df_prediction = pd.concat(predicted_sect_speed.values()).sort_index()
    return lt_df_prediction, predicted_sect_speed


if __name__ == '__main__':

    # Load in measured data from 2008.
    measured_input_data = pd.read_excel("measured_wind_speed_and_direction.xlsx", dtype={"Time":str})
    df_meas = make_datetime_index(measured_input_data)['2008']

    # Load in reanalysis data from 2000 to 2020, and give it datetime columns.
    reanalysis = pd.read_excel('reanalysis.xlsx')
    reanalysis['Time'] = reanalysis['Time'].apply(num_to_time)
    reanalysis['Date'] = reanalysis['Date'].apply(num_to_date)
    df_ra1 = make_datetime_index(reanalysis)

    # Put them in the same dataframe, perform linear regression on each sector, then use the derived model to
    # predict what the measured data might have looked like for 20 years.
    df = combine_the_short_term_and_long_term_speed_measurements(reanalysis, df_meas)
    regressor_dict, sd, ccd = find_sectorial_correlations(df, 'Measured_Direction')
    lt_df_prediction, predicted_sect_speed = make_a_long_term_prediction(reanalysis, regressor_dict)
    #plt.subplots()
    #lt_df_prediction.plot()
    #plt.title('Predicted windspeed for 20 years')
    #plt.show()
    pd.set_option('display.max_rows', None),
    lt_df_prediction.to_csv('prediction_time_series3.csv'),
                            
    print(lt_df_prediction)
    
