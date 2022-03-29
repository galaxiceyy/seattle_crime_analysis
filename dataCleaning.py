"""

CSE 163
A file that contains some functions that
clean, modify, process and export the
Seattle crime data for analysis

"""
# pip install geopandas
import pandas as pd
import geopandas as gpd


def preprocess_crime_data(SPD_file_name):
    """
    The function loads in SPD csv file and coverts
    the offense start datatime into column to
    filter the crime incidents occurred between
    2016-2020.
    """
    chunk = pd.read_csv(SPD_file_name, chunksize=1000,
                        usecols=["Report Number",
                                 "Offense Start DateTime",
                                 "Report DateTime",
                                 "Offense Parent Group",
                                 "Offense Code", "Offense",
                                 "MCPP", "Longitude", "Latitude"])
    df = pd.concat(chunk)
    df['date2'] = pd.to_datetime(df['Offense Start DateTime'])
    df['Year'] = df['date2'].dt.year
    df['Month'] = df['date2'].dt.month
    df['Day'] = df['date2'].dt.day
    df['Hour'] = df['date2'].dt.hour
    df['Minute'] = df['date2'].dt.minute
    df = df.drop(['Offense Start DateTime'], axis=1)
    df = df.drop(['date2'], axis=1)
    df = df[df['Year'] >= 2016]
    df.to_csv('data/cleaned_crime_data.csv', index=False)
    return df


def clean_map_data(geoData):
    """
    This function takes in shp geolocation filr for
    seattle areas.
    Clean and correct some mistakes.
    Export the data into a shp file.
    """
    mcpp = gpd.read_file(geoData)
    # Load Data
    mcpp.loc[mcpp.NAME == "DOWNTOWN COMMERICAL",
             "NAME"] = "DOWNTOWN COMMERCIAL"
    mcpp.loc[mcpp.NAME == "INTERNATIONAL DISTRICT - EAST",
             "NAME"] = "CHINATOWN/INTERNATIONAL DISTRICT"
    mcpp.loc[mcpp.NAME == "INTERNATIONAL DISTRICT - WEST",
             "NAME"] = "CHINATOWN/INTERNATIONAL DISTRICT"
    mcpp.loc[mcpp.NAME == "NORTH CAPITOL HILL", "NAME"] = "CAPITOL HILL"
    mcpp.loc[mcpp.NAME == "JUDKINS PARK",
             "NAME"] = "JUDKINS PARK/NORTH BEACON HILL"
    mcpp.loc[mcpp.NAME == "MT BAKER/NORTH RAINIER", "NAME"] = "MOUNT BAKER"
    mcpp.loc[mcpp.NAME == "NORTH BEACON/JEFFERSON PARK",
             "NAME"] = "NORTH BEACON HILL"
    mcpp.loc[mcpp.NAME == "COMMERCIAL DUWAMISH", "NAME"] = "NORTH DELRIDGE"
    # Clean Data
    mcpp.to_file("data/mapping/cleaned_dis.shp")


def add_income(income_file_name, crime_file_name):
    """
    This function combines the crime dataset
    with the income and population dataset.
    Returns a merged dataset.
    """
    income_data = pd.read_csv(income_file_name,
                              usecols=["Zip Code",
                                       "Avg. Income/H/hold",
                                       "Population"]).dropna()
    crime_data = pd.read_csv(crime_file_name,
                             usecols=["Zipcode",
                                      "MCPP",
                                      "Offense Parent Group"]).dropna()
    income_data['Zip Code'] = income_data['Zip Code'].astype("Int64")
    crime_data['Zipcode'] = pd.to_numeric(crime_data['Zipcode'],
                                          errors='coerce')
    crime_data['Zipcode'] = crime_data['Zipcode'].astype("Int64")
    merged_data = crime_data.merge(income_data, left_on='Zipcode',
                                   right_on='Zip Code')
    merged_data.to_csv('data/Income_Crime_Data.csv', index=False)


def group_incident(df):
    """
    This function uses the formula: incident#
    /population to calculate crime rate for
    each area in Seattle.
    Create a new dataset with crime rate
    """
    df = pd.read_csv('data/Income_Crime_Data.csv')
    df = pd.read_csv('data/Income_Crime_Data.csv')
    df = df.groupby(['Zipcode', 'Population',
                    'Avg. Income/H/hold',
                     'MCPP'])['Offense Parent Group'].count()
    df = df.reset_index(name='Incident count')
    string = 'Avg. Income/H/hold'
    df[string] = df[string].str.replace(',', '')
    df[string] = df[string].str.replace('$', '').astype(float)
    df['Population'] = df['Population'].str.replace(',', '').astype(float)
    df['Crime Rate'] = df['Incident count'] / df['Population'] * 100000
    df = df.loc[(df["Crime Rate"] < 100000) & (df["Crime Rate"] > 0)
                                            & df["Avg. Income/H/hold"] > 0]
    df.to_csv('data/income_crime_rate.csv', index=False)


def add_column(File_name):
    """
    This function adds a new column to each case and labels it with
    its time zone
    """
    df = pd.read_csv(File_name)
    df.loc[df['Hour'] <= 6, 'Timezone'] = '1:Earlymonring'
    df.loc[((df['Hour'] > 6) & (df['Hour'] <= 12)), 'Timezone'] = '2:Morning'
    df.loc[((df['Hour'] > 12) & (df['Hour'] <= 18)),
           'Timezone'] = '3:Afternoon'
    df = df.fillna('4:Night')
    df.to_csv('data/plot_data.csv', index=False)


def filter_university(filename):
    """
    This function takes in data for plotting
    Filters out the data for University area
    Export the data into a CSV file.
    """
    df = pd.read_csv(filename)
    df = df[(df['MCPP'] == 'UNIVERSITY')]
    df.to_csv('data/university.csv', index=False)


def main():
    preprocess_crime_data('data/SPD_Crime_Data__2008-Present.csv')
    add_income('data/income/median_income.csv',
               'data/Crime_2020_With_Zip.csv')
    group_incident('data/Income_Crime_Data.csv')
    clean_map_data('data/mapping/MCPPAreas.shp')
    add_column('data/cleaned_crime_data.csv')
    filter_university('data/plot_data.csv')


if __name__ == "__main__":
    main()
