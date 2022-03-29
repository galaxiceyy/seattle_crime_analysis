"""
This is the file that focuses on converting
the coordinate in the crime data into
zipcode.The dataset is from Seattle Police
Department with crime incidents info in
each Seattle District.
"""
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter


def load_in_data(file_name):
    """
    This functions mainly load in the
    prepocessing data and filter down
    the data into Decemeber 2020.
    """
    df = pd.read_csv(file_name)
    df = df.loc[(df["Longitude"] != 0) & (df["Latitude"] != 0) &
                (df['Year'] == 2020) & (df['Month'] == 12)]
    return df


def parse_zipcode(location):
    """
    This function returns zipcode by converting
    the latitude and longtitude if exists,
    otherwise return None.
    """
    if location.raw.get('address') and location.raw['address'].get('postcode'):
        return location.raw['address']['postcode']
    else:
        return None


def add_zip_column(df):
    """
    This function add the zipcode information into
    dataset and convert the dataset in the format
    of csv file.
    """
    df['Zipcode'] = df['Location'].apply(parse_zipcode)
    file = df.to_csv('Crime_2020_With_Zip.csv', index=False)
    return file


def main():
    df = load_in_data('/data/cleaned_crime_data.csv')
    geolocator = Nominatim(user_agent="my_application")
    reverse = RateLimiter(geolocator.reverse,
                          min_delay_seconds=1)
    df['Location'] = df.apply(lambda row:
                              reverse((row['Latitude'],
                                       row['Longitude'])), axis=1)
    parse_zipcode(df['Location'])
    add_zip_column(df)


if __name__ == "__main__":
    main()
