from cse163_utils import assert_equals
import dataCleaning
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


def test_data_cleaning(crime_file_name):
    cleaned_df = dataCleaning.preprocess_crime_data(crime_file_name)
    assert_equals(True, len(cleaned_df) > 0)
    assert_equals(True, cleaned_df['Year'].min == 2016)


def test_graph_total_count(filename):
    df = pd.read_csv(filename)
    col = 'Offense Parent Group'
    top_10 = df.groupby(col)[col].count().nlargest(10)
    top_10 = pd.DataFrame(top_10)
    print(top_10)
    top_10_type = top_10.index.values.tolist()
    df = df.loc[df['Offense Parent Group'].isin(top_10_type)]
    df = df.groupby(['Offense Parent Group', 'Year']).size()
    sns.relplot(data=df, kind='line', x='Year',
                y=df.values, hue='Offense Parent Group')
    plt.xticks([2016, 2017, 2018, 2019, 2020, 2021, 2022])
    plt.ylabel('Counts')
    plt.title('Trend of Top 10 Crimes By Type')
    plt.savefig('Test file total counts graph')


def main():
    crime_file_name = 'data/SPD_Crime_Data__2008-Present.csv'
    test_data_cleaning(crime_file_name)
    test_graph_total_count('data/Test_file.csv')


if __name__ == '__main__':
    main()
