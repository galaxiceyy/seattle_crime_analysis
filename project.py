"""
CSE 163
A file that contains multiple functions
that make several visualizations to analyze
the Seattle crime dataset.
"""
# pip install geopandas
# pip install folium
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
sns.set()


def crimeMap(crimeData, geoData):
    """
    This function takes the grouped data
    and plots the scatter plot to indicate
    relationship between household income
    and crime rate in a file 'income_crime_plot.png'.
    """
    df = pd.read_csv(crimeData).dropna()
    mcpp = gpd.read_file(geoData)
    dis = df.groupby('MCPP', as_index=False).size()
    merged_df = dis.merge(mcpp, left_on='MCPP',
                          right_on='NAME', how='outer').dropna()
    merged_df.rename({'size': 'Incidents'}, axis=1, inplace=True)
    merged_df = gpd.GeoDataFrame(merged_df)
    seaMap = folium.Map([47.6062, -122.3321], zoom_start=9)
    # Create map
    folium.TileLayer('CartoDB positron', name="Light Map",
                     control=False).add_to(seaMap)
    choropleth = folium.Choropleth(
        geo_data=merged_df,
        fill_opacity=0.5,
        line_weight=1.5,
        data=merged_df,
        columns=['MCPP', 'Incidents'],
        key_on='feature.properties.MCPP',
        highlight=True,
        fill_color='YlOrRd',
        nan_fill_opacity=0,
        name='Incidents',
    ).add_to(seaMap)
    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(['MCPP', 'Incidents'], labels=True)
    )
    seaMap.save('vis/seaChoropleth.html')


def income_crime_plot(group_data):
    """
    This function takes the grouped data
    and plots the scatter plot to indicate
    relationship between household income
    and crime rate in a file 'income_crime_plot.png'.
    """
    group_data = pd.read_csv(group_data)
    sns.lmplot(x="Avg. Income/H/hold", y="Crime Rate", data=group_data)
    plt.title('Household Income vs Crime Rate')
    plt.xlabel('Household Income')
    plt.ylabel('Crime Rate')
    plt.xticks(rotation=-45, fontsize=8)
    plt.savefig('vis/income_crime_plot.png')
    corr, _ = pearsonr(group_data['Avg. Income/H/hold'],
                       group_data['Crime Rate'])
    print('Pearsons correlation: %.3f' % corr)


def income_bar_plot(income_file):
    """
    This function takes the grouped data
    and plots the bar chart to indicate
    the area with relatively low income
    in a file 'income_plot.png'.
    """
    income = pd.read_csv(income_file,
                         usecols=["Zip Code",
                                  "Avg. Income/H/hold",
                                  "Population"]).dropna()
    string = 'Avg. Income/H/hold'
    income[string] = income[string].str.replace(',', '')
    income[string] = income[string].str.replace('$', '').astype(float)
    top_5_area = income.sort_values(by=string)
    top_5_area = top_5_area.loc[(top_5_area[string] > 0)]
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.barplot(x="Zip Code", y=string,
                data=top_5_area.head(), ax=ax)
    plt.title('Top 5 Low Household Income in Seattle', fontsize=15)
    plt.xlabel('Area')
    plt.ylabel('Income', fontsize=15)
    plt.xticks(rotation=-45, fontsize=10)
    plt.savefig('vis/income_plot.png')


def prediction(crimeData):
    """
    This function takes the cleaned crime data
    train a machine learning model with offense
    parent group as label and MCPP, year, month,
    day, and hour as features.
    Print out the accurancy for both training and
    testing sets.
    And produce a graph with accuracy and for both groups
    as y, and max depth as x.
    """
    data = pd.read_csv(crimeData)

    data = data[["Offense Parent Group", 'MCPP',
                 'Year', 'Month', 'Day', 'Hour']]
    data = data[data['Year'] >= 2014]

    labels = data['Offense Parent Group']

    features = data.loc[:, data.columns != 'Offense Parent Group']
    features = pd.get_dummies(features)
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)

    print(len(features_train), len(features_test))

    model2 = DecisionTreeClassifier(max_depth=2)

    model2.fit(features_train, labels_train)

    train_predictions = model2.predict(features_train)
    print('Train Accuracy:', accuracy_score(labels_train, train_predictions))

    # Compute test accuracy
    test_predictions = model2.predict(features_test)
    print('Test  Accuracy:', accuracy_score(labels_test, test_predictions))

    accuracies = []
    for i in range(1, 20):
        model = DecisionTreeClassifier(max_depth=i, random_state=1)
        model.fit(features_train, labels_train)

        pred_train = model.predict(features_train)
        train_acc = accuracy_score(labels_train, pred_train)

        pred_test = model.predict(features_test)
        test_acc = accuracy_score(labels_test, pred_test)

        accuracies.append({'max depth': i, 'train accuracy': train_acc,
                           'test accuracy': test_acc})
    accuracies = pd.DataFrame(accuracies)

    def plot_accuracies(accuracies, column, name):
        """
        Parameters:
            * accuracies: A DataFrame show the train/test accuracy
            * for various max_depths
            * column: Which column to plot (e.g., 'train accuracy')
            * name: The display name for this column (e.g., 'Train')
        """
        sns.relplot(kind='line', x='max depth', y=column, data=accuracies)
        plt.title(f'{name} Accuracy as Max Depth Changes')
        plt.xlabel('Max Depth')
        plt.ylabel(f'{name} Accuracy')
        plt.ylim(0, 1)

        plt.show()  # Display the graph

    # Plot the graphs
    plot_accuracies(accuracies, 'train accuracy', 'Train')
    plot_accuracies(accuracies, 'test accuracy', 'Test')
    importance = dict(zip(features_train.columns, model2.feature_importances_))
    print(importance)


def crime_cases_trend(filename, export_name):
    """
    Take in the cleaned plotting data and a file name.
    Plot the number fo cases for each crime by year.
    Export the plot with the name passed in
    """
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
    plt.savefig(export_name)


def pie_chart_most_common(filename, export_name):
    """
    Take in the cleaned plotting data and a file name.
    Plot a pie chart that shows the proportion of each crime
    Export the plot with the name passed in
    """
    df = pd.read_csv(filename)
    df = df.groupby('Offense Parent Group')['Offense Parent Group'].count()
    plt.pie(x=df.values, radius=1.6)
    plt.legend(labels=df.index, fontsize=5, loc='upper right')
    plt.title('Crime_Count')
    plt.savefig(export_name)


def graph_top_ten_university_by_timezone(filename):
    df = pd.read_csv(filename)
    col = 'Offense Parent Group'
    top_10 = df.groupby(col)[col].count().nlargest(10)
    top_10 = pd.DataFrame(top_10)
    print(top_10)
    top_10_type = top_10.index.values.tolist()
    df = df.loc[df['Offense Parent Group'].isin(top_10_type)]
    df = df.groupby(['Offense Parent Group', 'Timezone']).size()
    sns.relplot(data=df, x='Timezone', y=df.values,
                hue='Offense Parent Group', kind='line')
    plt.xticks(['1:Earlymonring', '2:Morning', '3:Afternoon', '4:Night'])
    plt.ylabel('Counts')
    plt.savefig('Toptencrimesinuniversitybytimezone.png')


def main():
    income_crime_plot('data/income_crime_rate.csv')
    income_bar_plot('data/income/median_income.csv')
    crimeMap('data/cleaned_crime_data.csv', 'data/mapping/cleaned_dis.shp')
    prediction('data/cleaned_crime_data.csv')
    crime_cases_trend('data/plot_data.csv', 'vis/seattle.png')
    crime_cases_trend('data/university.csv', 'vis/university.png')
    pie_chart_most_common('data/plot_data.csv', 'vis/seattle_pie.png')
    pie_chart_most_common('data/university.csv', 'vis/university_pie.png')
    graph_top_ten_university_by_timezone('data/university.csv')


if __name__ == "__main__":
    main()
