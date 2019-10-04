import pandas as pd
import plotly.graph_objs as go
from boto3 import client
from pandas.io.json import json_normalize

# Use this file to read in your data and prepare the plotly visualizations. The path to the data files are in
# `data/file_name.csv`

c = client('dynamodb')

def cleandata(dataset, keepcolumns = ['Country Name', '1990', '2015'], value_variables = ['1990', '2015']):
    """Clean world bank data for a visualizaiton dashboard

    Keeps data range of dates in keep_columns variable and data for the top 10 economies
    Reorients the columns into a year, country and value
    Saves the results to a csv file

    Args:
        dataset (str): name of the csv data file

    Returns:
        None

    """
    df = pd.read_csv(dataset, skiprows=4)

    # Keep only the columns of interest (years and country name)
    df = df[keepcolumns]

    top10country = ['United States', 'China', 'Japan', 'Germany', 'United Kingdom', 'India', 'France', 'Brazil', 'Italy', 'Canada']
    df = df[df['Country Name'].isin(top10country)]

    # melt year columns  and convert year to date time
    df_melt = df.melt(id_vars='Country Name', value_vars = value_variables)
    df_melt.columns = ['country','year', 'variable']
    df_melt['year'] = df_melt['year'].astype('datetime64[ns]').dt.year

    # output clean csv file
    return df_melt


def return_figures():
    """Creates four plotly visualizations

    Args:
        None

    Returns:
        list (dict): list containing the four plotly visualizations

    """

    # first chart plots arable land from 1990 to 2015 in top 10 economies
    # as a line chart

    train_df = pd.read_csv('data/sentiment_analysis/train.tsv', sep="\t")
    graph_two = []

    class_count = train_df['Sentiment'].value_counts()
    x = class_count.index.tolist()
    y = class_count.values.tolist()

    graph_one = []


    graph_one.append(
        go.Pie(
            labels = x,
            values = y,
        )
    )

    layout_one = dict(title = 'Distribution of sentiments')

    # second chart plots ararble land for 2015 as a bar chart



    graph_two.append(
        go.Bar(
            x = x,
            y = y,
        )
    )

    layout_two = dict(title = 'Number of reviews per sentiment',
                      xaxis = dict(title = 'Sentiment',),
                      yaxis = dict(title = 'Number of reviews'),
                      )


    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))


    return figures




def get_item(username):
    response = c.get_item(
        TableName='item',
        Key={
            'username': username
        },
    )
    return response['Item']

