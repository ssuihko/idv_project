
from jinja2.utils import markupsafe
from markupsafe import escape
import plotly as plt
import numpy as np
import pandas as pd
import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from IPython.display import Image
import sys
import os
import geopandas as gpd
import cartopy.crs as ccrs
from dash import Dash, html, dcc, Input, Output, dash_table
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
#from dash_extensions.enrich import Output, Input, State, ServersideOutput

BS = "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"
app = Dash(external_stylesheets=[BS])

#fig2 = go.Figure()
# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

# df0 = pd.read_csv("f2019/dev2019.csv")
# df1 = pd.read_csv("f2019/dev2020.csv")
# df2 = pd.read_csv("f2019/dev2021.csv")

# df = pd.concat([df0, df1, df2])
# df = df.rename(columns={"orig_lon": "origlon","orig_lat": "origlat", "dest_lat":"destlat", "dest_lon":"destlon", "country_origin":"countryorigin", "country_destination": "countrydestination"})
# df["firstseen"] = pd.to_datetime(df["firstseen"])
# df["Year"] = df["firstseen"].dt.year
df = pd.read_csv("f2019/devfiltered.csv")

app.layout = html.Div(children=[

    html.H1(children='Interactive data visualization project'),

    html.Div(children='''
        Flight Map.
    '''),
    
    html.Div(children=[
    html.Div(dcc.Dropdown(
        df['countryorigin'].unique(),
        value='Brazil',
        placeholder="Country",
        id='countryslider',
        style={'width': '50%'}
    )),
    html.Div(dcc.Checklist(
        df["Year"].unique(),
        value=[2019, 2020, 2021],
        id="yearslider")
    ),
    dcc.Graph(
        id="pie",
        style={'display': 'inline-block'}
    ),
    dcc.Graph(
        id='graphwithdropdown', 
        style={"display": "inline-block"}
    )])
    #dcc.Loading(dcc.Store(id="store-data", data=[], storage_type="memory"), type="dot")
])

# @app.callback(
#     ServersideOutput('store-data', 'data'),
#     Input('countryslider', 'value'),
#     Input('yearslider', 'value2'))
# def query_data(value, value2):
#     time.sleep(1)
#     return px.data.gapminder()

@app.callback(
    Output('graphwithdropdown', 'figure'), 
    Output('pie', 'figure'), 
    Input('countryslider', 'value'), 
    Input('yearslider', 'value'))
def buildmap(countryin, yearin=None):


    print(countryin)
    fig = go.Figure()
    fig2 = go.Figure()

    dfbraz = df[df['countryorigin'] == countryin]

    if yearin is None and countryin is not None:
        yearin = [2019, 2020, 2021]

    if yearin is not None and countryin is not None:
        print(yearin)
        dfbraz = df[df['countryorigin'] == countryin]
        dfbraz = dfbraz[(dfbraz["Year"].isin(yearin))]
        if len(yearin) > 1:
            #print(dfbraz)
            dfbraz= dfbraz.groupby(["countryorigin", "countrydestination", "origlat", "destlat", "origlon", "destlon"], as_index=False).agg({"fcount":"sum"})
            print(dfbraz)
            #print(dfbraz[["countryorigin", "fcount"]])

    print("datashape: ", dfbraz.shape)

    #dfbraz['fcount'] = dfbraz['countrydestination'].map(dfbraz["countrydestination"].value_counts())

    bfpie = dfbraz.nlargest(15,["fcount"])
    fig2 = px.histogram(bfpie, x='fcount', y="countrydestination", color="countrydestination", title="TOP 15 DESTINATIONS")
    fig2.update_layout(height=700, width=700, margin={'l': 0, 'b': 350, 't': 50, 'r': 10})

    sourcetodest = zip(dfbraz["origlat"], dfbraz["destlat"],
                     dfbraz["origlon"], dfbraz["destlon"],
                     dfbraz["fcount"])

    ## Loop thorugh each flight entry to add line between source and destination
    for slat,dlat, slon, dlon, numflights in sourcetodest:
        fig.add_trace(go.Scattergeo(
                            lat = [slat,dlat],
                            lon = [slon, dlon],
                            mode = 'lines',
                            line = dict(width = np.sqrt(numflights)/100 , color="red")
                            ))

    ## Logic to create labels of source and destination cities of flights
    fcounts = dfbraz["countrydestination"].values.tolist() + dfbraz["fcount"].values.tolist()
    countries = dfbraz["countryorigin"].values.tolist()+dfbraz["countrydestination"].values.tolist()

    def fsum(fcount, country):
        if len(yearin) > 1:
            a = dfbraz.groupby(["countryorigin","countrydestination"])["fcount"].sum()
            print(a)
            print(a.columns())
            b = a[a["countryorigin"] == country]["fcount"]
            return b
        else: 
            return fcount

    scatterhoverdata = [country + "\n flights: "+ str(fcount) for country, fcount in zip(countries, fcounts)]

    ## Loop thorugh each flight entry to plot source and destination as points.
    fig.add_trace(
        go.Scattergeo(
                    lon = dfbraz["origlon"].values.tolist()+dfbraz["destlon"].values.tolist(),
                    lat = dfbraz["origlat"].values.tolist()+dfbraz["destlat"].values.tolist(),
                    hoverinfo = 'text',
                    text = scatterhoverdata,
                    mode = 'markers',
                    marker = dict(
                        size = 10, 
                        color = 'blue', 
                        opacity=0.1,))
    )

    ## Update graph layout to improve graph styling.
    fig.update_layout(title_text="",
                      autosize=False,
                      height=900, width=900,
                      margin={"t":0,"b":0,"l":0, "r":0, "pad":100},
                      showlegend=False,
                      #mapbox=dict(
                      #  accesstoken=mapbox_access_token,
                       # bearing=0,
                      #  pitch=0,
                      #  zoom=0,
                      #  style="open-street-map"
                     # )
                      geo= dict(scope='world', showland = True, landcolor = 'rgb(250, 250, 250)', countrycolor = 'rgb(250, 250, 250)', subunitcolor = 'rgb(217, 217, 217)',  countrywidth = 0.5, subunitwidth=0.5)
                      )

    return fig, fig2


if __name__ == '__main__':
    app.run_server(debug=True)