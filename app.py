
from jinja2.utils import markupsafe
from markupsafe import escape
import plotly as plt
import numpy as np
import pandas as pd
from IPython.display import Image
import geopandas as gpd
from dash import Dash, html, dcc, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

app = Dash(external_stylesheets=[dbc.themes.MINTY])

df = pd.read_csv("data/devfiltered.csv")

app.layout = dbc.Container([

    dbc.Row([
        dbc.Col(
            html.H1('Interactive data visualization project: Covid-19 Flight Map', className='text-center text-primary, mb-4')
        )

    ]),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                df['countryorigin'].unique(),
                value='Brazil',
                placeholder="Country",
                id='countryslider',
                style={'width': '50%'},
                clearable=False,
            )
        ], width={"size":10, "offset":1, "order":1}),
        dbc.Col([
            dcc.Checklist(
                df["Year"].unique(),
                value=[2019, 2020, 2021],
                id="yearslider",
            )
        ], width={"size":5, "offset":1, "order":2}),
        dbc.Col([
            dcc.Graph(
                id="pie",
                style={'display': 'inline-block'}
            ),
            dcc.Graph(
                id='graphwithdropdown', 
                style={"display": "inline-block"}
            )
        ], width={"size": 10, "offset":1, "order":3})
    ], justify='start')
], fluid=True) 


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
            dfbraz= dfbraz.groupby(["countryorigin", "countrydestination", "origlat", "destlat", "origlon", "destlon"], as_index=False).agg({"flights":"sum"})
            print(dfbraz)

    bfbar = dfbraz.nlargest(10,["flights"])
    fig2 = px.histogram(bfbar, x='flights', y="countrydestination", color="countrydestination", title="TOP 10 DESTINATIONS", labels={"countrydestination": "destinations"})
    fig2.update_layout(xaxis_title="sum of flights", yaxis_title="destinations", height=700, width=700, margin={'l': 0, 'b': 350, 't': 50, 'r': 10})

    sourcetodest = zip(dfbraz["origlat"], dfbraz["destlat"],
                     dfbraz["origlon"], dfbraz["destlon"],
                     dfbraz["flights"])

    for slat,dlat, slon, dlon, numflights in sourcetodest:
        fig.add_trace(go.Scattergeo(
                            lat = [slat,dlat],
                            lon = [slon, dlon],
                            mode = 'lines',
                            line = dict(width = np.sqrt(numflights)/100 , color="red")
                            ))

    fcounts = dfbraz["countrydestination"].values.tolist() + dfbraz["flights"].values.tolist()
    countries = dfbraz["countryorigin"].values.tolist()+dfbraz["countrydestination"].values.tolist()

    scatterhoverdata = [country + "\n flights: "+ str(flights) for country, flights in zip(countries, fcounts)]
 
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
                        opacity=0.5,))
    )

    fig.update_geos(
        visible=False, resolution=50, showcountries=True, countrycolor="black"
    )

    fig.update_layout(title_text="",
                      autosize=False,
                      height=900, width=900,
                      margin={"t":0,"b":0,"l":0, "r":0, "pad":100},
                      showlegend=False,
                      geo_scope="world",
                      geo= dict(scope='world', showland = True, countrywidth = 0.5, subunitwidth=0.5)
                      )

    return fig, fig2


if __name__ == '__main__':
    app.run_server(debug=True)