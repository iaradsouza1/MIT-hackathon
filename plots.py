import pickle
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import json
import streamlit as st

from datetime import date

@st.cache(suppress_st_warning=True)
def pickle_to_df(results_pkl, geojson):
    
    # Read 'results.pkl' file
    with open(results_pkl, 'rb') as posteriors, open(geojson) as geoj:
        df_hdi = pickle.load(posteriors)
        brasil = json.load(geoj)

        # Get geocodes
        df_geocode = dict()
        for i in brasil['features']:
            estado = i['properties']['uf_05']
            geocodigo = i['properties']['geocodigo']
            df_geocode[estado] = geocodigo
    
        # Create the columns 'state' and 'geocodes'
        for k in df_hdi.keys():
            df_hdi[k]['state'] = k
            df_hdi[k]['geocode'] = df_geocode[k]
            df_hdi[k] = df_hdi[k].reset_index()
        
        # Concatenate all dfs
        df_results = pd.concat([v for k, v in df_hdi.items()])

        return(df_results)

def plot_map(df_results, date, geojson):

    with open(geojson) as geoj:
        brasil = json.load(geoj)

    date = pd.to_datetime(date)

    df_tmp = df_results[df_results.date.eq(date)]

    fig = go.Figure(
        go.Choroplethmapbox(
            geojson=brasil, 
            locations=df_tmp.geocode, 
            text=df_tmp.state,
            z=df_tmp.ML,
            colorscale=px.colors.sequential.OrRd, 
            zmin=0, 
            zmax=4,
            featureidkey='properties.geocodigo',
            hoverinfo='text+z'
        )
    )

    fig.update_layout(
        mapbox_style="carto-positron", 
        mapbox_zoom=3, 
        mapbox_center={"lat": -15.5517, "lon": -58.7073},
        margin={"r":0,"t":0,"l":0,"b":0}
    )
    
    return(fig)

def plot_new_cases(state):

    original_path =  'metrics/' + str(state) + '_original.pkl'
    smoothed_path = 'metrics/' + str(state) + '_smoothed.pkl'
    with open(original_path, 'rb') as original, open(smoothed_path, 'rb') as smoothed:
        df_original = pickle.load(original)
        df_smoothed = pickle.load(smoothed)

    df = pd.DataFrame({
        'date': df_original.xs(state, level='state').index,
        'cases': df_original.xs(state, level='state').values,
        'smoothed': df_smoothed.xs(state, level='state').values
    })

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['cases'],
        name='Nº de casos',
        line=dict(
            color='rgb(90,128,216)'
        )
    ))

    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['smoothed'],
        name='Smoothed',
        line=dict(
            dash='dot',
            color='rgb(215,80,70)'
        ),
        hoverinfo='text+z'
    ))

    fig.update_layout(
        plot_bgcolor='white',
        xaxis_title='Dias',
        yaxis_title='Novos casos por dia',
    )

    fig.update_xaxes(showgrid=True, gridcolor='rgb(215,215,215)')
    fig.update_yaxes(showgrid=True, gridcolor='rgb(215,215,215)')

    return(fig)


def plot_rt_state(state, df_results):

    df_tmp = df_results[ df_results.state.eq(state) ]

    upper = go.Scatter(
        name = 'Upper bound',
        x = df_tmp.date,
        y = df_tmp.High,
        mode = 'lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        connectgaps=True
    )

    rt = go.Scatter(
        name = 'Rt',
        x = df_tmp.date,
        y = df_tmp.ML,
        mode = 'lines',
        line = dict(
            color = 'rgba(225,65,65,255)'
        ),
        connectgaps=True,
        fillcolor='rgba(154, 143, 200, 0.3)',
        fill='tonexty'
    )

    low = go.Scatter(
        name = 'Lower Bound',
        x = df_tmp.date,
        y = df_tmp.Low,
        marker = dict(color="#444"),
        line = dict(width = 0),
        mode='lines',
        connectgaps=True,
        fillcolor='rgba(154, 143, 200, 0.3)',
        fill='tonexty'
    )

    data = [upper, rt, low]

    layout = go.Layout(
        showlegend = False,
        plot_bgcolor='white',
        xaxis=dict(
            gridcolor='rgb(215,215,215)'
        ),
        yaxis = dict(
            title='Rt',
            gridcolor='rgb(215,215,215)'
        ),
        hovermode='x'
    )

    fig = go.Figure(data = data, layout = layout)

    return(fig)

