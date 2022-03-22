"""
MiDAS - Mission Driven Artificial Lift Intelligent Systems

    MiDAS ESP Replacement Forecast graph templates

    Last modified March 2022

Permissions:
    This code and its documentation can integrated with company
    applications provided the unmodified code and this notice
    are included.

    This code cannot be copied or modified in whole or part.
"""

import pandas as pd
import numpy as np
from dash import Input, Output, State, dcc, html
from datetime import date
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import RFconfig

years_to_forecast = RFconfig.YEARS_TO_FORECAST
confidence_interval = RFconfig.CONFIDENCE_INTERVAL

plotly_template = ['ggplot2', 'seaborn', 'simple_white', 'plotly',
                   'plotly_white', 'plotly_dark', 'presentation',
                   'xgridoff', 'ygridoff', 'gridon', 'none']

midas_template = plotly_template[5]


def pulls_graph(df, year_range, log_scale):
    # filter year range
    df_fig = df[(df.Year >= year_range[0]) & (df.Year <= year_range[1])]
    df_fig = df_fig[~df_fig['STATUS'].isin(['RUNNING'])]
    df_fig = df_fig[['Year', 'STATUS', 'WELL']]
    df_fig = df_fig.groupby(by=['Year', 'STATUS']).count().reset_index()
    fig = px.bar(df_fig, x='Year', y='WELL', color='STATUS',
                 # log_x=True, size='WELL', size_max=60,
                 # height=500, width=1100, template='simple_white',
                 #log_y=log_scale,
                 template=midas_template,
                 hover_name='STATUS',
                 hover_data={
                     'STATUS': False,
                     'Year': False,
                 },
                 title='Failure / NonFailure Pull Distribution',
                 labels=dict(
                     Year='Year', WELL='Number of Pulls',
                     STATUS='Pull Reason')
                 )
    fig.update_xaxes(type='category')
    fig.update_layout(font_family='Droid Sans,Balto,Arial',
                      # 'Arial', 'Balto', 'Courier New', 'Droid Sans',
                      # 'Droid Serif', 'Droid Sans Mono', 'Gravitas One',
                      # 'Old Standard TT', 'Open Sans', 'Overpass', 'PT Sans Narrow',
                      # 'Raleway', 'Times New Roman'.
                      legend=dict(orientation='h', title='Pull Reason',
                                  y=1.1, x=1, xanchor='right', yanchor='bottom'))
    # fig.update_xaxes(tickprefix='$', range=[2, 5], dtick=1)
    # fig.update_yaxes(range=[30, 90])
    return fig


def pulls_age_relative(df, year_range, log_scale):
    df_fig = df[(df.Year >= year_range[0]) & (df.Year <= year_range[1])]  # filter year range
    df_fig = df_fig[['Year', 'dx', 'Age2', 'Age3']]  # select columns
    df_fig = df_fig.sort_values(['Year', 'Age3'], ascending=(True, True))  # sort categories
    num_groups = len(df_fig.Age2.unique())
    total_year = []
    for year in range(year_range[0], year_range[1]):
        total_dx = df_fig[df_fig.Year == year].dx.sum()

        total_year.extend(np.repeat(total_dx, num_groups))
    df_fig = df_fig.assign(dx_pc=df_fig.dx / total_year)

    fig = px.bar(df_fig, x='Year', y='dx_pc', color='Age2',
                 # log_x=True,  # log scale
                 # size='XXX', size_max=60,
                 # height=500, width=1100, template='simple_white',  # dimensions and template
                 #log_y=log_scale,
                 template=midas_template,
                 hover_name='Age2',
                 hover_data={
                     'Age2': False,
                     'Year': False,
                     'dx_pc': ':.1%',
                 },
                 title='Distribution as % of Total Pulls',

                 labels=dict(
                     Year='Year', dx_pc='Proportion',
                     Age2='Age Group')
                 )
    fig.update_xaxes(type='category')
    fig.update_layout(
        yaxis=dict(
            tickformat=',.0%'
        ),
    )
    # fig.update_traces(hovertemplate='Year=%{x} , Age Group %{Age}, %{y:.2f}')
    fig.update_layout(font_family='Raleway, Droid Sans, Balto ,Arial',
                      # 'Arial', 'Balto', 'Courier New', 'Droid Sans',
                      # 'Droid Serif', 'Droid Sans Mono', 'Gravitas One',
                      # 'Old Standard TT', 'Open Sans', 'Overpass', 'PT Sans Narrow',
                      # 'Raleway', 'Times New Roman'.
                      legend=dict(orientation='h', title='Age Group',
                                  y=1.1, x=1, xanchor='right', yanchor='bottom'))
    return fig


def pulls_age_absolute(df, year_range, log_scale):
    df_fig = df[(df.Year >= year_range[0]) & (df.Year <= year_range[1])]  # filter year range
    df_fig = df_fig[['Year', 'dx', 'Age2', 'Age3']]  # select colums
    df_fig = df_fig.sort_values(['Year', 'Age3'], ascending=(True, True))  # sort categories

    fig = px.bar(df_fig, x='Year', y='dx', color='Age2',
                 # log_x=True,  # log scale
                 # size='XXX', size_max=60,
                 # height=500, width=1100,
                 #log_y=log_scale,
                 template=midas_template,
                 hover_name='Age2',
                 hover_data={
                     'Age2': False,
                     'Year': False
                 },
                 title='Distribution of Pulls',
                 labels=dict(
                     Year='Year', dx='Pulls',
                     Age2='Age Group')
                 )
    fig.update_xaxes(type='category')
    fig.update_layout(font_family='Raleway, Droid Sans, Balto ,Arial',
                      # 'Arial', 'Balto', 'Courier New', 'Droid Sans',
                      # 'Droid Serif', 'Droid Sans Mono', 'Gravitas One',
                      # 'Old Standard TT', 'Open Sans', 'Overpass', 'PT Sans Narrow',
                      # 'Raleway', 'Times New Roman'.
                      legend=dict(orientation='h', title='Age Group',
                                  y=1.1, x=1, xanchor='right', yanchor='bottom'))
    return fig


def operating_absolute(df, year_range, log_scale):
    df_fig = df[(df.Year >= year_range[0]) & (df.Year <= year_range[1])]  # filter year range
    df_fig = df_fig[['Year', 'units_year_start', 'Age2', 'Age3']]  # select columns
    df_fig = df_fig.sort_values(['Year', 'Age3'], ascending=(True, True))  # sort categories

    fig = px.bar(df_fig, x='Year', y='units_year_start', color='Age2',
                 # log_x=True,  # log scale
                 # size='XXX', size_max=60,
                 # height=500, width=1100, template='simple_white',  # dimensions and template
                 #log_y=log_scale,
                 template=midas_template,
                 hover_name='Age2',
                 hover_data={
                     'Age2': False,
                     'Year': False,
                 },
                 title='Units Operating at Year Start',
                 labels=dict(
                     Year='Year', units_year_start='Units',
                     Age2='Age Group')
                 )
    fig.update_xaxes(type='category')
    fig.update_layout(font_family='Raleway, Droid Sans, Balto ,Arial',
                      # 'Arial', 'Balto', 'Courier New', 'Droid Sans',
                      # 'Droid Serif', 'Droid Sans Mono', 'Gravitas One',
                      # 'Old Standard TT', 'Open Sans', 'Overpass', 'PT Sans Narrow',
                      # 'Raleway', 'Times New Roman'.
                      legend=dict(orientation='h', title='Age Group',
                                  y=1.1, x=1, xanchor='right', yanchor='bottom'))
    return fig


def operating_relative(df, year_range, log_scale):
    df_fig = df[(df.Year >= year_range[0]) & (df.Year <= year_range[1])]  # filter year range
    df_fig = df_fig[['Year', 'units_year_start', 'Age2', 'Age3']]  # select columns
    df_fig = df_fig.sort_values(['Year', 'Age3'], ascending=(True, True))  # sort categories

    num_groups = len(df_fig.Age2.unique())
    total_year = []
    for year in range(year_range[0], year_range[1]):
        total_units_year_start = df_fig[df_fig.Year == year].units_year_start.sum()

        total_year.extend(np.repeat(total_units_year_start, num_groups))
    df_fig = df_fig.assign(units_year_start_pc=df_fig.units_year_start / total_year)

    fig = px.bar(df_fig, x='Year', y='units_year_start_pc', color='Age2',
                 # log_x=True,  # log scale
                 # size='XXX', size_max=60,
                 # height=500, width=1100, template='simple_white',  # dimensions and template
                 #log_y=log_scale,
                 template=midas_template,
                 hover_name='Age2',
                 hover_data={
                     'Age2': False,
                     'Year': False,
                     'units_year_start_pc': ':.1%',
                 },
                 title='Distribution as % of Total Units Operating',
                 labels=dict(
                     Year='Year', units_year_start_pc='Proportion',
                     Age2='Age Group')
                 )
    fig.update_xaxes(type='category')
    fig.update_layout(font_family='Raleway, Droid Sans, Balto ,Arial',
                      # 'Arial', 'Balto', 'Courier New', 'Droid Sans',
                      # 'Droid Serif', 'Droid Sans Mono', 'Gravitas One',
                      # 'Old Standard TT', 'Open Sans', 'Overpass', 'PT Sans Narrow',
                      # 'Raleway', 'Times New Roman'.
                      legend=dict(orientation='h', title='Age Group',
                                  y=1.1, x=1, xanchor='right', yanchor='bottom'))
    return fig



def new_wells_graph(df, year_range, log_scale):
    df_fig = df[(df.Year >= year_range[0]) & (df.Year <= year_range[1])]  # filter year range
    df_fig = df_fig[['Year', 'new_wells', 'Age2']]

    fig = px.bar(df_fig, x='Year', y='new_wells', color='Age2',
                 # log_x=True,  # log scale
                 # size='XXX', size_max=60,
                 # height=500, width=1100, template='simple_white',  # dimensions and template
                 #log_y=log_scale,
                 template=midas_template,
                 hover_name='Age2',
                 hover_data={
                     'Age2': False,
                     'Year': False,
                 },
                 title='New Wells Installed per Year',
                 labels=dict(
                     Year='Year', new_wells='New Wells',
                     Age2='Age Group')
                 )
    fig.update_xaxes(type='category')
    fig.update_layout(font_family='Raleway, Droid Sans, Balto ,Arial',
                      # 'Arial', 'Balto', 'Courier New', 'Droid Sans',
                      # 'Droid Serif', 'Droid Sans Mono', 'Gravitas One',
                      # 'Old Standard TT', 'Open Sans', 'Overpass', 'PT Sans Narrow',
                      # 'Raleway', 'Times New Roman'.
                      legend=dict(orientation='h', title='Age Group',
                                  y=1.1, x=1, xanchor='right', yanchor='bottom'))
    return fig


def pull_rate_group(df, year_range, log_scale):
    df_fig = df[(df.Year >= year_range[0]) & (df.Year <= year_range[1])]  # filter year range
    df_fig = df_fig[['Year', 'mx', 'Age2']]

    fig = px.line(df_fig, x='Year', y='mx', color='Age2',
                  # log_x=True,  # log scale
                  # size='XXX', size_max=60,
                  # height=500, width=1100, template='simple_white',  # dimensions and template
                  log_y=log_scale,
                  template=midas_template,
                  hover_name='Age2',
                  hover_data={
                      'Age2': False,
                      'Year': False,
                      'mx' : ':.2f'
                  },
                  title='Equipment Pull Rate by age Group',
                  labels=dict(
                      Year='Year', mx='Age Group Pull Rate',
                      Age2='Age Group')
                  )
    fig.update_xaxes(type='category')
    fig.update_layout(font_family='Raleway, Droid Sans, Balto ,Arial',
                      # 'Arial', 'Balto', 'Courier New', 'Droid Sans',
                      # 'Droid Serif', 'Droid Sans Mono', 'Gravitas One',
                      # 'Old Standard TT', 'Open Sans', 'Overpass', 'PT Sans Narrow',
                      # 'Raleway', 'Times New Roman'.
                      legend=dict(orientation='h', title='Age Group',
                                  y=1.1, x=1, xanchor='right', yanchor='bottom'))
    return fig


def k_forecast(df, year_range, log_scale):
    df_filtered = df[(df.Year >= year_range[0]) & (df.Year <= year_range[1]+years_to_forecast)]  # filter year range

    df_fig_hist = df_filtered[df_filtered['scenario'].isin(['current'])]
    df_fig_hist = df_fig_hist[['Year', 'khat']]
    df_fig_hist = df_fig_hist.groupby(by=['Year']).mean().reset_index()

    df_fig_mean = df_filtered[df_filtered['scenario'].isin(['mean'])]
    df_fig_mean = df_fig_mean[['Year',  'khat']]
    df_fig_mean = df_fig_mean.groupby(by=['Year']).mean().reset_index()

    df_fig_upper = df_filtered[df_filtered['scenario'].isin(['ci_upper'])]
    df_fig_upper = df_fig_upper[['Year', 'khat_ci_upper']]
    df_fig_upper = df_fig_upper.groupby(by=['Year']).mean().reset_index()

    df_fig_lower = df_filtered[df_filtered['scenario'].isin(['ci_lower'])]
    df_fig_lower = df_fig_lower[['Year', 'khat_ci_lower']]
    df_fig_lower = df_fig_lower.groupby(by=['Year']).mean().reset_index()

    upper_ci = int((1 - confidence_interval) * 100)
    lower_ci = int((confidence_interval) * 100)

    fig = go.Figure()  # make_subplots(specs=[[{'secondary_y': False}]])

    fig.add_traces(
        [
            go.Scatter(name='Fitted', x=df_fig_hist['Year'], y=df_fig_hist['khat']),
            #go.Bar(name='Mean Forecast', x=df_fig_mean['Year'], y=df_fig_mean['dxhat']),
            go.Scatter(name='P' + str(upper_ci), x=df_fig_upper['Year'], y=df_fig_upper['khat_ci_upper']),
            go.Scatter(name='P50', x=df_fig_mean['Year'], y=df_fig_mean['khat']),
            go.Scatter(name='P' + str(lower_ci), x=df_fig_lower['Year'], y=df_fig_lower['khat_ci_lower']),
        ]
    )

    fig.update_layout(
                  template=midas_template,
                  title='<i>k</i> Factor - Change Factor By Year',
    )

    fig.layout.yaxis.title = '<i>k</i> factor'
    fig.update_layout(legend_title_text='Scenarios')
    fig.update_xaxes(type='category')
    fig.update_layout(font_family='Raleway, Droid Sans, Balto ,Arial',
                      # 'Arial', 'Balto', 'Courier New', 'Droid Sans',
                      # 'Droid Serif', 'Droid Sans Mono', 'Gravitas One',
                      # 'Old Standard TT', 'Open Sans', 'Overpass', 'PT Sans Narrow',
                      # 'Raleway', 'Times New Roman'.
                      # legend=dict(orientation='h', title='Age Group', y=1.1, x=1, xanchor='right', yanchor='bottom')
                      )
    return fig

def mtbp_graph(df, year_range, log_scale):
    df_fig = df[(df.Year >= year_range[0]) & (df.Year <= year_range[1])]  # filter year range
    df_fig = df_fig[['Year', 'MTBP', 'PR']]
    df_fig = df_fig.assign(PR1000=df_fig['PR']*365*1000)
    df_fig.PR1000 = df_fig.PR1000.round(0)
    df_fig = df_fig.assign(MTBPYear=df_fig['MTBP'] / 365)
    df_fig.MTBPYear = df_fig.MTBPYear.round(1)
    df_fig = df_fig.groupby(by=['Year']).mean().reset_index()

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{'secondary_y': True}]])

    fig.add_trace(go.Scatter(name='MTBP in years', x=df_fig['Year'], y=df_fig['MTBPYear']), secondary_y=False)
    fig.add_trace(go.Scatter(name='Pulls per 1000', x=df_fig['Year'], y=df_fig['PR1000']), secondary_y=True)

    if log_scale:
        fig.update_yaxes(type='log')
    fig.layout.yaxis.title = 'MTBP (Years)'
    fig.layout.yaxis2.title = 'Pulls per 1000 operating Units'
    fig.update_layout(
                  template=midas_template,
                  # hover_name='Year',
                  # hover_data={
                  #     'Year': False,
                  # },
                  title='Statistical Mean Time Between Pulls and Pull Rate',
    )
    fig.update_xaxes(type='category')
    fig.update_layout(font_family='Raleway, Droid Sans, Balto ,Arial',
                      # 'Arial', 'Balto', 'Courier New', 'Droid Sans',
                      # 'Droid Serif', 'Droid Sans Mono', 'Gravitas One',
                      # 'Old Standard TT', 'Open Sans', 'Overpass', 'PT Sans Narrow',
                      # 'Raleway', 'Times New Roman'.
                      #legend=dict(orientation='h', title='Age Group', y=1.1, x=1, xanchor='right', yanchor='bottom')
                      )
    return fig


def a_factor(df, year_range, log_scale):
    df_fig = df[(df.Year >= year_range[0]) & (df.Year <= year_range[1])]  # filter year range
    df_fig = df_fig[['a', 'Age2']]
    df_fig = df_fig.groupby(by=['Age2']).mean().reset_index()

    fig = px.bar(df_fig, x='Age2', y='a', color='Age2',
                  # log_x=True,  # log scale
                  # size='XXX', size_max=60,
                  # height=500, width=1100, template='simple_white',  # dimensions and template
                  # log_y=log_scale,
                 hover_name='Age2',
                 hover_data={
                     'Age2': False,
                     'a': ':.2f'
                 },

                  template=midas_template,
                  title='<i>a</i> factor - Log Mean by Age Group',
                  labels=dict(
                      a='<i>a</i> factor',
                      Age2='Age Group')
                  )
    fig.update_xaxes(type='category')
    fig.update_layout(font_family='Raleway, Droid Sans, Balto ,Arial',
                      # 'Arial', 'Balto', 'Courier New', 'Droid Sans',
                      # 'Droid Serif', 'Droid Sans Mono', 'Gravitas One',
                      # 'Old Standard TT', 'Open Sans', 'Overpass', 'PT Sans Narrow',
                      # 'Raleway', 'Times New Roman'.
                      legend=dict(orientation='h', title='Age Group',
                                  y=1.1, x=1, xanchor='right', yanchor='bottom'))
    return fig


def b_factor(df, year_range, log_scale):
    df_fig = df[(df.Year >= year_range[0]) & (df.Year <= year_range[1])]  # filter year range
    df_fig = df_fig[['b', 'Age2']]
    df_fig = df_fig.groupby(by=['Age2']).mean().reset_index()

    fig = px.bar(df_fig, x='Age2', y='b', color='Age2',
                  # log_x=True,  # log scale
                  # size='XXX', size_max=60,
                  # height=500, width=1100, template='simple_white',  # dimensions and template
                  log_y=log_scale,
                  hover_name='Age2',
                  hover_data={
                     'Age2': False,
                     'b': ':.2f'
                  },
                 template=midas_template,
                  title='<i>b</i> factor - Change Factor by Age Group',
                  labels=dict(
                      b='<i>b</i> factor',
                      Age2='Age Group')
                  )
    fig.update_xaxes(type='category')
    fig.update_layout(font_family='Raleway, Droid Sans, Balto ,Arial',
                      # 'Arial', 'Balto', 'Courier New', 'Droid Sans',
                      # 'Droid Serif', 'Droid Sans Mono', 'Gravitas One',
                      # 'Old Standard TT', 'Open Sans', 'Overpass', 'PT Sans Narrow',
                      # 'Raleway', 'Times New Roman'.
                      legend=dict(orientation='h', title='Age Group',
                                  y=1.1, x=1, xanchor='right', yanchor='bottom'))
    return fig


def replacement_forecast_graph(df, year_range, log_scale):
    df_filtered = df[(df.Year >= year_range[0]) & (df.Year <= year_range[1]+years_to_forecast)]  # filter year range

    df_fig_hist = df_filtered[df_filtered['scenario'].isin(['current'])]
    df_fig_hist = df_fig_hist[['Year', 'dx']]
    df_fig_hist = df_fig_hist.groupby(by=['Year']).sum().astype(int).reset_index()

    df_fig_mean = df_filtered[df_filtered['scenario'].isin(['mean'])]
    df_fig_mean = df_fig_mean[['Year',  'dxhat']]
    df_fig_mean = df_fig_mean.groupby(by=['Year']).sum().astype(int).reset_index()

    df_fig_upper = df_filtered[df_filtered['scenario'].isin(['ci_upper'])]
    df_fig_upper = df_fig_upper[['Year', 'dxhat']]
    df_fig_upper = df_fig_upper.groupby(by=['Year']).sum().astype(int).reset_index()


    df_fig_lower = df_filtered[df_filtered['scenario'].isin(['ci_lower'])]
    df_fig_lower = df_fig_lower[['Year', 'dxhat']]
    df_fig_lower = df_fig_lower.groupby(by=['Year']).sum().astype(int).reset_index()

    upper_ci = int((1 - confidence_interval) * 100)
    lower_ci = int((confidence_interval) * 100)

    #   Correct for interval crossing
    dxhatc = df_fig_mean['dxhat']
    dxhatc_ll = df_fig_lower['dxhat']
    dxhatc_up = df_fig_upper['dxhat']
    df_fig_lower['dxhat'] = np.where(dxhatc_ll > dxhatc, dxhatc, dxhatc_ll)
    df_fig_upper['dxhat'] = np.where(dxhatc_up < dxhatc, dxhatc, dxhatc_up)

    fig = go.Figure()  # make_subplots(specs=[[{'secondary_y': False}]])

    fig.add_traces(
        [
            go.Bar(name='History', x=df_fig_hist['Year'], y=df_fig_hist['dx']),
            #go.Bar(name='Mean Forecast', x=df_fig_mean['Year'], y=df_fig_mean['dxhat']),
            go.Scatter(name='P' + str(upper_ci), x=df_fig_upper['Year'], y=df_fig_upper['dxhat']),
            go.Scatter(name='P50', x=df_fig_mean['Year'], y=df_fig_mean['dxhat']),
            go.Scatter(name='P' + str(lower_ci), x=df_fig_lower['Year'], y=df_fig_lower['dxhat']),
        ]
    )

    fig.update_layout(
                  template=midas_template,
                  title='Replacement Forecast',
    )

    fig.layout.yaxis.title = 'Number of Replacements'
    fig.update_layout(legend_title_text='Scenarios')
    fig.update_xaxes(type='category')
    fig.update_layout(font_family='Raleway, Droid Sans, Balto ,Arial',
                      # 'Arial', 'Balto', 'Courier New', 'Droid Sans',
                      # 'Droid Serif', 'Droid Sans Mono', 'Gravitas One',
                      # 'Old Standard TT', 'Open Sans', 'Overpass', 'PT Sans Narrow',
                      # 'Raleway', 'Times New Roman'.
                      # legend=dict(orientation='h', title='Age Group', y=1.1, x=1, xanchor='right', yanchor='bottom')
                      )
    return fig


def replacement_forecast_table(df, year_range, log_scale):
    df_filtered = df[(df.Year >= year_range[0]) & (df.Year <= year_range[1]+years_to_forecast)]  # filter year range

    df_fig_hist = df_filtered[df_filtered['scenario'].isin(['current'])]
    df_fig_hist = df_fig_hist[['Year', 'dx']]
    df_fig_hist = df_fig_hist.groupby(by=['Year']).sum().astype(int).reset_index()

    df_fig_mean = df_filtered[df_filtered['scenario'].isin(['mean'])]
    df_fig_mean = df_fig_mean[['Year',  'dxhat']]
    df_fig_mean = df_fig_mean.groupby(by=['Year']).sum().astype(int).reset_index()

    df_fig_upper = df_filtered[df_filtered['scenario'].isin(['ci_upper'])]
    df_fig_upper = df_fig_upper[['Year', 'dxhat']]
    df_fig_upper = df_fig_upper.groupby(by=['Year']).sum().astype(int).reset_index()

    df_fig_lower = df_filtered[df_filtered['scenario'].isin(['ci_lower'])]
    df_fig_lower = df_fig_lower[['Year', 'dxhat']]
    df_fig_lower = df_fig_lower.groupby(by=['Year']).sum().astype(int).reset_index()

    upper_ci = int((1 - confidence_interval) * 100)
    lower_ci = int((confidence_interval) * 100)

    #   Correct for interval crossing
    dxhatc = df_fig_mean['dxhat']
    dxhatc_ll = df_fig_lower['dxhat']
    dxhatc_up = df_fig_upper['dxhat']
    df_fig_lower['dxhat'] = np.where(dxhatc_ll > dxhatc, dxhatc, dxhatc_ll)
    df_fig_upper['dxhat'] = np.where(dxhatc_up < dxhatc, dxhatc, dxhatc_up)

    upper_ci_str= 'P'+ str(upper_ci)
    lower_ci_str = 'P'+ str(lower_ci)

    df_fig_hist['history'] = df_fig_hist['dx']
    df_fig_mean['P50'] = df_fig_mean['dxhat']
    df_fig_upper[upper_ci_str] = df_fig_upper['dxhat']
    df_fig_lower[lower_ci_str] = df_fig_lower['dxhat']

    #df_table = df_fig_hist.join(df_fig_mean[['Year', 'P50']])
    df_table = df_fig_upper[['Year',upper_ci_str]].join(df_fig_mean['P50'])
    df_table = df_table.join(df_fig_lower[lower_ci_str])

    df_table_t = df_table.set_index('Year').T
    df_table_t.reset_index(inplace=True)
    df_table_t = df_table_t.rename(columns={'index': 'SCENARIO'})

    headers = list(df_table_t.columns)
    cell_values = [df_table_t[x] for x in df_table_t.columns]

    fig = go.Figure()  # make_subplots(specs=[[{'secondary_y': False}]])
    fig.add_traces(
        [
            go.Table(
                header=dict(values=headers,
                            font=dict(size=12),
                            align=['left', 'center'],
                            #fill_color='paleturquoise',
                            ),
                cells=dict(values=cell_values,
                           align=['left', 'center'],
                           #fill_color='lavender',
                           ))
        ]
    )

    fig.update_layout(
                  template=midas_template,
                  title='Replacement Forecast Table',
    )
    #
    # fig.layout.yaxis.title = 'Number of Replacements'
    # fig.update_layout(legend_title_text='Scenarios')
    # fig.update_xaxes(type='category')
    # fig.update_layout(font_family='Raleway, Droid Sans, Balto ,Arial',
    #                   # 'Arial', 'Balto', 'Courier New', 'Droid Sans',
    #                   # 'Droid Serif', 'Droid Sans Mono', 'Gravitas One',
    #                   # 'Old Standard TT', 'Open Sans', 'Overpass', 'PT Sans Narrow',
    #                   # 'Raleway', 'Times New Roman'.
    #                   # legend=dict(orientation='h', title='Age Group', y=1.1, x=1, xanchor='right', yanchor='bottom')
    #                   )
    return fig



