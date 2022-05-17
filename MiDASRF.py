"""
MiDAS - Mission Driven Artificial Lift Intelligent Systems

    MiDAS ESP Replacement Forecast User Interface

    Last modified March 2022

Permissions:
    This code and its documentation can be integrated with company
    applications provided the unmodified code and this notice
    are included.

    This code cannot be copied or modified in whole or part.

"""
# -*- coding: utf-8 -*-

import dash
import dash_bootstrap_components as dbc
from dash import dash_table
import dash_daq as daq
import pandas as pd
#  import numpy as np
from dash import Input, Output, State, dcc, html
from datetime import date
#  import plotly.express as px
#  import time
from dash.exceptions import PreventUpdate
import base64
import io
from dataprocessing import process_rul_life_data
import RFconfig
import esp_graphs
from lee_carter import run_forecasts
from os.path import exists

number_of_buckets = RFconfig.NUMBER_OF_BUCKETS
date_format = RFconfig.DATE_FORMAT
month_default = RFconfig.MONTH_DEFAULT
years_to_forecast = RFconfig.YEARS_TO_FORECAST
new_well_list_file = RFconfig.NEW_WELL_LIST_FILE

# default list of new wells
df_new_wells = pd.DataFrame([{'Year': x, 'NewWells': 100} for x in range(1980, 2100)])
df_new_wells.set_index('Year', inplace=True)

# read customized list of new wells if it exists
if exists(new_well_list_file):
    cust_new_wells_df = pd.read_csv(new_well_list_file)
else:
    cust_new_wells_df = pd.DataFrame()

if 'Year' in cust_new_wells_df.columns:
    cust_new_wells_df.set_index('Year', inplace=True)

# update with customized new wells
df_new_wells.update(cust_new_wells_df)
df_new_wells = df_new_wells.reset_index()

app = dash.Dash(external_stylesheets=[dbc.themes.CYBORG])

navbar = dbc.Navbar(dbc.Container([
    dbc.Row(
        [
            dbc.Col(html.Img(src='static/MiDAS_life.png', height='55px'), width=3),
            dbc.Col(html.H4('ESP Replacement Forecasting Tool')),
        ], justify='start', style={'width': '80%'}
    )], fluid=True),
    color='dark',
    dark=True,
)

current_month = date.today().month

if current_month < 8:
    current_year = date.today().year - 1
else:
    current_year = date.today()

# dbc.Spinner(,size='lg', color='primary', type='border', fullscreen=True)

database_info_panel = dbc.Row(html.Div('No database loaded', className='mb-0', id='db_stats'), style={'width': '50%'})

exp_graph_panel = dbc.Row(
    [
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(options=[
                        {'disabled': False, 'label': 'Units Pulled by Reason',
                         'title': 'Classified as Failures or Non Failures per year', 'value': 0},
                        {'disabled': False, 'label': 'Population MTBP and Pull Rate',
                         'title': 'Units Mean Time Between Pulls and Pull Rate per year', 'value': 5},
                        {'disabled': False, 'label': 'Units Pulled by Age Group Relative',
                         'title': 'Classified by age groups as percentage of the year', 'value': 1},
                        {'disabled': False, 'label': 'Units Pulled by Age Group Absolute',
                         'title': 'Classified by age groups as number of pulls', 'value': 2},
                        {'disabled': False, 'label': 'Units Operating by Age Group Relative',
                         'title': 'Classified by age groups as percentage of the year', 'value': 3},
                        {'disabled': False, 'label': 'Units Operating by Age Group Absolute',
                         'title': 'Classified by age groups as number of operating units', 'value': 4},
                        {'disabled': False, 'label': 'Population Pull Rate by Group',
                         'title': 'Units Pull Rate per year by age group', 'value': 6},
                        {'disabled': False, 'label': 'New installations per year',
                         'title': 'New ESP Wells installed per year', 'value': 7},
                    ], value=0, id='dd_exp_graph_selection', placeholder='Select Graph',
                    ), width=8
                ),
            ], align='start', style={'width': '50%'}
        ),
        dbc.Row(
            [
                dbc.Col(dbc.Label('Log Scale:')),
                dbc.Col(daq.BooleanSwitch(id='sw_log_scale', on=False)),
            ], align='start', style={'width': '28%'}
        ),
        html.Br(),
        dcc.Graph(id='data_exp_graph'),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(dbc.Label('Filters:'), width=2),
                dbc.Col(
                    [
                        dbc.Label('Years:'),
                        dcc.RangeSlider(0, 1, 1, value=[0, 1], id='rs_fig_year_range', allowCross=False, marks=None,
                                        tooltip={'placement': 'bottom', 'always_visible': True},
                                        )
                    ], width=10)
            ], align='start', style={'width': '50%'}
        ),
    ]
)

forecast_graph_panel = dbc.Row(
    [
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(options=[
                        {'disabled': False, 'label': 'Unit Replacement Forecast Graph',
                         'title': 'Unit replacement forecast by year', 'value': 0},
                        {'disabled': False, 'label': 'Unit Replacement Forecast Table',
                         'title': 'Unit replacement forecast by year', 'value': 1},
                        {'disabled': False, 'label': 'Change Factor by Year - k Factor',
                         'title': 'Reflects trend in time for replacement rate', 'value': 2},
                        {'disabled': False, 'label': 'Log Mean by Age Group - a Factor',
                         'title': 'Intrinsic Log Replacement Rate by group', 'value': 3},
                        {'disabled': False, 'label': 'Change Factor by Age Group - b Factor',
                         'title': 'Amount of change in a group per unit of k', 'value': 4},
                    ], value=0, id='dd_forecast_graph_selector', placeholder='Select Graph',
                    ), width=8
                ),
            ], align='start', style={'width': '50%'}
        ),
        dbc.Row(
            [
                dbc.Col(dbc.Label('Log Scale:')),
                dbc.Col(daq.BooleanSwitch(id='sw_log_scale2', on=False)),
            ], align='start', style={'width': '28%'}
        ),
        html.Br(),
        dcc.Graph(id='forecast_graph')
    ]
)

new_wells_table = dash_table.DataTable(
    id='table_new_wells',
    data=[
        {'Year': 2000, 'NewWells': 10},
        {'Year': 2001, 'NewWells': 10},
        {'Year': 2002, 'NewWells': 10}],
    columns=[{'id': 'Year', 'name': 'Year', 'editable': False},
             {'id': 'NewWells', 'name': 'New Wells', 'editable': True}],
    style_table={'overflowX': 'auto'},
    style_cell={
        'textAlign': 'center',
        'color': 'whitesmoke',
        'backgroundColor': 'lightsteelblue',
        # 'height': 'auto',
        # 'color': 'black',
        # 'border': '1px solid black',
    },
    style_header={
        'textAlign': 'center',
        'border': '1px solid black',
        'backgroundColor': 'lightsteelblue',
        'color': 'black',
        'fontWeight': 'bold'
    },
    style_data_conditional=[
        {
            'if': {
                'column_editable': True  # True | False
            },
            'backgroundColor': 'steelblue',
            # 'cursor': 'not-allowed'
        },
        {
            'if': {
                'column_editable': False  # True | False
            },
            'cursor': 'not-allowed'
        },
        # {'if': {'row_index': 'even'},
        # 'backgroundColor': 'rgb(220, 220, 220)'},
        {'if': {
            'state': 'active'  # 'active' | 'selected'
        },
            'color': 'black',
            'backgroundColor': 'white',
            'border': '1px solid red'}
    ],

    # editable=True,
    # css= [{
    #     'selector' : 'td.cell--selected, td.focused',
    #     'rule': 'background-color: rgb(41, 56, 55) !important;'
    # }, {
    #     'selector': 'td.cell--selected *, td.focused *',
    #     'rule': 'color: rgb(41, 56, 55) !important;'
    # }]

)

forecast_settings_panel = dbc.Row(
    [
        dbc.Label('New Wells :'),
        dbc.Row(new_wells_table, style={'width': '20%'}),
        html.Hr(),
        dbc.Label('Years to model : '),
        dbc.Row(dcc.RangeSlider(0, 1, 1, value=[0, 1], id='rs_forecast_years', allowCross=False, marks=None,
                                tooltip={'placement': 'bottom', 'always_visible': True}), style={'width': '50%'}),
        html.Hr(),
        dbc.Row(dbc.Button('Forecast', id='bt_run_forecast', color='primary', className='mb-3', n_clicks=0),
                justify='start', style={'width': '20%'}
                ),
        dbc.Spinner(dcc.Store(id='sto_forecast_table'),
                    size='md', color='primary', type='border', fullscreen=False)
    ]
)

upload_settings_panel = dbc.Row(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label('Month'),
                        dcc.Dropdown(
                            options=['January', 'February', 'March', 'April', 'May', 'June', 'July',
                                     'August', 'September', 'October', 'November', 'December'],
                            value=month_default,
                            id='dd_month')
                    ]
                ),
                dbc.Col(
                    [
                        dbc.Label('Year'),
                        dcc.Dropdown(
                            options=[x for x in range(current_year, current_year - 10, -1)],
                            value=current_year,
                            id='dd_year')
                    ]
                ),
            ], style={'width': '40%'}
        ),
        dbc.Label(html.Small('Select cut off Month and Year for data in the database file'), id='info_msg'),
        dcc.Upload(id='upload_esp_database',
                   children=dbc.Button('Upload', id='btn_upload', color='primary'),
                   accept='.csv', className='me-1'),
        dbc.Spinner(dcc.Store(id='sto_esp_mortality_table'),
                    size='md', color='primary', type='border', fullscreen=False),
        dcc.Store(id='sto_esp_database'),
    ]
)

forecast_accordion = dbc.Accordion(
    [
        dbc.AccordionItem(
            [
                dbc.Collapse(forecast_settings_panel, id='collapse_forecast_controls', is_open=False, ),
            ],
            title='Forecast Settings',
            item_id='acc_forecast_settings',
        ),
        dbc.AccordionItem(
            [
                dbc.Collapse(
                    [
                        dbc.Spinner(forecast_graph_panel, size='md', color='primary', type='border', fullscreen=False),
                    ], id='collapse_forecast_graph', is_open=False)
            ],
            item_id='acc_forecast_results',
            title='Forecast Results'
        )
    ],
    id='acc_forecast',
    active_item='acc_forecast_settings',
)

data_explore_accordion = dbc.Accordion(
    [
        dbc.AccordionItem(
            [
                dbc.Collapse(
                    [
                        dbc.Spinner(exp_graph_panel, size='md', color='primary', type='border', fullscreen=False),
                    ], id='collapse_exp_graph_panel', is_open=False)
            ],
            item_id='acc_exp_graph',
            title='Graphs'
        ),
        dbc.AccordionItem(
            [
                dbc.Collapse(database_info_panel, id='collapse_db_info', is_open=True),
            ],
            title='Database Information',
            item_id='acc_database_info',
        ),
        dbc.AccordionItem(
            [
                dbc.Collapse(upload_settings_panel, id='collapse_upload_controls', is_open=True),
            ],
            title='Upload Database',
            item_id='acc_upload_database',
        ),
    ],
    id='acc_data_explore',
    active_item='acc_upload_database',
)

tabs = dbc.Tabs(
    [
        dbc.Tab(data_explore_accordion, label='Data Exploration', id='graph_tab', tab_id='graph', disabled=False),
        dbc.Tab(forecast_accordion, label='Forecasting', id='forecast_tab', tab_id='forecast', disabled=False),
    ], id='tabs_'
)

app.layout = dbc.Container([
    navbar,
    tabs
], fluid=True)


@app.callback(
    [
        Output('info_msg', 'children'),
        Output('info_msg', 'color'),
        Output('sto_esp_database', 'data'),
        Output('sto_esp_mortality_table', 'data'),
    ],
    [
        Input('upload_esp_database', 'contents'),
        State('dd_month', 'value'),
        State('dd_year', 'value'),
    ])
def read_esp_database(contents, begin_forecast_month_name, begin_forecast_year):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df_raw_esp_database = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        required_columns = ['WELL', 'INSTALLATION DATE', 'FAILURE DATE', 'REMOVAL DATE']
        fields_exist = set(required_columns).issubset(set(df_raw_esp_database.columns))
        if not fields_exist:
            info_msg = html.Small('* Key columns are missing from this database file, please check and reload again')
            return info_msg, 'red', dash.no_update, dash.no_update
        else:
            df_esp_database, df_esp_mortality_table = \
                process_rul_life_data(df_raw_esp_database, begin_forecast_month_name, begin_forecast_year,
                                      number_of_buckets)

            #  mx_file_name = 'test_mortality_table.csv'
            #  df_esp_mortality_table.to_csv(mx_file_name, index=False)

            json_esp_database = df_esp_database.to_json(date_format='iso', orient='split')
            json_esp_mortality_table = df_esp_mortality_table.to_json(date_format='iso', orient='split')
            # print(df_esp_mortality_table.columns, df_esp_database.columns)
            info_msg = ''
    else:
        raise PreventUpdate
    return info_msg, dash.no_update, json_esp_database, json_esp_mortality_table


def get_db_details(df_esp_database):
    first_installation_year = pd.to_datetime(df_esp_database['INSTALLATION DATE'],
                                             format=date_format).min().year
    last_installation_year = pd.to_datetime(df_esp_database['INSTALLATION DATE'], format=date_format).max().year
    totalesps = df_esp_database.shape[0]
    failures = df_esp_database[df_esp_database['STATUS'] == 'FAILURE'].shape[0]
    nonfailures = df_esp_database[df_esp_database['STATUS'] == 'NONFAILURE'].shape[0]
    running = df_esp_database[df_esp_database['STATUS'] == 'RUNNING'].shape[0]

    table_body = html.Tbody([
        html.Tr([html.Td('Number of ESPs   '), html.Td(totalesps)]),
        html.Tr([html.Td('First installation  '), html.Td(first_installation_year)]),
        html.Tr([html.Td('Last installation   '), html.Td(last_installation_year)]),
        html.Tr([html.Td('Failures   '), html.Td(failures)]),
        html.Tr([html.Td('Non failures   '), html.Td(nonfailures, )]),
        html.Tr([html.Td('Currently Running   '), html.Td(running)]),
    ]),
    db_info_table = dbc.Table(table_body, bordered=True)

    return db_info_table


@app.callback(
    [
        Output('tabs_', 'active_tab'),
        Output('collapse_exp_graph_panel', 'is_open'),
        Output('collapse_forecast_controls', 'is_open'),
        Output('rs_fig_year_range', 'min'),
        Output('rs_fig_year_range', 'max'),
        Output('rs_fig_year_range', 'value'),
        Output('rs_forecast_years', 'min'),
        Output('rs_forecast_years', 'max'),
        Output('rs_forecast_years', 'value'),
        Output('db_stats', 'children'),
        Output('acc_data_explore', 'active_item'),
    ],
    [
        Input('sto_esp_database', 'data'),
        Input('sto_esp_mortality_table', 'data'),
        State('dd_year', 'value')
    ],
)
def update_controls(esp_data, mx_data, end_year):
    if esp_data is not None:
        df_esp_data = pd.read_json(esp_data, orient='split')
        df_mx_data = pd.read_json(mx_data, orient='split')
        #  years = df_esp_data['Year'].unique()
        rs_exp_minvalue = df_esp_data['Year'].min()
        rs_exp_maxvalue = end_year
        rs_exp_value = float(rs_exp_minvalue), float(rs_exp_maxvalue)
        rs_for_minvalue, rs_for_maxvalue, rs_for_value = update_forecast_slider(df_mx_data)

        file_details = html.Div(get_db_details(df_esp_data))
        active_tab = 'graph'
        active_accordion_item = 'acc_exp_graph'
        forecast_controls_open = True
        exp_graph_panel_open = True
    else:
        raise PreventUpdate
    return active_tab, exp_graph_panel_open, forecast_controls_open, \
           rs_exp_minvalue, rs_exp_maxvalue, rs_exp_value, \
           rs_for_minvalue, rs_for_maxvalue, rs_for_value, \
           file_details, active_accordion_item


def update_forecast_slider(df_raw_mx_data):
    # Check for complete mortality rates
    start_year = df_raw_mx_data[df_raw_mx_data.dx == 0].Year.max()
    filtered_mx_data = df_raw_mx_data[df_raw_mx_data['Year'] > start_year]

    # calculate year range to model
    rs_for_maxvalue = filtered_mx_data.Year.max() - RFconfig.YEARS_TO_TEST
    rs_for_minvalue = filtered_mx_data.Year.min()
    rs_for_value = float(rs_for_minvalue), float(rs_for_maxvalue)

    return rs_for_minvalue, rs_for_maxvalue, rs_for_value


@app.callback(
    [
        Output('collapse_forecast_graph', 'is_open'),
        Output('table_new_wells', 'data')
    ],
    [
        Input('rs_forecast_years', 'value'),
        Input('sto_forecast_table', 'data')
    ],
)
def update_forecast_controls(value, data):
    if id_of_trigger() is not None:
        open_forecast_graph = False
        if id_of_trigger() == 'sto_forecast_table':
            open_forecast_graph = True
        cond = (df_new_wells['Year'] > value[1]) & (df_new_wells['Year'] <= value[1] + years_to_forecast)
        new_wells_table_data = df_new_wells[cond].to_dict('records')
    else:
        raise PreventUpdate
    return open_forecast_graph, new_wells_table_data


def id_of_trigger():
    ctx = dash.callback_context
    if not ctx.triggered:
        comp_id = None
    else:
        comp_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # print(comp_id)
    return comp_id


@app.callback(
    [
        Output('acc_forecast', 'active_item'),
        Output('sto_forecast_table', 'data'),
    ],
    [
        Input('bt_run_forecast', 'n_clicks'),
        State('sto_esp_mortality_table', 'data'),
        State('rs_forecast_years', 'value'),
        State('table_new_wells', 'data')
    ],
)
def run_forecast_calculation(n_clicks, raw_mx_data, year_range, table_data):
    global df_new_wells
    if raw_mx_data is not None:
        df_raw_mx_data = pd.read_json(raw_mx_data, orient='split')
        min_year = year_range[0]
        max_year = year_range[1]
        years_to_model = max_year - min_year + 1

        # update customized list of wells
        df_added_new_wells = pd.DataFrame(table_data)
        df_added_new_wells['Year'] = df_added_new_wells['Year'].astype(float)
        df_added_new_wells['NewWells'] = df_added_new_wells['NewWells'].astype(float)
        df_added_new_wells.set_index('Year', inplace=True)
        df_new_wells.set_index('Year', inplace=True)
        df_new_wells.update(df_added_new_wells)
        df_new_wells = df_new_wells.reset_index()
        df_new_wells.to_csv(new_well_list_file, index=False)

        df_filtered_mx_data = df_raw_mx_data[
            (df_raw_mx_data['Year'] >= min_year) & (df_raw_mx_data['Year'] <= max_year)]

        if RFconfig.YEARS_TO_TEST > 0:
            mx_test_data = df_raw_mx_data[(df_raw_mx_data['Year'] > max_year)]
            df_mx_test_data = pd.DataFrame()
            for year in mx_test_data.Year.unique():
                year_mx_data = mx_test_data[mx_test_data['Year'] == year]
                df_dx_row = pd.DataFrame({
                    'Year': year,
                    'Dx': year_mx_data.dx.sum(),
                }, index=[0])
                df_mx_test_data = df_mx_test_data.append(df_dx_row, ignore_index=True)
        else:
            df_mx_test_data = pd.DataFrame()

        df_scenarios = run_forecasts(df_filtered_mx_data, min_year, max_year,
                                     df_new_wells, years_to_model, df_mx_test_data)

        # //TODO remove this code for saving temporary forecast table
        file_name = 'forecast_table.csv'
        df_scenarios.to_csv(file_name, index=False)

        json_forecast_data = df_scenarios.to_json(date_format='iso', orient='split')
        acc_active_item = 'acc_forecast_results'
    else:
        raise PreventUpdate
    return acc_active_item, json_forecast_data


@app.callback(
    Output('data_exp_graph', 'figure'),
    [
        Input('rs_fig_year_range', 'value'),
        Input('dd_exp_graph_selection', 'value'),
        Input('sw_log_scale', 'on'),
        State('sto_esp_database', 'data'),
        State('sto_esp_mortality_table', 'data'),
    ],
)
def update_exp_graph(year_range, graph_value, sw_log_scale, esp_data, mort_data):
    if esp_data is not None:
        if graph_value == 0:
            df = pd.read_json(esp_data, orient='split')
            fig = esp_graphs.pulls_graph(df, year_range, sw_log_scale)
        else:
            df = pd.read_json(mort_data, orient='split')
            if graph_value == 1:  # Pulls relative
                fig = esp_graphs.pulls_age_relative(df, year_range, sw_log_scale)
            elif graph_value == 2:  # Pulls absolute
                fig = esp_graphs.pulls_age_absolute(df, year_range, sw_log_scale)
            elif graph_value == 3:  # Operating absolute
                fig = esp_graphs.operating_relative(df, year_range, sw_log_scale)
            elif graph_value == 4:  # Operating absolute
                fig = esp_graphs.operating_absolute(df, year_range, sw_log_scale)
            elif graph_value == 5:  # MTBP Graph
                fig = esp_graphs.mtbp_graph(df, year_range, sw_log_scale)
            elif graph_value == 6:  # Pull rate graph
                fig = esp_graphs.pull_rate_group(df, year_range, sw_log_scale)
            else:  # New Wells
                fig = esp_graphs.new_wells_graph(df, year_range, sw_log_scale)

    else:
        raise PreventUpdate
    return fig


@app.callback(
    Output('forecast_graph', 'figure'),
    [
        Input('dd_forecast_graph_selector', 'value'),
        Input('sw_log_scale2', 'on'),
        Input('sto_forecast_table', 'data'),
        State('rs_forecast_years', 'value')
    ],
)
def update_for_graph(graph_value, sw_log_scale, forecast_data, year_range):
    if forecast_data is not None:
        df = pd.read_json(forecast_data, orient='split')
        if graph_value == 0:
            fig = esp_graphs.replacement_forecast_graph(df, year_range, sw_log_scale)
        elif graph_value == 1:
            fig = esp_graphs.replacement_forecast_table(df, year_range, sw_log_scale)
        elif graph_value == 2:  # K factor
            fig = esp_graphs.k_forecast(df, year_range, sw_log_scale)
        elif graph_value == 3:  # a factor
            fig = esp_graphs.a_factor(df, year_range, sw_log_scale)
        elif graph_value == 4:  # b factor
            fig = esp_graphs.b_factor(df, year_range, sw_log_scale)
        else:
            fig = esp_graphs.b_factor(df, year_range, sw_log_scale)
    else:
        raise PreventUpdate
    return fig


# def load_modal(clk1, clk2, is_open):
#     if clk1 or clk2:
#         return not is_open
#     return is_open
#
#
# app.callback(Output('db_upload_modal', 'is_open'),
#              [Input('ddm_load_database', 'n_clicks'),
#               Input('btn_close_modal', 'n_clicks'),
#               State('db_upload_modal', 'is_open')
#               ])(load_modal)

if __name__ == '__main__':
    app.run_server(debug=True)  # , port=7080)
