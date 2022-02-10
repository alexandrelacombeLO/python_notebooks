# -*- coding: utf-8 -*-
import base64
import io
from pathlib import Path

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.dash_table.Format import Format, Scheme
from dash.exceptions import PreventUpdate

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
import scipy.stats

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator
raw_symbols = SymbolValidator().values
symbol_sequence = [symbol for i, symbol in enumerate(raw_symbols) if i/12 == int(i/12) and symbol < 10]

color_sequence = px.colors.qualitative.Plotly.copy()
color_sequence.remove('#FECB52')
color_sequence.insert(0, '#FECB52')#E0BB42

theme = 'dark'
if theme == 'dark':
    font_color = 'white'
    assets_ignore = '.*light.*'
else:
    font_color = 'black'
    assets_ignore = '.*dark.*'

def ci95(data):
    n = len(data.dropna())
    se = scipy.stats.sem(data.dropna())
    ci = se * scipy.stats.t.ppf((1 + .95) / 2., n-1)
    return ci

chart_type = html.Div([
    html.Label("Chart Type"),
    dcc.Dropdown(
        id='chartType-dropdown', 
        value='Scatter',
        options=[{'label': c, 'value': c} for c in ['Boxplot', 'Scatter']],
        clearable=False
    ),
], style={'width':'10%', 'margin-right':'10px'})

title = html.Div([
    html.Label("Title"),
    dcc.Input(
        id='title-input', 
        type="text", 
        size='100',
        debounce=True
        )
], style={'width':'40%'})

x_axis = html.Div([
    html.Label("X-axis"),
    dcc.Dropdown(id='xaxis-dropdown')
], style={'width':'10%'})

y_axis = html.Div([
    html.Label("Y-axis"),
    dcc.Dropdown(id='yaxis-dropdown')
], style={'width':'10%'})

color = html.Div([
    html.Label("Color"),
    dcc.Dropdown(id='color-dropdown')
], style={'width':'10%'})

col = html.Div([
    html.Label("Columns"),
    dcc.Dropdown(id='col-dropdown')
], style={'width':'10%'})

row = html.Div([
    html.Label("Rows"),
    dcc.Dropdown(id='row-dropdown')
], style={'width':'10%'})

size = html.Div([
    html.Label("Size"),
    dcc.Dropdown(id='size-dropdown'),
], style={'width':'15%'})

symbol = html.Div([
    html.Label("Symbol"),
    dcc.Dropdown(id='symbol-dropdown'),
], style={'width':'15%'})

text = html.Div([
    html.Label("Text"),
    dcc.Dropdown(id='text-multidropdown', multi=True, value=[]),
], style={'width':'15%'})

text_position = html.Div([
    html.Label("Text position"),
    dcc.Input(
        id='textposition-input', 
        type="text", 
        size='50',
        debounce=True
        )
])

groupby = html.Div([
    html.Label("Groupby"),
    dcc.Dropdown(id='groupby-dropdown'),
], style={'width':'15%'})

hover_data = html.Div([
    html.Label("Hover data"),
    dcc.Dropdown(id='hover-multidropdown', multi=True, value=[])
], style={'width': '33%'})

reg = html.Div([
    html.Label("Linear Regression"),
    dcc.RadioItems(
        id='reg-radioitem',
        value='none',
        options=[{'label': c, 'value': c} for c in ['all', 'by color', 'none']],
        labelStyle={'display': 'inline-block', 'margin-right':'10px'}
    )
], style={'width': '15%'})         

errorbar = html.Div([
    html.Label("Show errorbars"),
    dcc.Checklist(
        id='errorbar-checklist',
        options=[{'label': 'Yes', 'value': 'Yes'}],
        value=['Yes']
    )
], style={'width': '12%'})

keep_cat_col = html.Div([
    html.Label('keep categorical columns'),
    dcc.Dropdown(id='keep_cat_col', multi=True, value=[])
], style={'width': '20%', 'margin-right':'10px'})

show_legend = html.Div([
    html.Label("Show legend"),
    dcc.Checklist(
        id='showlegend-checklist',
        options=[{'label': 'Yes', 'value': 'Yes'}],
        value=[]
    )
], style={'width': '12%'})

theme_select = dcc.RadioItems(
        id='theme-radioitem',
        value=theme,
        options=[{'label': c, 'value': c} for c in ['dark', 'light']],
        labelStyle={'margin-right':'10px'}
)

tabs_style = {
    'margin': 'auto',
    'height': '5%',
    'width': '30%',
    'borderRadius': '8px'
}

tab_style = {
    'backgroundColor': 'transparent',
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'fontWeight': 'bold',
    'padding': '6px',
}

app = dash.Dash(__name__, assets_ignore=assets_ignore)

app.layout = html.Div([
    html.Div(id='hidden-div', style={'display':'none'}),
    dcc.Store(id='memory'),
    dcc.Tabs(id='Tab', value='DataTable', children=[
        dcc.Tab(value='DataTable', label='DataTable', children=[
            dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '30%',
                'height': '40px',
                'lineHeight': '40px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '8px',
                'textAlign': 'center',
                'margin': '1% auto'
            },
            style_active={'backgroundColor': 'rgb(150, 150, 150)'},
            multiple=True
        ),
            html.Div([
                html.Div([
                    dcc.Input(
                        id='nomenclature',
                        type='text',
                        placeholder='Split with \'_\' separator',
                    ),
                    html.Button(
                        children='Split',
                        id='Split',
                        n_clicks=0,
                    ),
                    html.Button(
                        children='download',
                        id='download',
                        n_clicks=0,
                    )
                ],
                style={'display': 'flex', 'justify-content': 'flex-end', 'margin-bottom': '5px'}
                ),
            ]),
            dash_table.DataTable(
                id='df',
                style_as_list_view=True,
                style_cell={"whiteSpace": "pre-line"},
                style_header={
                    'height': 'auto',
                    'backgroundColor': 'rgb(200, 160, 50)' if font_color == 'white' else 'rgb(0, 60, 120)',
                    'color': 'white',
                    'font-size': '75%',
                    'font-family': 'Open Sans'
                },
                style_data={
                    'backgroundColor': 'rgb(50, 50, 50)' if font_color == 'white' else 'white',
                    'color': 'white'  if font_color == 'white' else 'black',
                    'font-size': '75%',
                    'font-family': 'Open Sans'
                },
                style_filter={
                    'backgroundColor': 'rgb(150, 150, 150)',
                }
            )
        ],
        style={'borderTopLeftRadius': '8px', 'borderBottomLeftRadius': '8px', **tab_style}, selected_style={'borderTopLeftRadius': '8px', 'borderBottomLeftRadius': '8px', **tab_selected_style},
        ),
        dcc.Tab(value='Plots', label='Plots', children=[
            html.Div(
                [
                    chart_type,
                    title,
                ],
                style={'display':'flex', 'flex-flow': 'row wrap', 'justify-content': 'stretch', 'margin-bottom': '10px', 'margin-top': '10px', 'font-size': '80%'}
            ),
            html.Div(
                [
                    x_axis,
                    y_axis,
                    color,
                    col,
                    row,
                    hover_data,
                    show_legend
                ], 
                style={'display':'flex', 'flex-flow': 'row wrap', 'justify-content': 'space-between', 'margin-bottom': '10px', 'margin-top': '10px', 'width': '100%', 'font-size': '80%'}
            ),
            html.Div(
                id='scatter_components', 
                children=[
                    groupby,
                    size,
                    symbol,
                    text,
                    text_position,
                    reg,
                ], style={'display': 'none'}
            ),
            html.Div(
                id='groupby_components', 
                children=[
                    keep_cat_col,
                    errorbar,
                ],
            ),
            html.Div(
                id='plot',
                children=[
                    dcc.Graph(id='graph'),
                    theme_select
                ])
        ], style={'borderTopRightRadius': '8px', 'borderBottomRightRadius': '8px', **tab_style}, selected_style={'borderTopRightRadius': '8px', 'borderBottomRightRadius': '8px', **tab_selected_style},
        )
    ], style=tabs_style
    )
])


@app.callback(
    Output('hidden-div', 'children'),
    [
        Input('download', 'n_clicks'),
        Input('df', 'selected_columns'),
        Input('df', 'data')
    ]
)
def download_pivot(n_clicks, cols, data):
    if n_clicks > 0 and data:
        if cols:
            df_cols = cols + ['RECORD', 'CROSS-SECTIONAL AREA (µm²)', 'ELASTIC MOD (MPa)', 'STRESS 15% (MPa)', 'STRESS 25% (MPa)', 'POSTYIELD MOD (MPa)', 'BREAK EXT (%)', 'BREAK STRESS (MPa)', 'TOUGHNESS (MJ/m\u00b3)']
            pivot_cols = [col for col in df_cols if not col in cols + ['RECORD']]

        else:
            df_cols = ['ID', 'RECORD', 'CROSS-SECTIONAL AREA (µm²)', 'ELASTIC MOD (MPa)', 'STRESS 15% (MPa)', 'STRESS 25% (MPa)', 'POSTYIELD MOD (MPa)', 'BREAK EXT (%)', 'BREAK STRESS (MPa)', 'TOUGHNESS (MJ/m\u00b3)']
            pivot_cols = [col for col in df_cols if not col in ['ID', 'RECORD']]

            
        df = pd.DataFrame(data)
        if 'L*(D65)' in df.columns:
            df_cols.append('L*(D65)')
            pivot_cols.append('L*(D65)')
        
        pivot = pd.pivot_table(
            df,
            index=cols if cols else ['ID'],
            values=pivot_cols,
            sort=False,
#             aggfunc=ci95
            )[pivot_cols]
        
        with pd.ExcelWriter(Path(r'C:\Users\lacombea\Downloads\HEV-INS-CLI-FR- Data.xlsx')) as writer:
            df[df_cols].to_excel(writer, sheet_name='RawData', float_format='%.1f')
            pivot.to_excel(writer, sheet_name='PivotTable', float_format='%.1f')
            
        raise PreventUpdate
    else:
        raise PreventUpdate


def parse_contents(content, filename):
    
    read_csv_params = {
        'sep': '\t',
        'skiprows': [0,1,4],
        'header': 0,
        'index_col': 0,
        'usecols': ['RECORD', 'CROSS-SECTIONAL AREA', 'MEAN DIAMETER', 'MIN DIAMETER', 'MAX DIAMETER', 'ELASTIC EMOD', 'ELASTIC EXT', 'STRESS 15%', 'STRESS 25%', 'PLATEAU STRESS', 'POSTYIELD GRADIENT', 'BREAK EXT', 'BREAK STRESS', 'TOUGHNESS'],
    #     'decimal': ','
    }
    
    content_type, content_string = content.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if '.txt' in filename:
            if filename == 'LAB.txt':
                name = filename.replace('.txt', '')
                content = pd.read_csv(io.StringIO(decoded.decode('latin-1')), sep='\t', decimal=',', index_col=False, header=0).rename(columns={'Description': 'ID'})
                content = content[~content.ID.str.contains('Moyenne')]
                content[['ID', 'index']] = content.ID.str.split(expand=True, pat='@')
                content = content.drop(columns='index')
                content.ID = content.ID.str.replace('.s-1', '/s')
            else:
                name = filename.replace('_modif', '').replace('.txt', '').replace('.s-1', '/s')
                content = pd.read_csv(io.StringIO(decoded.decode('latin-1')), **read_csv_params)
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return name, content


@app.callback(
    [
        Output('df', 'columns'),
        Output('df', 'data'),
        Output('df', 'editable'),
        Output('df', 'row_deletable'),
        Output('df', 'row_selectable'),
        Output('df', 'column_selectable'),
        Output('df', 'selected_columns'),
        Output('df', 'filter_action'),
        Output('df', 'sort_action'),
        Output('df', 'sort_mode'),
        Output('df', 'style_data_conditional'),
    ],
    [
        Input('upload-data', 'contents'),
        Input('Split', 'n_clicks')
    ],
    [
        State('upload-data', 'filename'),
        State('df', 'data'),
        State('nomenclature', 'value')
    ]
)
def update_output(list_of_contents, split_button, list_of_names, df_state, split_nomenclature):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        raise PreventUpdate
        
    elif ctx.triggered[0]['prop_id'].split('.')[0] == 'Split':
        df = pd.DataFrame(df_state)
        if 'ELASTIC MOD (MPa)' in df.columns and 'L*(D65)' in df.columns:
            ordered_columns = ['ID', 'RECORD', 'CROSS-SECTIONAL AREA (µm²)', 'MEAN DIAMETER (µm)', 'MIN DIAMETER (µm)', 'MAX DIAMETER (µm)', 'MEAN DEVIATION', 'ELASTIC EXT (%)', 'ELASTIC MOD (MPa)', 'PLATEAU STRESS (MPa)', 'STRESS 15% (MPa)', 'STRESS 25% (MPa)', 'POSTYIELD MOD (MPa)', 'BREAK EXT (%)', 'BREAK STRESS (MPa)', 'BREAK RATIO', 'TOUGHNESS (MJ/m\u00b3)', 'L*(D65)']
        elif 'ELASTIC MOD (MPa)' in df.columns:
            ordered_columns = ['ID', 'RECORD', 'CROSS-SECTIONAL AREA (µm²)', 'MEAN DIAMETER (µm)', 'MIN DIAMETER (µm)', 'MAX DIAMETER (µm)', 'MEAN DEVIATION', 'ELASTIC EXT (%)', 'ELASTIC MOD (MPa)', 'PLATEAU STRESS (MPa)', 'STRESS 15% (MPa)', 'STRESS 25% (MPa)', 'POSTYIELD MOD (MPa)', 'BREAK EXT (%)', 'BREAK STRESS (MPa)', 'BREAK RATIO',  'TOUGHNESS (MJ/m\u00b3)']
        else:
            ordered_columns = ['ID', 'L*(D65)']
        split = split_nomenclature.split('_')
        df[split] = df.ID.str.split(pat='_', expand=True)
        for ele in reversed(split):
            ordered_columns.insert(1, ele)
        columns = [{'name': col, 'id': col, 'selectable': True} for col in df[ordered_columns].columns]

    elif ctx.triggered[0]['prop_id'].split('.')[0] == 'upload-data':
        
        if list_of_contents:
            files = [parse_contents(c, n) for c, n in zip(list_of_contents, list_of_names)]
            dico_fichiers = {}
            LAB = False
            upload = False
            for name, content in files:
                if not name == 'LAB':
                    dico_fichiers[name] = content
                else:
                    LAB = content
           
            if dico_fichiers != {}:
                upload = pd.concat(objs=dico_fichiers, names=['ID']).dropna(how='all').reset_index().sort_values(['ID', 'RECORD'])
                # conversion de la colonne RECORD en format int64
                upload.astype({'RECORD': 'int64'})

                upload['ELASTIC EMOD'] /= 1e6
                upload['TOUGHNESS'] /= 1e6
                upload['ELLIPTICITY'] = upload['MAX DIAMETER'] / upload['MIN DIAMETER']
                upload['MEAN DEVIATION'] = abs(upload['MEAN DIAMETER'] - (upload['MAX DIAMETER'] + upload['MIN DIAMETER']) / 2)
                upload['POSTYIELD MOD (MPa)'] = upload['POSTYIELD GRADIENT'] * 30
                upload['BREAK RATIO'] = upload['BREAK STRESS'] / upload['BREAK EXT']

                upload = upload.drop(columns=['POSTYIELD GRADIENT'])

                upload = upload.rename(columns={
                    'CROSS-SECTIONAL AREA': 'CROSS-SECTIONAL AREA (µm²)',
                    'MEAN DIAMETER': 'MEAN DIAMETER (µm)',
                    'MIN DIAMETER': 'MIN DIAMETER (µm)',
                    'MAX DIAMETER': 'MAX DIAMETER (µm)',
                    'ELASTIC EXT': 'ELASTIC EXT (%)',
                    'ELASTIC EMOD': 'ELASTIC MOD (MPa)',
                    'PLATEAU STRESS': 'PLATEAU STRESS (MPa)',
                    'STRESS 15%': 'STRESS 15% (MPa)',
                    'STRESS 25%': 'STRESS 25% (MPa)',
                    'BREAK EXT': 'BREAK EXT (%)',
                    'BREAK STRESS': 'BREAK STRESS (MPa)',
                    'TOUGHNESS': 'TOUGHNESS (MJ/m\u00b3)'
                })
            
            
    
            if isinstance(LAB, pd.DataFrame):
                if dico_fichiers != {}:
                    upload = pd.concat([upload, LAB], ignore_index=True)
                else:
                    upload = LAB
                  
            if df_state: 
                df = pd.concat([pd.DataFrame(df_state), upload]).drop_duplicates() 
            else:
                if isinstance(upload, pd.DataFrame) or isinstance(LAB, pd.DataFrame):
                    df = upload
            
            if 'ELASTIC MOD (MPa)' in df.columns and 'L*(D65)' in df.columns:
                ordered_columns = ['ID', 'RECORD', 'CROSS-SECTIONAL AREA (µm²)', 'MEAN DIAMETER (µm)', 'MIN DIAMETER (µm)', 'MAX DIAMETER (µm)', 'MEAN DEVIATION', 'ELASTIC EXT (%)', 'ELASTIC MOD (MPa)', 'PLATEAU STRESS (MPa)', 'STRESS 15% (MPa)', 'STRESS 25% (MPa)', 'POSTYIELD MOD (MPa)', 'BREAK EXT (%)', 'BREAK STRESS (MPa)', 'BREAK RATIO', 'TOUGHNESS (MJ/m\u00b3)', 'L*(D65)']
            elif 'ELASTIC MOD (MPa)' in df.columns:
                ordered_columns = ['ID', 'RECORD', 'CROSS-SECTIONAL AREA (µm²)', 'MEAN DIAMETER (µm)', 'MIN DIAMETER (µm)', 'MAX DIAMETER (µm)', 'MEAN DEVIATION', 'ELASTIC EXT (%)', 'ELASTIC MOD (MPa)', 'PLATEAU STRESS (MPa)', 'STRESS 15% (MPa)', 'STRESS 25% (MPa)', 'POSTYIELD MOD (MPa)', 'BREAK EXT (%)', 'BREAK STRESS (MPa)', 'BREAK RATIO', 'TOUGHNESS (MJ/m\u00b3)']
            else:
                ordered_columns = ['ID', 'L*(D65)']
            
            columns = [{'name': col, 'id': col, 'selectable': True} for col in df[ordered_columns].columns]

        else:
            raise PreventUpdate
            
    columns = [{**col, 'type': 'numeric', 'format': Format(scheme=Scheme.decimal_integer)} if col['id'] in ['CROSS-SECTIONAL AREA (µm²)', 'ELASTIC MOD (MPa)', 'POSTYIELD MOD (MPa)', 'BREAK STRESS (MPa)'] else {**col} for col in columns]
    columns = [{**col, 'type': 'numeric', 'format': Format(precision=1, scheme=Scheme.fixed)} if col['id']  in ['MEAN DEVIATION', 'PLATEAU STRESS (MPa)', 'STRESS 15% (MPa)', 'STRESS 25% (MPa)', 'BREAK EXT (%)', 'TOUGHNESS (MJ/m\u00b3)'] else {**col} for col in columns]
    columns = [{**col, 'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.fixed)} if col['id']  in ['ELASTIC EXT (%)', 'BREAK RATIO', 'ELLIPTICITY'] else {**col} for col in columns]
    
    style_data_conditional=[
        {
            'if': {
                'filter_query': '{ELASTIC EXT (%)} <= 0.4 or {ELASTIC EXT (%)} >= 4',
            },
            'color': 'orange',
        },
        {
            'if': {
                'filter_query': '{ELASTIC EXT (%)} <= 0.4 or {ELASTIC EXT (%)} >= 4',
                'column_id': 'ELASTIC EXT (%)'
            },
            'fontWeight': 'bold'
        },
        {
            'if': {
                'filter_query': '{MEAN DEVIATION} >= 6',
            },
            'color': 'orange',
        },
        {
            'if': {
                'filter_query': '{MEAN DEVIATION} >= 6',
                'column_id': 'MEAN DEVIATION'
            },
            'fontWeight': 'bold'
        },
        {
            'if': {
                'filter_query': '{ELLIPTICITY} >= 2.5',
            },
            'color': 'orange',
        },
        {
            'if': {
                'filter_query': '{ELLIPTICITY} >= 2.5',
                'column_id': 'ELLIPTICITY'
            },
            'fontWeight': 'bold'
        },
    ]    
    
    return columns, df.to_dict('records'), True, True, 'multi', 'multi', [], 'native', 'native', 'multi', style_data_conditional

@app.callback(
    Output('memory', 'data'), 
    Input('df', 'data'),
)
def store_data(data):
    if data:
        return pd.DataFrame(data).to_dict('records')
    else:
        raise PreventUpdate

# Define callback to adapt dropdowns to boxplot or scatter
@app.callback(
    [
        Output('scatter_components', 'style'),
        Output('xaxis-dropdown', 'options'),
    ],
    Input('chartType-dropdown', 'value'),
    Input('memory', 'data'),
)
def set_dropdowns_layout(chartType, data):    
    if data:
        df = pd.DataFrame(data)
        if chartType == 'Boxplot':
            return {'display': 'none'}, \
                    [{'label': c, 'value': c} for c in df.select_dtypes(['object', 'category']).columns.to_list()]

        elif chartType == 'Scatter':
            return {'display': 'flex', 'flex-flow': 'row wrap', 'justify-content': 'space-between', 'margin-bottom': '10px', 'margin-top': '10px', 'width': '100%','font-size': '80%'}, \
                    [{'label': c, 'value': c} for c in df.select_dtypes(['float64', 'int64']).columns.to_list()]
        
    else:
        raise PreventUpdate

# Define callback to select all dropdowns options
@app.callback(
    [
        Output('yaxis-dropdown', 'options'),
        Output('color-dropdown', 'options'),
        Output('col-dropdown', 'options'),
        Output('row-dropdown', 'options'),
        Output('hover-multidropdown', 'options'),
        Output('size-dropdown', 'options'),
        Output('symbol-dropdown', 'options'),
        Output('text-multidropdown', 'options'),
        Output('groupby-dropdown', 'options'),
    ],
    [
        Input('chartType-dropdown', 'value'),
        Input('Tab', 'value'),
        Input('memory', 'data'),
    ]
)
def set_dropdowns_options(chartType, tab, data):
    if tab == 'Plots' and data:
            # Set DataFrame
            df = pd.DataFrame(data)
            # options for numeric columns
            num = [{'label': c, 'value': c} for c in df.select_dtypes(['float64', 'int64']).columns.to_list()]
            # options for categorical columns
            cat = [{'label': c, 'value': c} for c in df.select_dtypes(['object', 'category']).columns.to_list()]
            # options for all columns
            return num, cat+num if chartType == 'Scatter' else cat, cat, cat, cat+num, num, cat, cat, cat
    else:
        raise PreventUpdate
    
# Define callback to disable groupby settings if no groupby is selected
@app.callback(
        [
            Output('groupby_components', 'style'),
            Output('keep_cat_col', 'options'),
            Output('keep_cat_col', 'value'),
        ],
        [
            Input('groupby-dropdown', 'value'),
            Input('chartType-dropdown', 'value'),
            Input('memory', 'data'),
        ]
)
def set_groupby_settings(groupby, chartType, data):
    if groupby and chartType == 'Scatter':
        df = pd.DataFrame(data)
        list_cat = df.select_dtypes(['object', 'category']).columns.to_list()
        list_cat.remove(groupby)
        dict_cat = [{'label': c, 'value': c} for c in list_cat]
        return {'display': 'flex', 'flex-flow': 'row wrap', 'justify-content': 'flex-start', 'align-items': 'center', 'margin-bottom': '10px', 'margin-top': '10px', 'font-size': '80%'}, dict_cat, list_cat
    else:
        return {'display': 'none'}, [], []

# delete x-axis dropdown value if chart type is changed
@app.callback(
    Output('xaxis-dropdown', 'value'),
    Output('color-dropdown', 'value'),
    Input('chartType-dropdown', 'value')
)
def delete_dropdowns_content(aaa):
    return None, None

# Define callback to update graph
@app.callback(
    [
        Output('graph', 'figure'),
        Output('graph', 'config'),
        Output('plot', 'style')
    ],
    [
        Input('chartType-dropdown', 'value'),
        Input('title-input', 'value'),
        Input('xaxis-dropdown', 'value'),
        Input('yaxis-dropdown', 'value'),
        Input('color-dropdown', 'value'),
        Input('col-dropdown', 'value'),
        Input('row-dropdown', 'value'),
        Input('hover-multidropdown', 'value'),
        Input('showlegend-checklist', 'value'),
        Input('size-dropdown', 'value'),
        Input('symbol-dropdown', 'value'),
        Input('text-multidropdown', 'value'),
        Input('textposition-input', 'value'),
        Input('groupby-dropdown', 'value'),
        Input('reg-radioitem', 'value'),
        Input('errorbar-checklist', 'value'),
        Input('keep_cat_col', 'value'),
        Input('memory', 'data'),
        Input('theme-radioitem', 'value')
    ],
    State('graph', 'figure')
)
# ___________________________________________________________________ Draw plot function ___________________________________________________________________
def update_figure(chart_type, title, x, y, color, facet_col, facet_row, hover_data, showlegend, size, symbol, text, textposition, groupby, reg, errorbar, keep_cat_col,  df_state, theme, fig):
    
    if x and y:

        # html.Div style for showing plot and theme selector
        style={'display':'flex', 'flex-flow': 'row wrap', 'justify-content': 'stretch', 'margin-bottom': '10px', 'margin-top': '10px', 'font-size': '80%'}
        
        # dark or light theme for plot (the whole app theme cannot be easily updated, cannot change css file in callback)
        if theme == 'dark':
            font_color = 'white'
            plot_bgcolor = 'rgba(0, 0, 0, 0)'
            gridcolor = '#999999'
            # color_sequence = px.colors.qualitative.Set3
        else:
            font_color = 'black'
            plot_bgcolor = None
            gridcolor = 'white'
            # color_sequence = px.colors.qualitative.Plotly
        
        data = pd.DataFrame(df_state)
            
        N_cols = 1 if facet_col is None else data[facet_col].nunique()
        N_rows = 1 if facet_row is None else data[facet_row].nunique()
        
        # 'Save picture' button properties
        config = {
            'toImageButtonOptions': {
                'format': 'png', # one of png, svg, jpeg, webp
                'filename': 'custom_image',
                'height': 432 + 216 * N_cols,
                'width': 768 + 384 * N_rows,
                # 'height': 500 * N_rows * .75 if facet_row else 500,
                # 'width': (500 * N_rows * .75 if facet_row else 500) * 16 / 9,
            }
        }       

        layout =  {
            'title': {     
                'text': title,
                'font_size': 20,
                'font_color': font_color,
                'x':.5,
                'y': .99,
                'xanchor': 'center',
                'yanchor': 'top',     
            },
            'legend': {
                'bgcolor': 'rgba(0, 0, 0, 0)',
                'title': {'text': '', 'font_size': 18, 'font_color': font_color, 'side': 'top'},
                'tracegroupgap': 20,
                'font_size': 16, 
                'font_color': font_color,
                'y': .5, 
                'xanchor': 'left',
                'yanchor': 'middle', 
                'orientation': 'v'
            },
            'coloraxis': {'colorbar': {'tickfont_color': font_color, 'title_font_color': font_color}},
            'margin': {
                'l': 20,
                'r': 20,
                'b': 20,
                't': 20 if not title else 40
            },
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            'plot_bgcolor': plot_bgcolor,
        }
        
        xaxes = {
            'titlefont_size': 18,
            'titlefont_color': font_color,
            'tickfont_size': 16,
            'tickfont_color': font_color,
            'showgrid': True if chart_type == 'Scatter' else False,
            'gridcolor': None if chart_type == 'Box' else gridcolor,
            'hoverformat': '.1f',
            'matches': None if facet_col else 'x'
        }

        yaxes = {
            'titlefont_size': 18,
            'titlefont_color': font_color,
            'tickfont_size': 16,
            'tickfont_color': font_color,
            'gridcolor': gridcolor,
            'hoverformat': '.1f',
            'matches': None if facet_row else 'y'
        }
        
        if chart_type == 'Scatter':
        
            traces = {
                'legendgrouptitle_font': {'size': 14, 'color': font_color},
                'textposition': textposition.split(', ') if textposition else 'top right', 
                'textfont_size': 14,
                'textfont_color': font_color,
                'error_x_thickness': 1,
                'error_y_thickness': 1,
            }
            
        else:
            traces = {
                'boxpoints': 'all',
                'marker_size': 3,
                'line_width': 1,
                'boxmean': True, #'sd' pour afficher les écart-types
                'pointpos': 0,
                'jitter': .75,
                'showlegend': True if showlegend else False
            } 
            
        ctx = dash.callback_context
        
        if not ctx.triggered:
            raise PreventUpdate
 
        elif ctx.triggered[0]['prop_id'].split('.')[0] in ['theme-radioitem', 'title-input', 'textposition-input', 'showlegend-checklist']:
            fig = go.Figure(fig)
        
        else:

            plot_arguments = {
                'x': x,
                'y': y,
                'color': color,
                'color_discrete_sequence': color_sequence,
                'facet_col': facet_col,
                'facet_row': facet_row,
                'hover_data': hover_data,
                'height': 720,
                'width': 1280
            }

            if chart_type == 'Boxplot':
                
                fig = px.box(
                    data,
                    **plot_arguments,
                )
                # if not color: fig.for_each_trace(lambda c: c.update(marker_color='rgb(224,187,66)'))
                
            elif chart_type == 'Scatter':
                              
                if data[x].dtypes in ['int64', 'float64']:

                    if groupby:
                        group_by = [groupby] + keep_cat_col
                        columns = [x, y, color, facet_col, facet_row, size, symbol] + hover_data + group_by + text
                        columns = list(set([col for col in columns if not col is None]))
                        data = data[columns].groupby(group_by, as_index=False, sort=False, observed=True)
                        if errorbar:
                            errorbars = data.agg(ci95)
                        data = data.mean()                        

                    fig = px.scatter(
                        data,
                        **plot_arguments,
                        error_x=errorbars[x] if groupby and errorbar else None,
                        error_y=errorbars[y] if groupby and errorbar else None,
                        range_color=[20, 70] if color == 'L*(D65)' else None,
                        color_continuous_scale='turbid_r' if color == 'L*(D65)' else 'solar',
                        opacity=1 if color == 'L*(D65)' else None,
                        size=size,
                        symbol=symbol,
                        symbol_sequence=symbol_sequence,
                        text=[f"{' '.join(val)}" for val in list(zip(*[list(data[x]) for x in text]))] if text else None,
                    )

                    if x == 'BREAK STRESS (MPa)':
                        fig.add_vrect(
                            x0=0,
                            x1=80,
                            annotation_text='<b>Break zone</b>',
                            annotation_textangle=270,
                            annotation_position='bottom right',
                            annotation_font_color='rgb(230, 0, 0)',
                            fillcolor='rgba(230, 0, 0, 0.2)',
                            line_width=0
                        )
                        fig.add_vrect(
                            x0=80,
                            x1=100,
                            annotation_text='<b>Alert zone</b>',
                            annotation_textangle=270,
                            annotation_position='bottom right',
                            annotation_font_color='rgb(255, 170, 0)',
                            fillcolor='rgba(255, 170, 0, 0.2)',
                            line_width=0
                        )
                    
                    if y == 'BREAK STRESS (MPa)':
                        fig.add_hrect(y0=0,
                                      y1=80,
                                      annotation_text='<b>Break zone</b>',
                                      annotation_position='top left',
                                      annotation_font_color='rgb(230, 0, 0)',
                                      fillcolor='rgba(230, 0, 0, 0.2)',
                                      line_width=0)
                        fig.add_hrect(y0=80,
                                      y1=100,
                                      annotation_text='<b>Alert zone</b>',
                                      annotation_position='top left',
                                      annotation_font_color='rgb(255, 170, 0)',
                                      fillcolor='rgba(255, 170, 0, 0.2)',
                                      line_width=0)
                        
                    if color:
                        if data[color].dtypes.name in ['object', 'category']:
                            for i, ele in enumerate(data[color].unique()):
                                c = i
                                while c >= 10:
                                    c -= 10
                                fig.add_trace(go.Scatter(
                                    x=[None],
                                    y=[None],
                                    mode='markers',
                                    marker=dict(color=color_sequence[c]),
                                    name=ele,
                                    legendgroup=color,
                                    legendgrouptitle_text=color
                                ))
                        else:
                            fig.update_layout(legend_x=1.15, coloraxis_colorbar_title_font_color=font_color, coloraxis_colorbar_tickfont_color=font_color)
                            
                    if symbol:
                        for i, ele in enumerate(data[symbol].unique()):
                            fig.add_trace(go.Scatter(
                                x=[None],
                                y=[None],
                                mode='markers',
                                marker=dict(symbol=symbol_sequence[i], color=font_color),
                                name=ele,
                                legendgroup=symbol,
                                legendgrouptitle_text=symbol,
                            ))
                                           
                    if not size: fig.update_traces(marker_size=12)
                        
                    # linear regression if asked (and possible)
                    if not reg == 'none':

                        model = LinearRegression()
                        if not(facet_col and facet_row):
                            if color and reg == 'by color':

                                x_range = []
                                y_range = []
                                score = []

                                for i, ele in enumerate(data[color].unique()):
                                    query = data.query(f"{color} == '{ele}'")[[x, y]].dropna()
                                    if query.size > 2:

                                        model.fit(query[[x]], query[y])

                                        x_range.append(np.linspace(query[x].min(), query[x].max(), 100))
                                        y_range.append(model.predict(x_range[i].reshape(-1, 1)))

                                        score.append(model.score(query[[x]], query[y]))

                                        fig.add_traces(go.Scatter(
                                            x=x_range[i],
                                            y=y_range[i],
                                            line_color=px.colors.qualitative.Plotly[i],
                                            opacity=.75,
                                            legendgroup='reg'
                                        ))
                                fig.add_annotation(
                                    x=.05,
                                    xanchor='left',
                                    xref='paper',
                                    y=.95,
                                    yanchor='top',
                                    yref='paper',
                                    showarrow=False,
                                    text='<br>'.join([f"R² = {'{:.3f}'.format(sc)}" for sc in score])
                                )

                            else:
                                query = data[[x,y]].dropna()
                                model.fit(query[[x]], query[y])
                                x_range = np.linspace(query[x].min(), query[x].max(), 100)
                                y_range = model.predict(x_range.reshape(-1, 1))
                                score = model.score(query[[x]], query[y])

                                fig.add_traces(go.Scatter(
                                    x=x_range,
                                    y=y_range, 
                                    line_color=font_color,
                                    opacity=.75,
                                    legendgroup='reg'
                                ))
                                
                                fig.add_annotation(
                                    x=.05,
                                    xanchor='left',
                                    xref='paper',
                                    y=.95,
                                    yanchor='top',
                                    yref='paper',
                                    showarrow=False,
                                    text=f"R² = {'{:.3f}'.format(score)}"
                                )
                else:
                    raise PreventUpdate
            else:
                raise PreventUpdate
         
        fig.update_layout(**layout).update_xaxes(**xaxes).update_yaxes(**yaxes).update_traces(**traces)
        fig.for_each_annotation(lambda a: a.update(font_size=16, font_color=font_color) if not 'zone' in a.text else a.update(font_size=16))
        if chart_type == 'Scatter': 
            fig.for_each_trace(
                lambda t: t.update(
                    marker_color=font_color if t.legendgroup == symbol else t.marker.color,
                    line_color=font_color if t.legendgroup == 'reg' else t.line.color,
                    showlegend=(True if t.legendgroup in [color, symbol] else False) if showlegend else False
                )
            )
        
            
    else:
        #  Empty figure and config
        fig = go.Figure()#.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)', xaxis_visible=False, yaxis_visible=False)
        config = {}
        # html.Div style for showing / hiding plot and theme selector
        style = {'display': 'none'}

    return fig, config, style

if __name__ == '__main__':
    app.run_server(debug=True)