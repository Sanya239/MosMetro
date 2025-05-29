from dash import Dash, dcc, html, Input, Output, State
from datetime import datetime


def date_picker(min_date, max_date):
    return html.Div([
        html.Label("Дата:"),
        dcc.DatePickerRange(
            id='date-picker',
            min_date_allowed=min_date,
            max_date_allowed=max_date,
        ),
    ])

def output_size():
    return html.Div([
            html.Label("Размер вывода"),
            dcc.Input(id='top-number', type='text', value='20'),
        ], style={'marginTop': 10, 'marginBottom': 10})