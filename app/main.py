import logging
print("Starting application")

print("Importing modules")
from datetime import datetime
import duckdb
import pandas as pd
import plotly.express as px
import polars as pl
from dash import Dash, dcc, html, Input, Output, State
from waitress import serve
print("Imports complete")

logging.getLogger('waitress').setLevel(logging.ERROR)
data_folder = 'data1/'
csv_path = data_folder + "big_data.csv"  # путь к жирному файлу
parquet_path = csv_path.replace('.csv', '.parquet')


def read_parquet_with_duckdb(data_path: str, no_metro=False, no_overground=False):
    where_clause = ""
    if no_overground:
        where_clause += " WHERE TRANSPORT_TYPE_ID = 1 "
    if no_metro:
        where_clause += " WHERE TRANSPORT_TYPE_ID > 1 "

    query = f"""
        SELECT *
        FROM (
            SELECT
                TRAN_DATE,
                DEVICE_NO,
                TRANSPORT_TYPE_ID,
                PLACE_ID,
                BUS_RT_NO,
                ROW_NUMBER() OVER () AS row_num
            FROM '{data_path}'
            {where_clause}
        ) AS sub
    """

    result = (duckdb.query(query).pl().
    with_columns([
        pl.col("TRAN_DATE").dt.date().alias("TRAN_ONLY_DATE"),
        pl.col("TRAN_DATE").dt.time().alias("TRAN_ONLY_TIME")
    ]).with_columns(
        pl.when(
            (pl.col("TRANSPORT_TYPE_ID") == 1) & (pl.col("BUS_RT_NO").is_null())
        )
        .then(-239)
        .otherwise(pl.col("BUS_RT_NO"))
        .alias("BUS_RT_NO")).with_columns(
        pl.col("TRAN_DATE").dt.strftime("%A").alias("DAY_NAME")
    ))
    return result

print("Reading data")
big_data = read_parquet_with_duckdb(parquet_path)
overground = read_parquet_with_duckdb(parquet_path, no_metro=True)
underground = read_parquet_with_duckdb(parquet_path, no_overground=True)
print("Data loaded")


def show(fig):
    fig.update_layout(
        coloraxis_showscale=False,
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "buttons": [{
                "label": "Play",
                "method": "animate",
                "args": [None, {
                    "frame": {"duration": 2000, "redraw": True},  # 1000 мс = 1 секунда на кадр
                    "fromcurrent": True,
                    "transition": {"duration": 500, "easing": "linear"}  # плавность
                }]
            }, {
                "label": "Pause",
                "method": "animate",
                "args": [[None], {
                    "frame": {"duration": 0, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 0}
                }]
            }]
        }],
        sliders=[{
            "transition": {"duration": 500},
            "currentvalue": {"pref"
                             "ix": "День: "},
            "pad": {"t": 30},
            "len": 0.9
        }],
        height=500,
        width=1200,
    )

    fig.show()


def filter_by_time(data, start_dt, end_dt, start_tm, end_tm, continuous):
    if continuous:
        filtered = data.filter(
            (pl.col("TRAN_DATE") >= datetime.strptime(start_dt + " " + start_tm, "%Y-%m-%d %H:%M")) &
            (pl.col("TRAN_DATE") <= datetime.strptime(end_dt + " " + end_tm, "%Y-%m-%d %H:%M"))
        )
    else:
        filtered = data.filter(
            (pl.col("TRAN_ONLY_DATE") >= datetime.strptime(start_dt, "%Y-%m-%d").date()) &
            (pl.col("TRAN_ONLY_DATE") <= datetime.strptime(end_dt, "%Y-%m-%d").date()) &
            (pl.col("TRAN_ONLY_TIME") >= datetime.strptime(start_tm, "%H:%M").time()) &
            (pl.col("TRAN_ONLY_TIME") <= datetime.strptime(end_tm, "%H:%M").time())
        )
    return filtered





def load_top_stops_overground(top=20, start_dt="2025-03-10", end_dt="2025-03-16", start_tm="00:00",
                              end_tm="23:59", continuous=False):
    place_lookup = pl.read_csv("data1/REF_PSG_PLACES_202503251822.csv", separator=';')
    # Фильтрация по времени
    try:
        filtered = filter_by_time(overground, start_dt, end_dt, start_tm, end_tm, continuous)
    except:
        return "Wrong data format"
    # Топ остановок
    top_stops = (
        filtered.join(
            place_lookup.select(["PLACE_ID", "NAME", "Название линии"]),
            left_on="PLACE_ID",
            right_on="PLACE_ID",
            how="left"
        )
        .group_by("NAME")
        .agg(pl.len().alias("count"))
        .sort("count", descending=[True])
        .drop_nulls()
        .with_columns(
            pl.col("count").rank("dense", descending=True).alias("rank")
        )
        .filter(pl.col("rank") <= top))

    return top_stops


def load_top_stops_underground(top=20, start_dt="2025-03-10", end_dt="2025-03-16", start_tm="00:00",
                               end_tm="23:59", continuous=False):
    try:
        place_lookup = pl.read_csv(data_folder + "REF_PSG_PLACES_202503251822.csv", separator=';')
    except FileNotFoundError:
        return "No file to look up places: REF_PSG_PLACES_202503251822.csv"

    try:
        filtered = filter_by_time(underground, start_dt, end_dt, start_tm, end_tm, continuous)
    except:
        return "Wrong data format"
    top_stops = (
        filtered.join(
            place_lookup.select(["PLACE_ID", "NAME", "Название линии"]),
            left_on="PLACE_ID",
            right_on="PLACE_ID",
            how="left"
        )
        .group_by("PLACE_ID", "NAME", "Название линии")
        .agg(pl.len().alias("count"))
        .sort("count", descending=[True])
        .drop_nulls()
        .with_columns(
            pl.col("count").rank("dense", descending=True).alias("rank")
        )
        .filter(pl.col("rank") <= top))

    return top_stops


def load_top_routes_overground(top=20, start_dt="2025-03-10", end_dt="2025-03-16", start_tm="00:00",
                               end_tm="23:59", continuous=False, popularity=False):
    route_lookup = pl.read_csv(data_folder + "REF_TRANSPORT_WAY_202503251803.csv", separator=';')
    transport_type_lookup = pl.read_csv(data_folder + "REF_TRANSPORT_TYPE_202503251727.csv", separator=';')

    filtered = filter_by_time(overground, start_dt, end_dt, start_tm, end_tm, continuous)
    top_routes = (
        filtered.join(
            route_lookup.select(["WAY_ID", "NAME", "TRANSPORT_ID"]),
            left_on="BUS_RT_NO",
            right_on="WAY_ID",
            how="left").filter(pl.col("BUS_RT_NO").is_not_null())
        .group_by([
            "BUS_RT_NO", "NAME", "TRANSPORT_ID",
        ])
        .agg([pl.len().alias("count"),
              pl.col("DEVICE_NO").n_unique().alias("vehicle_count")
              ])
        .drop_nulls()
        .with_columns([
            (pl.col("count") / pl.col("vehicle_count")).alias("rides_per_vehicle")
        ])
        .with_columns([
            pl.col("count").rank("dense", descending=True).alias("popularity"),
            pl.col("rides_per_vehicle").rank("dense", descending=True).alias("overload")]
        ).join(
            transport_type_lookup, left_on="TRANSPORT_ID", right_on="TRANSPORT_ID", how="left"
        )
    )
    if popularity:
        return top_routes.filter(pl.col("popularity") <= top)
    else:
        return top_routes.filter(pl.col("overload") <= top)


def plot(data: pl.DataFrame, x: str, y: str, title="", xaxis ="", yaxis="",
         color=None, do_show=False):
    df_pandas = data.to_pandas()
    if color == None:
        fig = px.bar(
            df_pandas,
            x=x,
            y=y,
            title=title,
        )
    else:
        fig = px.bar(
            df_pandas,
            x=x,
            y=y,
            title=title,
            color=color,
        )

    fig.update_layout(
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        xaxis_categoryorder="total descending",
        xaxis_tickangle=-45,
        margin=dict(t=50, b=100),
        coloraxis_showscale=False
    )
    if do_show:
        show(fig)

    return fig


def output_size():
    return html.Div([
        dcc.Input(id='top-number', type='text', value='20'),
        html.Span("Размер вывода", className=""),
    ], className="field border max", style={'padding': 10})


def date_picker(min_date, max_date):
    return html.Div([
        html.Div([
            dcc.DatePickerRange(
                id='date-picker',
                min_date_allowed=min_date,
                max_date_allowed=max_date,
                start_date=min_date,
                end_date=max_date,
                className="center transparent-datepicker",
                style={'zIndex': 9999,},
            ), ]),
        html.Span(
            "Дата от/до",
            className="",
            style={
                'display': 'block',
                'textAlign': 'center',
                'margin': '0 auto'
            }),
    ], className="column", style={'padding': 10})


def time_picker():
    return html.Div([
        html.Div([
            html.Div([
                html.Div([
                    dcc.Input(id='start-time', type='text', value='00:00', className=""),
                    html.Span("Начало", className=""),
                ], className="field border max")
            ], className="max"),
            html.Div([
                html.Div([
                    dcc.Input(id='end-time', type='text', value='23:59', className=""),
                    html.Span("Конец", className=""),
                ], className="field border max")
            ], className="max"),
        ], className="row")
    ], style={'padding': 10})


def time_segment_type():
    html.Div([
        html.Label("Тип временного отрезка"),
        dcc.RadioItems(

            labelStyle={'display': 'inline-block', 'marginLeft': '15px'},
            className='horizontal',
        ),
    ], style={'marginBottom': 10})

    return html.Div([
        html.Div([
            dcc.Dropdown(
                id='time-type',
                options=[
                    {'label': 'Непрерывно', 'value': 'h0'},
                    {'label': 'Кусочно', 'value': 'h1'},
                ],
                value='h0',
                className="beer",
                style={'backgroundColor': 'transparent'},
            ),
            html.Span("Выбор отсчёта времени", className="",
                      style={'display': 'inline-block', 'marginTop': '-15px'}),
        ], className="field border")
    ], className="max", style={'padding-right': 10})


def graph_type():
    return html.Div([
        html.Div([
            dcc.Dropdown(
                id='data-source',
                options=[
                    {'label': 'Маршруты популярность', 'value': 't0'},
                    {'label': 'Маршруты перегруженность', 'value': 't1'},
                    {'label': 'Остановки', 'value': 't2'},
                    {'label': 'Метро популярность', 'value': 't3'},
                ],
                value='t0',  # Начальное значение
                className="beer",
                style={'backgroundColor': 'transparent'},

            ),
            html.Span("Выбор графика", className="", style={'display': 'inline-block', 'marginTop': '-15px'}),
        ], className="field border")
    ], className="max", style={'padding-left': 10})


def submit_button():
    return html.Div([html.Button("Построить график", id='submit-button', n_clicks=0),
                     ], className="")


def layout(df):
    return html.Div([
        html.Div(className="s2"),
        html.Div([
            html.Br(),
            html.H3(
                "Инфографика транспорта Москвы",
                className="center",
                style={"color": "orange", "textAlign": "center"},
            ),

            html.Fieldset([
                date_picker(df["TRAN_ONLY_DATE"].min(), df["TRAN_ONLY_DATE"].max()),
                time_picker(),
                html.Br(),

                html.Div([
                    graph_type(),
                    time_segment_type(), ], className="row"),
                html.Br(),
                html.Div([
                    output_size(),
                    submit_button(), ], className="row"),
                html.Br(),

            ]),
            dcc.Graph(id='graph', style={'width': '100%'}), ]
            , className="s8"),
        html.Div(className="s2"),

    ], className="grid")


def application(df):
    app = Dash(__name__)
    app.layout = layout(df)

    @app.callback(
        Output('graph', 'figure'),
        Input('submit-button', 'n_clicks'),
        State('date-picker', 'start_date'),
        State('date-picker', 'end_date'),
        State('start-time', 'value'),
        State('end-time', 'value'),
        State("data-source", "value"),
        State('top-number', 'value'),
        State('time-type', 'value'),
    )
    def update_graph(n_clicks, start_date, end_date, start_time_str, end_time_str, data_source, top_number, time_type):
        if not n_clicks or data_source == 't-1':
            return px.scatter(title="Выберите параметры и нажмите кнопку")
        continuous = True
        if time_type == 'h1':
            continuous = False

        try:
            top_number = int(top_number)
            start_dt = pd.to_datetime(f"{start_date} {start_time_str}")
            end_dt = pd.to_datetime(f"{end_date} {end_time_str}")
        except Exception as e:
            return px.scatter(title=f"Ошибка в формате даты/времени: {e}")

        if data_source == 't0':
            data = load_top_routes_overground(top_number, start_date, end_date, start_time_str, end_time_str,
                                              continuous,
                                              True)
            fig = plot(data, "NAME", "count", color="Вид транспорта", xaxis="Название маршрута", yaxis="Количество поездок")
        elif data_source == 't1':
            data = load_top_routes_overground(top_number, start_date, end_date, start_time_str, end_time_str,
                                              continuous,
                                              False)
            fig = plot(data, "NAME", "rides_per_vehicle", color="Вид транспорта",xaxis="Название маршрута", yaxis="Количество поездок на еденицу транспорта")
        elif data_source == 't2':
            data = load_top_stops_overground(top_number, start_date, end_date, start_time_str, end_time_str, continuous)
            fig = plot(data, "NAME", "count", xaxis="Название остановки", yaxis="Количество посадок")
        elif data_source == 't3':
            data = load_top_stops_underground(top_number, start_date, end_date, start_time_str, end_time_str,
                                              continuous)
            fig = plot(data, "NAME", "count", color="Название линии", xaxis="Название станции", yaxis="Количество посадок")

        return fig

    return app


app = application(big_data)

if __name__ == '__main__':
    print('Starting server on port 8239...')
    print("Checkout --> http://localhost:8239/")
    serve(app.server, host='0.0.0.0', port=8239)
else:
    print(f"__name__ is: {__name__} (has to be __main__)")