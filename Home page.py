import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
PLOTLY_LOGO = "https://i.ibb.co/S67gj58/Screenshot-2020-01-22-at-22-58-29.png"
PLOTLY_background = "https://www.fusioncharts.com/blog/wp-content/uploads/2018/05/Best-Python-Data-Visualization-Libraries-fusioncharts.png"

app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
)
sidebar_header = dbc.Row(
    [
        dbc.Col(html.H2("", className="display-4")),
        dbc.Col(
            html.Button(
                html.Span(className="navbar-toggler-icon"),
                className="navbar-toggler",
                style={
                    "color": "rgba(0,0,0,.5)",
                    "border-color": "rgba(0,0,0,.1)",
                },
                id="toggle",
            ),
            width="auto",
            align="center",
        ),
    ]
)

sidebar = html.Div(
    [
        sidebar_header,
        html.Div(
            [
                html.Img(src=PLOTLY_LOGO,style={'height':'70%', 'width':'70%','text-align': 'center'}),   
                html.Hr(),
                
                html.P(
                    " "
                    ,
                    className="lead",
                ),
            ],
            id="blurb",
        ),
        dbc.Collapse(
            dbc.Nav(
                [
                    dbc.NavLink("Home", href="/page-1", id="page-1-link"),
                    dbc.NavLink("Graph 1", href="/page-2", id="page-2-link"),
                    dbc.NavLink("Graph 2", href="/page-3", id="page-3-link"),
                ],
                vertical=True,
                pills=True,
            ),
            id="collapse",
        ),
    ],
    id="sidebar",
)


content = html.Div(id="page-content")

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])
@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1, 4)],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/":
        return True, False, False
    return [pathname == f"/page-{i}" for i in range(1, 4)]


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
     
    if pathname in ["/", "/page-1"]:
         return html.Img(src=PLOTLY_background,style={'width':'110%','height':'105%','verticalAlign':'middle','align': 'center','margin-left':'5%','margin-right':'10%'}),          
    elif pathname == "/page-2":
        return html.Div([
                
                dcc.Graph(
        id='example',
        figure={
            'data': [
                {'x': [1, 2, 3, 4, 5], 'y': [2, 6, 2, 1, 5], 'type': 'line', 'name': 'Boats'},
                {'x': [1, 2, 3, 4, 5], 'y': [8, 7, 2, 7, 3], 'type': 'bar', 'name': 'Cars'},
            ],
            'layout': {
                'title': 'Basic Dash Example'
            }
        }
    )
                
                
 ])
            
    elif pathname == "/page-3":
        return html.P("page 3!")
    return dbc.Jumbotron(
             [
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


@app.callback(
    Output("collapse", "is_open"),
    [Input("toggle", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


if __name__ == "__main__":
    app.run_server(port=8050, debug=False)