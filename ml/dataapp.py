# from flask import Flask, escape, request
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from typing import Optional

from . import dataset
from .models import files
from . import plots


MOUNT_PATH = '/data/'

external_stylesheets = [
    {
        'href': "//unpkg.com/bootstrap/dist/css/bootstrap.min.css",
        'rel': 'stylesheet',
        'crossorigin': 'anonymous'
    },
    {
        'href': "//unpkg.com/bootstrap-vue@latest/dist/bootstrap-vue.min.css",
        'rel': 'stylesheet',
        'crossorigin': 'anonymous'
    }
]

data_app = dash.Dash(__name__, requests_pathname_prefix=MOUNT_PATH,
                     #  external_stylesheets=external_stylesheets,
                     external_stylesheets=[dbc.themes.BOOTSTRAP],
                     meta_tags=[{"name": "viewport", "content": "width=device-width"}])

colors = {
    'background': '#444444',
    'text': '#7FDBFF'
}

style = {'font-size': '18px',
         'text-align': 'center',
         'padding-top': '20px',
         'padding-bottom': '40px',
         'columnCount': 1
         }

data_app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        <div class="d-flex flex-column flex-md-row align-items-center p-3 px-md-4 mb-3 bg-white border-bottom shadow-sm">
            <h5 class="my-0 mr-md-auto font-weight-normal"><a class="p-2 text-dark"
                href="/">LinkedInfo ML</a></h5>
            <nav class="my-2 my-md-0 mr-md-3">
                <a class="p-2 text-dark"
                href="https://www.linkedinfo.co">Linkedinfo</a>
                <a class="p-2 text-dark"
                href="/">Tag Prediction</a>
                <a class="p-2 text-dark"
                href="/data/">Data</a>
                <!-- <a class="p-2 text-dark"
                href="https://www.linkedinfo.co/about/">About</a> -->
            </nav>
        </div>

        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
        <div>My Custom footer</div>
    </body>
</html>
'''

# ds = dataset.load_dataapp_set()
ds: Optional[dataset.DataappSet] = None
pca_titles: Optional[pd.DataFrame] = None
top_tags = 30
top_creators = 30
top_domains = 30


def lazy_load():
    # print('start to load model and data')

    global ds, pca_titles

    if not ds:
        ds = dataset.load_dataapp_set(
            filename=files.model_file(files.DS_DATA_APP))

    if not pca_titles:
        pca_titles = pd.read_pickle(files.ALL_MODELS[files.PCA_DATA_APP])


def page_description():
    # lazy_load()
    with open('vuejs/data-page.md') as f:
        txt = f.read()
    return txt


lazy_load()

if ds is not None:
    app_children = [
        dcc.Markdown(children=page_description(),
                     style={'margin-bottom': '40px'}),
        # html.H2(children=f'Number of Tags: {ds.target.shape[1]}',
        dcc.Graph(id='pca-plot'),
        dbc.Row(children=[
            dbc.Col(
                dcc.Slider(
                    id='pca-plot-dot-size-slider',
                    min=3,
                    max=12,
                    value=6,
                    marks={str(size): size for size in range(3, 13)},
                    step=None
                ), sm=12, md=6, lg=4)], justify='center'),
        html.P('Adjust dot size', className='text-muted'),
        dbc.Row(children=[
            dbc.Col(
                dbc.Card([
                    # dbc.CardHeader("Number of Tags"),
                    dbc.CardBody([
                        html.H3("Number of Tags"),
                        html.H5(len(ds.tags))
                        # style={
                        #     'textAlign': 'center',
                        #     # 'color': colors['text']
                        # }),
                    ])
                ]), sm=12, md=12, lg=4),
        ], justify='center'),

        dbc.Row(children=[
            dbc.Col(children=[
                dcc.Graph(figure=plots.lan_fig(ds)),
            ], md=6, lg=6),
            dbc.Col(children=[
                dcc.Graph(figure=plots.tags_per_article(ds)),
            ], md=6, lg=6),
        ]),
        # html.Label(f'Here are the top {top_tags} tags'),
        dcc.Graph(figure=plots.tags_rank_fig(ds, top_tags)),
        dcc.Graph(figure=plots.creators_rank_fig(ds, top_creators)),
        dcc.Graph(figure=plots.domain_rank_fig(ds, top_domains)),
    ]
else:
    app_children = [
        html.H2(children=f'Load dataset failed, try again later...')]

data_app.title = 'Data of LinkedInfo.co'
data_app.layout = html.Div(
    style=style,
    children=app_children)


@data_app.callback(
    Output('pca-plot', 'figure'),
    [Input('pca-plot-dot-size-slider', 'value')]
)
def update_pca_plot(dot_size):
    figure = plots.pca_titles_3d_fig(
        pca_titles, sub_sample_size=1, dot_size=dot_size)

    return figure


if __name__ == '__main__':
    data_app.run_server(debug=True, host='127.0.0.1', port=5000)
