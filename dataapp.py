from collections import Counter

from flask import Flask, escape, request
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

import dataset
import plots


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
                     external_stylesheets=external_stylesheets,
                     meta_tags=[{"name": "viewport", "content": "width=device-width"}])

colors = {
    'background': '#444444',
    'text': '#7FDBFF'
}

style = {'font-size': '18px',
         'text-align': 'center',
         'padding-top': '40px',
         'padding-bottom': '40px',
         'columnCount': 1
         }

# ds_param = dict(from_batch_cache='info', lan=None,
#                 concate_title=False,
#                 filter_tags_threshold=0)
# ds = dataset.ds_info_tags(**ds_param)
ds = dataset.load_dataapp_set()
top_tags = 30
top_creators = 30


def page_description():
    with open('vuejs/data-page.md') as f:
        txt = f.read()
    return txt


# data_app.layout = html.Div(style={'backgroundColor': colors['background']},
# children=[
    # html.Title('Data of LinkedInfo.co'),
data_app.title = 'Data of LinkedInfo.co'
data_app.layout = html.Div(style=style, children=[
    dcc.Markdown(children=page_description()),
    dcc.Graph(figure=plots.lan_fig(ds)),
    # html.H2(children=f'Number of Tags: {ds.target.shape[1]}',
    html.H2(children=f'Number of Tags: {len(ds.tags)}',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }),
    # html.Label(f'Here are the top {top_tags} tags'),
    dcc.Graph(figure=plots.tags_rank_fig(ds, top_tags)),
    dcc.Graph(figure=plots.tags_per_article(ds)),
    dcc.Graph(figure=plots.creators_rank_fig(ds, top_creators)),
    dcc.Graph(figure=plots.domain_rank_fig(ds, top_creators)),
])

if __name__ == '__main__':
    data_app.run_server(debug=True, host='127.0.0.1', port=5000)
