from collections import Counter

from flask import Flask, escape, request
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

import dataset


MOUNT_PATH = '/data/'


data_app = dash.Dash(__name__, requests_pathname_prefix=MOUNT_PATH,
                    meta_tags=[{"name": "viewport", "content": "width=device-width"}])

colors = {
    'background': '#444444',
    'text': '#7FDBFF'
}

style = {'font-size': '18px',
         'text-align': 'center',
         'columnCount': 1
         }

ds_param = dict(from_batch_cache='info', lan=None,
                concate_title=False,
                filter_tags_threshold=0)
ds = dataset.ds_info_tags(**ds_param)
top_tags = 30


def page_description():
    with open('vuejs/data-page.md') as f:
        txt = f.read()
    return txt


def lan_fig(ds) -> dcc.Graph:
    language = ds.data['language'].str.replace(
        'en', 'English').replace('cn', 'Chinese')
    fig_lan = px.pie(language, names='language',
                     title='Language Ratio of Indexed Articles', hole=0.3)
    fig_lan.update_traces(textinfo='percent+label')
    fig_lan.update_layout(showlegend=False, title_x=0.5)
    return dcc.Graph(figure=fig_lan)


def tags_rank_fig(ds, top: int=20) -> dcc.Graph:
    c = Counter([tag for tags in ds.target_decoded for tag in tags])

    dfc = pd.DataFrame.from_dict(c, orient='index', columns=['count']).sort_values(
        by='count', ascending=False)[:top]
    dfc['Rank'] = range(1, dfc.size + 1)
    dfc['Portion'] = dfc['count'] / ds.data.shape[0]

    fig_Y = px.bar(dfc, x=dfc.index, y='Portion',
                   #    text='count',
                   labels={'Rank': 'Rank',
                           'x': 'Tags'},
                   hover_data=['Rank'],
                   title=f'Top {top} Appearance of Tags')
    # fig_Y.update_traces(texttemplate='%{text}')
    fig_Y.update_yaxes(showticklabels=False)
    fig_Y.update_layout(title_x=0.5)
    return dcc.Graph(figure=fig_Y)


# data_app.layout = html.Div(style={'backgroundColor': colors['background']},
# children=[
    # html.Title('Data of LinkedInfo.co'),
data_app.title = 'Data of LinkedInfo.co'
data_app.layout = html.Div(style=style, children=[
    dcc.Markdown(children=page_description()),
    lan_fig(ds),
    html.H1(children=f'Number of Tags: {ds.target.shape[1]}',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }),
    # html.Label(f'Here are the top {top_tags} tags'),
    tags_rank_fig(ds, top_tags),
])

if __name__ == '__main__':
    data_app.run_server(debug=True, port=8000)
