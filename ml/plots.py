from collections import Counter

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

pio.templates.default = 'plotly_white'


def tags_per_article(ds) -> go.Figure:
    num_tags_lst = [len(tags) for tags in ds.target_decoded]
    dfl = pd.DataFrame(num_tags_lst, columns=['num_tags'])
    fig_tl = px.histogram(dfl, x='num_tags', histnorm='percent')
    # fig_tl.update_traces(
    #     xbins=dict(
    #         start=0,
    #         end=7,
    #         # size=1,
    #     )
    # )
    fig_tl.update_layout(title_text='Percentage of Number of Tags for each Article',  # title of plot
                         xaxis_title_text='Number of Tags',  # xaxis label
                         yaxis_title_text='Percent of Articles %',  # yaxis label
                         title_x=0.5,
                         bargap=0.2,
                         xaxis=dict(dtick=1),
                         #  width=1000, height=800
                         #  height=500
                         )

    return fig_tl


def lan_fig(ds) -> go.Figure:
    language = ds.data['language'].str.replace(
        'en', 'English').replace('cn', 'Chinese')
    fig_lan = px.pie(language, names='language',
                     title='Language Ratio of Indexed Articles', hole=0.3)
    fig_lan.update_traces(textinfo='percent+label')
    fig_lan.update_layout(showlegend=False, title_x=0.5)
    return fig_lan


def tags_rank_fig(ds, top: int = 20) -> go.Figure:
    c = Counter([tag for tags in ds.target_decoded for tag in tags])

    dfc = pd.DataFrame.from_dict(c, orient='index', columns=['count']).sort_values(
        by='count', ascending=False)[:top]
    dfc['Rank'] = range(1, dfc.size + 1)
    dfc['Portion'] = dfc['count'] / ds.data.shape[0]

    fig_Y = px.bar(dfc, x=dfc.index, y='Portion',
                   #    text='count',
                   labels={'Rank': 'Rank',
                           'x': 'Tag'},
                   hover_data=['Rank'],
                   title=f'Top {top} Appearance of Tags')
    # fig_Y.update_traces(texttemplate='%{text}')
    fig_Y.update_yaxes(showticklabels=False)
    fig_Y.update_layout(title_x=0.5)
    return fig_Y


def creators_rank_fig(ds, top: int = 20) -> go.Figure:
    c = Counter([tag for tags in ds.creators_per_info for tag in tags])

    dfc = pd.DataFrame.from_dict(c, orient='index', columns=['count']).sort_values(
        by='count', ascending=False)[:top]
    dfc['Rank'] = range(1, dfc.size + 1)
    dfc['Portion'] = dfc['count'] / ds.data.shape[0]

    fig_Y = px.bar(dfc, x=dfc.index, y='Portion',
                   #    text='count',
                   labels={'Rank': 'Rank',
                           'x': 'Author'},
                   hover_data=['Rank'],
                   title=f'Top {top} Appearance of Authors')
    # fig_Y.update_traces(texttemplate='%{text}')
    fig_Y.update_yaxes(showticklabels=False)
    fig_Y.update_layout(title_x=0.5)
    return fig_Y


def domain_rank_fig(ds, top: int = 20) -> go.Figure:
    # c = Counter([tag for tags in ds.creators_per_info for tag in tags])
    c = ds.data['host'].value_counts()

    dfc = pd.DataFrame(c)[:top]
    dfc['Rank'] = range(1, dfc.size + 1)
    dfc['Portion'] = dfc['host'] / ds.data.shape[0]

    fig_Y = px.bar(dfc, x=dfc.index, y='Portion',
                   #    text='count',
                   labels={'Rank': 'Rank',
                           'x': 'Host'},
                   hover_data=['Rank'],
                   title=f'Top {top} Hosts')
    # fig_Y.update_traces(texttemplate='%{text}')
    fig_Y.update_yaxes(showticklabels=False)
    fig_Y.update_layout(title_x=0.5)
    return fig_Y
