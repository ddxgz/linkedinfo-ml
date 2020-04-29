from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from typing import List, Callable, Union, Tuple


STACKFILES = [
    'data/stackexchange/stackoverflow-top50k.csv',
    'data/stackexchange/stackoverflow-score81-47-50k.csv',
    'data/stackexchange/stackoverflow-score46-33-50k.csv',
    'data/stackexchange/stackoverflow-score32-25-50k.csv',
]


@dataclass
class DatasetStack:
    data: pd.DataFrame
    target: pd.DataFrame
    target_names: pd.DataFrame
    target_decoded: pd.DataFrame
    mlb: MultiLabelBinarizer


def ds_stack(stackfiles: List[str] = STACKFILES, text_cols: List[str] = [], name_text_col: str = 'description',
             n_tag: int = 500, concat_title: bool = False, body_extractor: Callable = None):
    from bs4 import BeautifulSoup

    df = pd.concat([pd.read_csv(f) for f in stackfiles], ignore_index=True)

    tags_all = df['Tags'].str.extractall(r'\<([^\>]+)\>')

    tags = tags_all.unstack()
    dfc = tags_all[0].value_counts()[:n_tag]

    tags_valid = tags_all[tags_all[0].isin(dfc.index)]
    tags_valid.xs(0, level='match')
    idx_keep = tags_valid.xs(0, level='match').index
    df_keep = df.iloc[idx_keep]

    tags_keep = tags_valid.iloc[idx_keep]
    tags_withnan = tags_keep.values.tolist()

    def not_nan(lst):
        return [tag for tag in lst if str(tag) != 'nan']

    tags_lst = map(not_nan, tags_withnan)

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(tags_lst)

    def extract_stack_body_simple(rec):
        soup = BeautifulSoup(rec, 'html.parser')

        return ''.join(soup.strings).replace('\n', ' ')

    if body_extractor is None:
        body_extractor = extract_stack_body_simple

    df_keep['body'] = df_keep['body'].apply(body_extractor)

    if concat_title:
        df_keep[name_text_col] = df_keep['Title'].str.cat(
            df_keep['body'], sep=' ')

    return DatasetStack(df_keep, Y, mlb.classes_, tags_lst, mlb)
