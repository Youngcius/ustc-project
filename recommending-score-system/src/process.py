import pandas as pd
import re


def read_trainset(fname: str, col_names=None) -> pd.DataFrame:
    with open(fname, mode='r', encoding='utf-8') as f:
        txt = re.split('\n|,\n', f.read().strip())
    #################
    # txt = txt[:1000]
    #################
    txt = [record.split(',') for record in txt]
    train_df = {name: None for name in col_names}
    num_row = len(txt)
    num_col = len(col_names)
    for i in range(num_col):
        if i < num_col - 1:
            train_df[col_names[i]] = [txt[j][i] for j in range(num_row)]
        else:
            train_df[col_names[i]] = [txt[j][i:] for j in range(num_row)]
    return pd.DataFrame(train_df, columns=col_names)


def read_testset(fname: str, col_names=None) -> pd.DataFrame:
    with open(fname, mode='r', encoding='utf-8') as f:
        txt = re.split('\n|,\n', f.read().strip())
    #################
    # txt = txt[:1000]
    #################
    txt = [record.split(',') for record in txt]
    test_df = {name: None for name in col_names}
    num_row = len(txt)
    num_col = len(col_names)
    for i in range(num_col):
        if i < num_col - 1:
            test_df[col_names[i]] = [txt[j][i] for j in range(num_row)]
        else:
            test_df[col_names[i]] = [txt[j][i:] for j in range(num_row)]
    return pd.DataFrame(test_df, columns=col_names)


def read_relation(fname: str) -> dict:
    with open(fname, mode='r', encoding='utf-8') as f:
        txt = f.read().strip().split('\n')
    txt = [re.split(':|,', record) for record in txt]
    relation = {int(record[0]): list(map(int, record[1:])) for record in txt}
    return relation
