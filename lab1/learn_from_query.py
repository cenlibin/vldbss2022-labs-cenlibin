import xgboost

import evaluation_utils as eval_utils
import matplotlib.pyplot as plt
import numpy as np
import range_query as rq
import json
import torch
import torch.nn as nn
import lab1.statistics as stats

import xgboost as xgb
from range_query import ParsedRangeQuery


def min_max_normalize(v, min_v, max_v):
    # The function may be useful when dealing with lower/upper bounds of columns.
    assert max_v > min_v
    if v < min_v:
        return 0
    elif v > max_v:
        return 1
    return (v - min_v) / (max_v - min_v)


def extract_features_from_query(parser, table_stats, considered_cols):
    # feat:     [c1_begin, c1_end, c2_begin, c2_end, ... cn_begin, cn_end, AVI_sel, EBO_sel, Min_sel]
    #           <-                   range features                    ->, <-     est features     ->

    feature = []
    for idx, col_name in enumerate(considered_cols):
        if col_name in parser.col_left.keys():
            left, right = parser.col_left[col_name], parser.col_right[col_name]
            col_stats = table_stats.columns[col_name].topn
            _min, _max = col_stats[0].data, col_stats[-1].data
            left, right = min_max_normalize(left, _min, _max), min_max_normalize(right, _min, _max)

        else:
            left, right = 0.0, 1.0

        feature += [left, right]
    avi = stats.AVIEstimator.estimate(parser, table_stats)
    ebo = stats.ExpBackoffEstimator.estimate(parser, table_stats)
    min_sel = stats.MinSelEstimator.estimate(parser, table_stats)
    feature += [avi, ebo, min_sel]
    return feature


def preprocess_queries(queris, table_stats, columns):
    """
    preprocess_queries turn queries into features and labels, which are used for regression model.
    """
    features, labels = [], []
    for item in queris:
        query, act_rows = item['query'], item['act_rows']
        parsed = ParsedRangeQuery.parse_range_query(query)
        feature = extract_features_from_query(parsed, table_stats, considered_cols=columns)
        label = act_rows / table_stats.row_count
        # YOUR CODE HERE: transform (query, act_rows) to (feature, label)
        # Some functions like rq.ParsedRangeQuery.parse_range_query and extract_features_from_query may be helpful.
        label = [label, ]
        features.append(feature)
        labels.append(label)
    return features, labels


class QueryDataset(torch.utils.data.Dataset):
    def __init__(self, queries, table_stats, columns):
        super().__init__()
        self.features, self.labels = preprocess_queries(queries, table_stats, columns)

    def __getitem__(self, index):
        return torch.FloatTensor(self.features[index]), torch.FloatTensor(self.labels[index])

    def __len__(self):
        return len(self.features)


def est_mlp(train_data, test_data, table_stats, columns):
    """
    est_mlp uses MLP to produce estimated rows for train_data and test_data
    """
    # YOUR CODE HERE: train procedure
    train_dataset = QueryDataset(train_data, table_stats, columns)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=1)

    total_rows = table_stats.row_count
    feature_dim = len(train_dataset[0][0])
    hidden_dim = 32
    epochs = 30
    lr = 1e-3

    model = nn.Sequential(
        nn.Linear(feature_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
        # nn.Sigmoid()
    )
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    for epoch in range(epochs):
        for idx, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            est = model(batch_x)
            loss = loss_func(est, batch_y)
            loss.backward()
            optimizer.step()
        # print(loss)

    train_est_rows, train_act_rows = [], []
    for x, y in train_dataset:
        y_hat = model(x.unsqueeze(0))
        train_est_rows.append(y_hat.item() * total_rows)
        train_act_rows.append(y.item() * total_rows)

    test_dataset = QueryDataset(test_data, table_stats, columns)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=1)
    test_est_rows, test_act_rows = [], []

    # YOUR CODE HERE: test procedure
    for x, y in test_dataset:
        y_hat = model(x.unsqueeze(0))
        test_est_rows.append(y_hat.item() * total_rows)
        test_act_rows.append(y.item() * total_rows)

    return train_est_rows, train_act_rows, test_est_rows, test_act_rows


def est_xgb(train_data, test_data, table_stats, columns):
    """
    est_xgb uses xgboost to produce estimated rows for train_data and test_data
    """
    print("estimate row counts by xgboost")
    train_x, train_y = preprocess_queries(train_data, table_stats, columns)
    test_x, test_y = preprocess_queries(test_data, table_stats, columns)

    model = xgboost.XGBRegressor()
    model.fit(
        train_x,
        train_y,
        eval_set=[(test_x, test_y)],
        eval_metric='rmse'
    )

    pred_train = model.predict(train_x)
    row_count = table_stats.row_count
    train_est_rows, train_act_rows = [], []
    for pred, y in zip(pred_train, train_y):
        train_est_rows.append(pred * row_count)
        train_act_rows.append(y[0] * row_count)

    pred_test = model.predict(test_x)
    test_est_rows, test_act_rows = [], []
    for pred, y in zip(pred_test, test_y):
        test_est_rows.append(pred * row_count)
        test_act_rows.append(y[0] * row_count)
    model.save_model('./xgboost_cardinality_estimator.json')
    return train_est_rows, train_act_rows, test_est_rows, test_act_rows


def eval_model(model, train_data, test_data, table_stats, columns):
    if model == 'mlp':
        est_fn = est_mlp
    else:
        est_fn = est_xgb

    train_est_rows, train_act_rows, test_est_rows, test_act_rows = est_fn(train_data, test_data, table_stats, columns)

    name = f'{model}_train_{len(train_data)}'
    eval_utils.draw_act_est_figure(name, train_act_rows, train_est_rows)
    p50, p80, p90, p99 = eval_utils.cal_p_error_distribution(train_act_rows, train_est_rows)
    print(f'{name}, p50:{p50}, p80:{p80}, p90:{p90}, p99:{p99}')

    name = f'{model}_test_{len(test_data)}'
    eval_utils.draw_act_est_figure(name, test_act_rows, test_est_rows)
    p50, p80, p90, p99 = eval_utils.cal_p_error_distribution(test_act_rows, test_est_rows)
    print(f'{name}, p50:{p50}, p80:{p80}, p90:{p90}, p99:{p99}')


class CardinalityEstimator:
    def __init__(self):
        self.xgb = xgboost.XGBRegressor()
        self.xgb.load_model('../lab1/xgboost_cardinality_estimator.json')
        self.columns = ['kind_id', 'production_year', 'imdb_id', 'episode_of_id', 'season_nr', 'episode_nr']
        stats_json_file = '../lab1/data/title_stats.json'
        self.table_stats = stats.TableStats.load_from_json_file(stats_json_file, self.columns)
        # table_stats = stats.TableStats.load_from_json_file(stats_json_file, columns)
        pass
 
    def get_cardinality(self, req):
        conds = req.split('and')
        col_left, col_right = {}, {}
        for cond in conds:
            cond = cond.strip()
            for op in ["<", ">"]:
                idx = cond.find(op)
                if idx > 0:
                    col = cond[:idx]
                    val = float(cond[idx + len(op):])
                    if op == ">":
                        col_left[col] = int(val)
                    elif op == "<":
                        col_right[col] = int(val)
        feature = []
        for idx, col_name in enumerate(self.columns):
            if col_name in col_left.keys():
                left, right = col_left[col_name], col_right[col_name]
                col_stats = self.table_stats.columns[col_name].topn
                _min, _max = col_stats[0].data, col_stats[-1].data
                left, right = min_max_normalize(left, _min, _max), min_max_normalize(right, _min, _max)
 
            else:
                left, right = 0.0, 1.0
 
            feature += [left, right]
 
        avi = stats.AVIEstimator.estimate_from_tuple((col_left, col_right), self.table_stats)
        ebo = stats.ExpBackoffEstimator.estimate_from_tuple((col_left, col_right), self.table_stats)
        min_sel = stats.MinSelEstimator.estimate_from_tuple((col_left, col_right), self.table_stats)
        feature += [avi, ebo, min_sel]
        feature = np.array(feature).reshape([1, -1])
        sel = self.xgb.predict(feature).item()
        est = sel * self.table_stats.row_count
        return est

if __name__ == '__main__':
    stats_json_file = './data/title_stats.json'
    train_json_file = './data/query_train_20000.json'
    test_json_file = './data/query_test_5000.json'
    columns = ['kind_id', 'production_year', 'imdb_id', 'episode_of_id', 'season_nr', 'episode_nr']
    table_stats = stats.TableStats.load_from_json_file(stats_json_file, columns)
    with open(train_json_file, 'r') as f:
        train_data = json.load(f)
    with open(test_json_file, 'r') as f:
        test_data = json.load(f)

    eval_model('mlp', train_data, test_data, table_stats, columns)
    eval_model('xgb', train_data, test_data, table_stats, columns)
