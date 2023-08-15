import torch
import torch.nn as nn
from xgboost import XGBRegressor
from lab2.plan import Operator
import numpy as np

operators = ["PAD", "Projection", "Selection", "Sort", "HashAgg", "HashJoin", "TableScan", "IndexScan", "TableReader",
             "IndexReader", "IndexLookUp"]

op2vec = {op: [1 if i == idx else 0 for i in range(1, len(operators))] for idx, op in enumerate(operators)}

max_row_cnt = 0


# There are many ways to extract features from plan:
# 1. The simplest way is to extract features from each node and sum them up. For example, we can get
#      a  the number of nodes;
#      a. the number of occurrences of each operator;
#      b. the sum of estRows for each operator.
#    However we lose the tree structure after extracting features.
# 2. The second way is to extract features from each node and concatenate them in the DFS traversal order.
#                  HashJoin_1
#                  /          \
#              IndexJoin_2   TableScan_6
#              /         \
#          IndexScan_3   IndexScan_4
#    For example, we can concatenate the node features of the above plan as follows:
#    [Feat(HashJoin_1)], [Feat(IndexJoin_2)], [Feat(IndexScan_3)], [Feat(IndexScan_4)], [Padding], [Feat(TableScan_6)], [Padding]
#    Notice1: When we traverse all the children in DFS, we insert [Padding] as the end of the children. In this way, we
#    have an one-on-one mapping between the plan tree and the DFS order sequence.
#    Notice2: Since the different plans have the different number of nodes, we need padding to make the lengths of the
#    features of different plans equal.
class PlanFeatureCollector:
    def __init__(self):
        # YOUR CODE HERE: define variables to collect features from plans
        self.vecs = []
        pass

    def add_operator(self, op: Operator):
        # YOUR CODE HERE: extract features from op
        op_type = op.id.split('_')[0]
        # print(op_type)
        if op_type not in operators:
            return
        op_type_vec = op2vec[op_type]
        op_rows = float(op.est_rows)
        self.vecs.append(op_type_vec + [op_rows, ])

    def walk_operator_tree(self, op: Operator):
        self.add_operator(op)
        for child in op.children:
            self.walk_operator_tree(child)
        # YOUR CODE HERE: process and return the features
        self.vecs.append(op2vec['PAD'] + [0, ])
        return self.vecs


class PlanDataset(torch.utils.data.Dataset):
    def __init__(self, plans, max_operation_num):
        super().__init__()
        self.vecs, self.ts = [], []
        for plan in plans:
            collector = PlanFeatureCollector()
            vec = collector.walk_operator_tree(plan.root)
            # YOUR CODE HERE: maybe you need padding the features if you choose the second way to extract the features.
            while len(vec) < max_operation_num:
                vec.append(op2vec['PAD'] + [0, ])
            features = torch.Tensor(vec)
            exec_time = torch.Tensor([plan.exec_time_in_ms()])
            self.vecs.append(features)
            self.ts.append(exec_time)

    def __getitem__(self, index):
        return self.vecs[index], self.ts[index]

    def __len__(self):
        return len(self.vecs)


# Define your model for cost estimation
class CostModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(11, 16, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(torch.relu(out))
        return out

    def init_weights(self):
        for net in [self.fc, self.lstm]:
            for name, param in net.named_parameters():
                if name.startswith("weight"):
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.zeros_(param)


def count_operator_num(op: Operator):
    num = 2  # one for the node and another for the end of children
    row = float(op.est_rows)
    global max_row_cnt
    max_row_cnt = max(max_row_cnt, row)
    for child in op.children:
        num += count_operator_num(child)
    return num


def estimate_learning(train_plans, test_plans):
    max_operator_num = 0
    for plan in train_plans:
        max_operator_num = max(max_operator_num, count_operator_num(plan.root))
    for plan in test_plans:
        max_operator_num = max(max_operator_num, count_operator_num(plan.root))
    print(f"max_operator_num:{max_operator_num}")

    train_dataset = PlanDataset(train_plans, max_operator_num)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)

    model = XGBRegressor()
    x, y = train_dataset.vecs, train_dataset.ts
    x, y = [i.numpy().reshape([-1]) for i in x], [j.numpy() for j in y]
    x, y = np.array(x), np.array(y)
    model.fit(
        x, y,
        eval_metric='rmse',
        eval_set=[(x, y)],
    )
 

    train_est_times, train_act_times = [], []
    for xi, yi in zip(x, y):
        yh = model.predict(xi.reshape([1, -1]))
        train_est_times.append(yh.item())
        train_act_times.append(yi.item())

    test_dataset = PlanDataset(test_plans, max_operator_num)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=1)

    test_est_times, test_act_times = [], []
    x, y = test_dataset.vecs, test_dataset.ts
    x, y = [i.numpy().reshape([-1]) for i in x], [j.numpy() for j in y]
    x, y = np.array(x), np.array(y)
    for xi, yi in zip(x, y):
        yh = model.predict(xi.reshape([1, -1]))
        test_est_times.append(yh.item())
        test_act_times.append(yi.item())

    model.save_model('./xgboost_cost_estimator.json')
    return train_est_times, train_act_times, test_est_times, test_act_times


class CostEstimator:
    def __init__(self):
        self.xgb = XGBRegressor()
        self.xgb.load_model('../lab2/xgboost_cost_estimator.json')
        self._curr_vec = None
        pass

    def get_cost(self, req):
        self._curr_vec = []
        self.cal_feature_from_req(req)
        while len(self._curr_vec) != 24:
            self._curr_vec.append(op2vec['PAD'] + [0, ])
        vec = np.array(self._curr_vec).reshape([1, -1])
        est = self.xgb.predict(vec)
        return est.item()

    def cal_feature_from_req(self, req):
        assert isinstance(req, dict)
        op = req['id'].split('_')[0]
        if op not in operators:
            return self._curr_vec
        op_rows = float(req['est_rows'])
        self._curr_vec.append(op2vec[op] + [op_rows, ])
        if req['children'] is not None:
            for child in req['children']:
                self.cal_feature_from_req(child)
        self._curr_vec.append(op2vec['PAD'] + [0, ])


