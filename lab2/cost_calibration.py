import math

import torch.nn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def estimate_plan(operator, factors, weights):
    cost = 0.0
    for child in operator.children:
        cost += estimate_plan(child, factors, weights)
    if len(operator.children) == 0:
        input_row_cnt = 0
        input_row_size = 0
    else:
        input_row_cnt = operator.children[0].est_row_counts()
        input_row_size = operator.children[0].row_size()
    output_row_cnt = operator.est_row_counts()
    output_row_size = operator.row_size()

    if operator.is_hash_agg():
        # YOUR CODE HERE: hash_agg_cost = input_row_cnt * cpu_fac
        weights['cpu'] += input_row_cnt
        pass

    elif operator.is_hash_join():
        # hash_join_cost = (build_hashmap_cost + probe_and_pair_cost)
        #   = (build_row_cnt * cpu_fac) + (output_row_cnt * cpu_fac)
        output_row_cnt = operator.est_row_counts()
        build_side = int(operator.children[1].is_build_side())
        build_row_cnt = operator.children[build_side].est_row_counts()
        # cost += (build_row_cnt + output_row_cnt) * factors['cpu']
        weights['cpu'] += (build_row_cnt + output_row_cnt)

    elif operator.is_sort():
        # YOUR CODE HERE: sort_cost = input_row_cnt * log(input_row_cnt) * cpu_fac
        if input_row_cnt == 0:
            pass
        else:
            weights['cpu'] += input_row_cnt * math.log2(input_row_cnt)

    elif operator.is_selection():
        # YOUR CODE HERE: selection_cost = input_row_cnt * cpu_fac
        weights['cpu'] += input_row_cnt
        pass

    elif operator.is_projection():
        # YOUR CODE HERE: projection_cost = input_row_cnt * cpu_fac
        weights['cpu'] += input_row_cnt

    elif operator.is_table_reader():
        # YOUR CODE HERE: table_reader_cost = input_row_cnt * input_row_size * net_fac
        weights['net'] += input_row_cnt * input_row_size

    elif operator.is_table_scan():
        # YOUR CODE HERE: table_scan_cost = row_cnt * row_size * scan_fac
        weights['scan'] += output_row_cnt * output_row_size

    elif operator.is_index_reader():
        # YOUR CODE HERE: index_reader_cost = input_row_cnt * input_row_size * net_fac
        weights['net'] += input_row_cnt * input_row_size

    elif operator.is_index_scan():
        # YOUR CODE HERE: index_scan_cost = row_cnt * row_size * scan_fac
        weights['scan'] += output_row_cnt * output_row_size

    elif operator.is_index_lookup():
        # index_lookup_cost = net_cost + seek_cost
        #   = (build_row_cnt * build_row_size + probe_row_cnt * probe_row_size) * net_fac +
        #     (build_row_cnt / batch_size) * seek_fac
        build_side = int(operator.children[1].is_build_side())
        build_row_cnt = operator.children[build_side].est_row_counts()
        build_row_size = operator.children[build_side].row_size()
        probe_row_cnt = operator.children[1 - build_side].est_row_counts()
        probe_row_size = operator.children[1 - build_side].row_size()
        batch_size = operator.batch_size()

        # cost += (build_row_cnt * build_row_size + probe_row_cnt * probe_row_size) * factors['net']
        weights['net'] += (build_row_cnt * build_row_size + probe_row_cnt * probe_row_size)

        # cost += (build_row_cnt / batch_size) * factors['seek']
        weights['seek'] += (build_row_cnt / batch_size)

    else:
        print(operator.id)
        assert (1 == 2)  # unknown operator

    for k in factors:
        cost += factors[k] * weights[k]
    return cost


def estimate_calibration(train_plans, test_plans):
    # init factors
    factors = {
        "cpu": 1,
        "scan": 1,
        "net": 1,
        "seek": 1,
    }

    # get training data: factor weights and act_time
    est_costs_before = []
    act_times = []
    weights = []
    for p in train_plans:
        w = {"cpu": 0, "scan": 0, "net": 0, "seek": 0}
        cost = estimate_plan(p.root, factors, w)
        weights.append(w)
        act_times.append(p.exec_time_in_ms())
        est_costs_before.append(cost)

    input_dim = output_dim = 4
    hidden_dim = 32
    lr = 1e-3
    epochs = 20

    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, output_dim)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr)
    # for epoch in range(epochs):
    #     for iter, (weight, act_time) in enumerate(zip(weights, act_times)):
    #         optimizer.zero_grad()
    #         input = torch.FloatTensor(list(weight.values())).unsqueeze(0)
    #         output = model(input)
    #         est_time = (output[0] * input).sum()
    #         loss = abs(est_time - act_time)
    #         loss.backward()
    #         optimizer.step()
    #         print(loss)

    model = LinearRegression()
    x = np.array([list(w.values()) for w in weights])
    # scaler = StandardScaler()
    # # x = scaler.fit_transform(X=x)
    y = np.array(act_times)
    model.fit(x, y)
    for w, a in zip(weights, act_times):
        e = model.predict([list(w.values())])
        print(e, a)
    codf = model.coef_


    # YOUR CODE HERE
    # calibrate your cost model with regression and get the best factors
    # factors * weights ==> act_time
    new_factors = {
        "cpu": codf[0], "scan": codf[1], "net": codf[2], "seek": codf[3]
    }
    print("--->>> regression cost factors: ", new_factors)

    # evaluation
    est_costs = []
    for p in test_plans:
        w = {"cpu": 0, "scan": 0, "net": 0, "seek": 0}
        cost = estimate_plan(p.root, new_factors, w)
        est_costs.append(cost)

    return est_costs
