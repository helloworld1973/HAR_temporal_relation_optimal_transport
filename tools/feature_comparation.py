import numpy as np
import torch
import plotly.express as px
from loss.loss_transfer import TransferLoss


# generate graph edge, cal cosine of two features in the same time series, compare different periods
def compare_a_user_features_a_class(data_bags, loss_type, data_name):
    num_of_windows = len(data_bags)
    a_bag_edge_matrix = np.full(shape=(num_of_windows, num_of_windows), fill_value=-1)
    for i in range(num_of_windows):
        for j in range(i, num_of_windows):
            left_instance_features = data_bags[i]
            right_instance_features = data_bags[j]

            criterion_transder = TransferLoss(loss_type=loss_type, input_dim=num_of_windows)
            dist = criterion_transder.compute(torch.Tensor(left_instance_features),
                                              torch.Tensor(right_instance_features))
            a_bag_edge_matrix[i][j] = dist.item()

    fig = px.imshow(a_bag_edge_matrix, labels=dict(x=data_name, y="time sequence"))
    fig.show()

    return a_bag_edge_matrix


def compare_two_users_features_a_class(s_data_bags, t_data_bags, data_name):
    s_num_of_windows = len(s_data_bags)
    t_num_of_windows = len(t_data_bags)
    a_bag_edge_matrix = np.full(shape=(s_num_of_windows, t_num_of_windows), fill_value=-1)

    for i in range(s_num_of_windows):
        for j in range(t_num_of_windows):
            left_instance_features = s_data_bags[i]
            right_instance_features = t_data_bags[j]
            a_bag_edge_matrix[i][j] = np.sqrt(np.sum(np.square(left_instance_features - right_instance_features)))

    fig = px.imshow(a_bag_edge_matrix, labels=dict(x=data_name, y="time sequence"))
    fig.show()

    return a_bag_edge_matrix
