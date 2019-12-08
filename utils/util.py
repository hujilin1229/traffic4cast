import numpy as np
import h5py
from pathlib import Path
import os, re, datetime, sys
import scipy.sparse as sp
from datetime import datetime, date
import torch
import torch.nn.functional as F

def load_data(file_path, length=6):
    """Load data for one day, return as numpy array with normalized samples of each
        6 time steps in random order.

        Args.:
            file_path (str): file path of h5 file for one day

        Returns: numpy array of shape (48, 6, 3, 495, 436)
    """
    # load h5 file
    fr = h5py.File(file_path, 'r')
    data = fr.get('array').value
    fr.close()
    # split in 48 samples of each length 6 time bins
    # consisting of 48 evenly split non-overlapping time intervals
    # data = np.stack(data, axis=0)

    num_splits = data.shape[0] // length
    data = data[:num_splits*length, ...]
    data = np.array(np.split(data, num_splits, axis=0), dtype=np.uint8)
    # transpose to (samples, timesteps, channels, rows, columns)
    data = np.transpose(data, (0, 1, 4, 2, 3))

    # randomly shuffle, rescale and return data
    # np.random.shuffle(data)
    # data /= 255.

    return data


def list_filenames(directory, excluded_dates=[]):
    """Auxilliary function which returns list of file names in directory in random order,
        filtered by excluded dates.

        Args.:
            directory (str): path to directory
            excluded_dates (list): list of dates which should not be included in result list,
                e.g., ['2018-01-01', '2018-12-31']

        Returns: list
    """
    filenames = os.listdir(directory)
    # np.random.shuffle(filenames)

    if len(excluded_dates) > 0:
        # check if in excluded dates
        excluded_dates = [datetime.datetime.strptime(x, '%Y-%m-%d').date() for x in excluded_dates]
        filenames = [x for x in filenames if return_date(x) not in excluded_dates]

    return filenames


def return_date(file_name):
    """Auxilliary function which returns datetime object from Traffic4Cast filename.

        Args.:
            file_name (str): file name, e.g., '20180516_100m_bins.h5'

        Returns: date string, e.g., '2018-05-16'
    """

    match = re.search(r'\d{4}\d{2}\d{2}', file_name)
    date = datetime.datetime.strptime(match.group(), '%Y%m%d').date()
    return date

def create_directory_structure(root):
    berlin = os.path.join(root, "Berlin","Berlin_test")
    istanbul = os.path.join(root, "Istanbul","Istanbul_test")
    moscow = os.path.join(root, "Moscow", "Moscow_test")
    try:
        os.makedirs(berlin)
        os.makedirs(istanbul)
        os.makedirs(moscow)
    except OSError:
        print("failed to create directory structure")
        # sys.exit(2)

def load_input_data(file_path, indicies, seq_len=3):
    """
    Given a file path, load the relevant training data pieces into a tensor that is returned.
    Return: tensor of shape (number_of_test_cases_per_file =5, 3, 495, 436, 3)
    """
    # load h5 file into memory.
    fr = h5py.File(file_path, 'r')
    data = fr['array'].value
    fr.close()

    # get relevant training data pieces
    data = [data[y-seq_len:y] for y in indicies]
    data = np.stack(data, axis=0)

    # type casting
    data = data.astype(np.float32)
    return data

def write_data(data, filename):
    f = h5py.File(filename, 'w', libver='latest')
    dset = f.create_dataset('array', shape=(data.shape), data = data, compression='gzip', compression_opts=9)
    f.close()

def cast_moving_avg(data):
    """
    Returns cast moving average (cast to np.uint8)
    data = tensor of shape (5, 3, 495, 436, 3) of  type float32
    Return: tensor of shape (5, 3, 495, 436, 3) of type uint8
    """

    prediction = []
    for i in range(3):
        data_slice = data[:, i:]
        t = np.mean(data_slice, axis = 1)
        t = np.expand_dims(t, axis = 1)
        prediction.append(t)
        data = np.concatenate([data, t], axis =1)

    prediction = np.concatenate(prediction, axis = 1)
    prediction = np.around(prediction)
    prediction = prediction.astype(np.uint8)

    return prediction

"""
The Following code is copied from mdeff/cnn_graph lib/graph.py
"""

import sklearn.metrics
import sklearn.neighbors
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial.distance

def grid(mx, my, dtype=np.int):
    """
    Construct a mesh grid

    :param mx: num of cols
    :param my: num of rows
    :param dtype: dtype of the return z
    :return: index set for the mesh grid
    """

    M = mx * my
    x = np.arange(mx)
    y = np.arange(my)
    xx, yy = np.meshgrid(x, y)
    z = np.empty((M, 2), dtype)
    z[:, 0] = yy.reshape(M)
    z[:, 1] = xx.reshape(M)

    return z

def construct_road_network(num_cols, node_pos):

    # non_zeros_x, non_zeros_y = np.nonzero(select_position)
    select_idx = node_pos[:, 0] * num_cols + node_pos[:, 1]
    idx_dict = dict(enumerate(select_idx))
    idx_revert_dict = {v: k for k, v in idx_dict.items()}

    return idx_dict, idx_revert_dict

def adjacency(edge_pairs, select_position, idx_dict, sigma2=1.0, directed=False):
    """
    Return the adjacency matrix of a kNN graph.
    """

    num_rows, num_cols = select_position.shape
    N, k = edge_pairs.shape
    row_list = []
    col_list = []
    dist_list = []
    for i in range(N):
        node_i = idx_dict[i]
        i_row, i_col = node_i // num_cols, node_i % num_cols
        for j in edge_pairs[i]:
            if j == -1:
                continue
            row_list.append(i)
            col_list.append(j)
            node_j = idx_dict[j]
            j_row, j_col = node_j // num_cols, node_j % num_cols
            dist_i_j = 1.0 - np.abs(select_position[i_row, i_col] - select_position[j_row, j_col]) / \
                       max(select_position[i_row, i_col], select_position[j_row, j_col])

            # dist_i_j = 1.0
            dist_list.append(dist_i_j)

    W = scipy.sparse.coo_matrix((dist_list, (row_list, col_list)), shape=(N, N))

    # No self-connections.
    W.setdiag(0)

    if not directed:
        # Non-directed graph.
        bigger = W.T > W
        W = W - W.multiply(bigger) + W.T.multiply(bigger)
        assert W.nnz % 2 == 0
        assert np.abs(W - W.T).mean() < 1e-10

    # assert type(W) is scipy.sparse.csr.csr_matrix
    return W


def adjacency_connected(edge_pairs):
    """
    Return the adjacency matrix of a kNN graph.
    """

    N, k = edge_pairs.shape
    row_list = []
    col_list = []
    dist_list = []
    for i in range(N):
        for j in edge_pairs[i]:
            if j == -1:
                continue
            row_list.append(i)
            col_list.append(j)
            dist_i_j = 1.0
            dist_list.append(dist_i_j)

    W = scipy.sparse.coo_matrix((dist_list, (row_list, col_list)), shape=(N, N))

    # No self-connections.
    W.setdiag(0)

    # Non-directed graph.
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)
    assert W.nnz % 2 == 0
    assert np.abs(W - W.T).mean() < 1e-10

    # assert type(W) is scipy.sparse.csr.csr_matrix
    return W


def construct_road_network_from_grid(n_row, n_col, file_dir, neigh_k, least_ratio, sigma2):

    print("0. Get occurrence matrix...")
    count_np_2channel_file = os.path.join(file_dir, '..', 'count_np.npy')
    files = list_filenames(file_dir, excluded_dates=[])
    num_days = len(files)
    if os.path.exists(count_np_2channel_file):
        count_np = np.load(count_np_2channel_file)
    else:
        count_np_3 = np.zeros((288, n_row, n_col, 3), dtype=np.int)
        for file in files:
            try:
                fr = h5py.File(os.path.join(file_dir, file), 'r')
                data = fr['array'].value
                fr.close()
            except:
                continue
            non_zero_pos = data > 0
            non_zeros = np.zeros(data.shape, dtype=np.int)
            non_zeros[non_zero_pos] = 1
            count_np_3 += non_zeros
        count_np = np.sum(count_np_3[..., :2], axis=(0, -1))
        np.save(count_np_2channel_file, count_np)

    # 2. select the cells that meet the threshold requirement
    print("1. Selecting cells with threshold...")
    # set a number threshold for holding data
    num_threshold = num_days * 288 * 2 * least_ratio
    selected_emerge = np.zeros_like(count_np, np.int)
    selected_emerge[count_np > num_threshold] = count_np[count_np > num_threshold]

    # 1. construct a grid graph
    print("2. Constructing grid graph...")
    # z = grid(n_col, n_row)
    non_zeros_x, non_zeros_y = np.nonzero(selected_emerge)
    node_pos = np.stack([non_zeros_x, non_zeros_y], axis=1)
    _, edge_pairs = distance_sklearn_metrics(node_pos, k=neigh_k)

    # 3. get a correspondence of "pixel id"(row wise) with "graph node id"
    print("3. Constructing network with selected cells...")
    idx_dict, idx_revert_dict = construct_road_network(n_col, node_pos)

    # 4. construct a adjacency matrix of constructing road network
    print("4. Constructing adjacency matrix with selected cells...")
    adj = adjacency(edge_pairs, selected_emerge, idx_dict, sigma2=sigma2)

    return adj, node_pos


def construct_road_network_from_grid_speed(n_row, n_col, file_dir, neigh_k, least_ratio, sigma2):

    print("0. Get occurrence matrix...")
    count_np_2channel_file = os.path.join(file_dir, '..', 'speed_count.npy')
    speed_2channel_file = os.path.join(file_dir, '..', 'speed_avg.npy')
    files = list_filenames(file_dir, excluded_dates=[])
    num_days = len(files)
    if os.path.exists(count_np_2channel_file):
        count_np = np.load(count_np_2channel_file)
        avg_speed = np.load(speed_2channel_file)
    else:
        count_np_2 = np.zeros((288, n_row, n_col), dtype=np.int)
        speed_sum = np.zeros_like(count_np_2, dtype=np.float32)
        for file in files:
            try:
                fr = h5py.File(os.path.join(file_dir, file), 'r')
                data = fr['array'].value
                fr.close()
            except:
                continue
            non_zero_pos = data[..., 1] > 0
            non_zeros = np.zeros(count_np_2.shape, dtype=np.int)
            non_zeros[non_zero_pos] = 1
            count_np_2 += non_zeros
            speed_sum += data[..., 1]
        count_np = np.sum(count_np_2, axis=0)
        speed_sum_sum = np.sum(speed_sum, axis=0)
        avg_speed = speed_sum_sum / count_np
        avg_speed = np.where(np.isnan(avg_speed), np.zeros_like(avg_speed), avg_speed)
        np.save(count_np_2channel_file, count_np)
        np.save(speed_2channel_file, avg_speed)

    # 2. select the cells that meet the threshold requirement
    print("1. Selecting cells with threshold...")
    # set a number threshold for holding data
    num_threshold = num_days * 288 * least_ratio
    selected_emerge = np.zeros_like(count_np, np.int)
    selected_emerge[count_np > num_threshold] = count_np[count_np > num_threshold]
    avg_speed_new = np.zeros_like(avg_speed, dtype=np.float32)
    # print("avg speed new shape is ", avg_speed_new.shape)
    # print("count np shape is ", count_np.shape)
    # print("avg_speed shape is ", avg_speed.shape)
    avg_speed_new[count_np > num_threshold] = avg_speed[count_np > num_threshold]

    # 1. construct a grid graph
    print("2. Constructing grid graph...")
    # z = grid(n_col, n_row)
    non_zeros_x, non_zeros_y = np.nonzero(selected_emerge)
    node_pos = np.stack([non_zeros_x, non_zeros_y], axis=1)
    _, edge_pairs = distance_sklearn_metrics(node_pos, k=neigh_k)

    # 3. get a correspondence of "pixel id"(row wise) with "graph node id"
    print("3. Constructing network with selected cells...")
    idx_dict, idx_revert_dict = construct_road_network(n_col, node_pos)

    # 4. construct a adjacency matrix of constructing road network
    print("4. Constructing adjacency matrix with selected cells...")
    adj = adjacency(edge_pairs, avg_speed_new, idx_dict, sigma2=sigma2)

    return adj, node_pos


def construct_hierarchical_grid_graphs(height, width, node_pos, neigh_k, levels, pooling_size=2):

    zero_maps = np.zeros((height, width))
    zero_maps[node_pos[:, 0], node_pos[:, 1]] = 1
    grid_0 = torch.from_numpy(zero_maps)
    grid_0 = torch.unsqueeze(grid_0, 0)
    grid_0 = torch.unsqueeze(grid_0, 0)

    node_pos_list = []
    adj_list = []
    for i in range(levels):
        non_zeros_x, non_zeros_y = np.nonzero(grid_0.squeeze().numpy())
        node_pos = np.stack([non_zeros_x, non_zeros_y], axis=1).astype(np.int)
        node_pos_list.append(node_pos)
        # construct the adjacency matrix
        _, edge_pairs = distance_sklearn_metrics(node_pos, k=neigh_k)
        adj = adjacency_connected(edge_pairs)
        adj_list.append(adj)
        grid_0 = F.max_pool2d(grid_0, pooling_size)

    return adj_list, node_pos_list


def construct_hierarchical_grid_graphs_weighted(height, width, node_pos, neigh_k,
                                                file_dir, sigma2, levels, pooling_size=2):

    print("0. Get occurrence matrix...")
    count_np_2channel_file = os.path.join(file_dir, '..', 'count_np.npy')
    files = list_filenames(file_dir, excluded_dates=[])
    num_days = len(files)
    if os.path.exists(count_np_2channel_file):
        count_np = np.load(count_np_2channel_file)
    else:
        count_np_3 = np.zeros((288, height, width, 3), dtype=np.int)
        for file in files:
            try:
                fr = h5py.File(os.path.join(file_dir, file), 'r')
                data = fr['array'].value
                fr.close()
            except:
                continue
            non_zero_pos = data > 0
            non_zeros = np.zeros(data.shape, dtype=np.int)
            non_zeros[non_zero_pos] = 1
            count_np_3 += non_zeros
        count_np = np.sum(count_np_3[..., :2], axis=(0, -1))
        np.save(count_np_2channel_file, count_np)

    # 2. select the cells that meet the threshold requirement
    print("1. Selecting cells with threshold...")
    # set a number threshold for holding data
    # num_threshold = num_days * 288 * 2 * least_ratio
    selected_emerge = np.zeros_like(count_np, np.float)
    selected_emerge[node_pos[:, 0], node_pos[:, 1]] = count_np[node_pos[:, 0], node_pos[:, 1]]

    # zero_maps = np.zeros((height, width))
    # zero_maps[node_pos[:, 0], node_pos[:, 1]] = 1
    grid_0 = torch.from_numpy(selected_emerge)
    grid_0 = torch.unsqueeze(grid_0, 0)
    grid_0 = torch.unsqueeze(grid_0, 0)

    node_pos_list = []
    adj_list = []
    for i in range(levels):
        selected_emerge = grid_0.squeeze().numpy()
        width = selected_emerge.shape[1]
        non_zeros_x, non_zeros_y = np.nonzero(selected_emerge)
        node_pos = np.stack([non_zeros_x, non_zeros_y], axis=1).astype(np.int)
        node_pos_list.append(node_pos)
        # construct the adjacency matrix
        _, edge_pairs = distance_sklearn_metrics(node_pos, k=neigh_k)
        # 3. get a correspondence of "pixel id"(row wise) with "graph node id"
        print("3. Constructing network with selected cells...")
        idx_dict, idx_revert_dict = construct_road_network(width, node_pos)

        # 4. construct a adjacency matrix of constructing road network
        print("4. Constructing adjacency matrix with selected cells...")
        adj = adjacency(edge_pairs, selected_emerge, idx_dict, sigma2=sigma2)

        # adj = adjacency_connected(edge_pairs)
        adj_list.append(adj)
        grid_0 = F.max_pool2d(grid_0, pooling_size)

    return adj_list, node_pos_list

# TODO: Need to complete the code
def construct_road_network_from_grid_speed_shrink(
        n_row, n_col, file_dir, neigh_k, least_ratio, sigma2, row_patches=5, col_patches=4):

    print("0. Get occurrence matrix...")
    count_np_2channel_file = os.path.join(file_dir, '..', 'speed_count.npy')
    speed_2channel_file = os.path.join(file_dir, '..', 'speed_avg.npy')
    files = list_filenames(file_dir, excluded_dates=[])
    num_days = len(files)
    if os.path.exists(count_np_2channel_file):
        count_np = np.load(count_np_2channel_file)
        avg_speed = np.load(speed_2channel_file)
    else:
        count_np_2 = np.zeros((288, n_row, n_col), dtype=np.int)
        speed_sum = np.zeros_like(count_np_2, dtype=np.float32)
        for file in files:
            try:
                fr = h5py.File(os.path.join(file_dir, file), 'r')
                data = fr['array'].value
                fr.close()
            except:
                continue
            non_zero_pos = data[..., 1] > 0
            non_zeros = np.zeros(count_np_2.shape, dtype=np.int)
            non_zeros[non_zero_pos] = 1
            count_np_2 += non_zeros
            speed_sum += data[..., 1]
        count_np = np.sum(count_np_2, axis=0)
        speed_sum_sum = np.sum(speed_sum, axis=0)
        avg_speed = speed_sum_sum / count_np
        avg_speed = np.where(np.isnan(avg_speed), np.zeros_like(avg_speed), avg_speed)
        np.save(count_np_2channel_file, count_np)
        np.save(speed_2channel_file, avg_speed)

    # 2. select the cells that meet the threshold requirement
    print("1. Selecting cells with threshold...")
    # set a number threshold for holding data
    num_threshold = num_days * 288 * least_ratio
    count_patches = count_np.reshape((n_row//row_patches, row_patches, n_col//col_patches, col_patches))
    count_patches = count_patches.transpose([0, 2, 1, 3])
    count_patches = count_patches.reshape(n_row//row_patches, n_col//col_patches, -1)

    # TODO: change from here

    selected_emerge = np.zeros_like(count_np, np.int)
    selected_emerge[count_np > num_threshold] = count_np[count_np > num_threshold]
    avg_speed_new = np.zeros_like(avg_speed, dtype=np.float32)
    # print("avg speed new shape is ", avg_speed_new.shape)
    # print("count np shape is ", count_np.shape)
    # print("avg_speed shape is ", avg_speed.shape)
    avg_speed_new[count_np > num_threshold] = avg_speed[count_np > num_threshold]

    # 1. construct a grid graph
    print("2. Constructing grid graph...")
    # z = grid(n_col, n_row)
    non_zeros_x, non_zeros_y = np.nonzero(selected_emerge)
    node_pos = np.stack([non_zeros_x, non_zeros_y], axis=1)
    _, edge_pairs = distance_sklearn_metrics(node_pos, k=neigh_k)

    # 3. get a correspondence of "pixel id"(row wise) with "graph node id"
    print("3. Constructing network with selected cells...")
    idx_dict, idx_revert_dict = construct_road_network(n_col, node_pos)

    # 4. construct a adjacency matrix of constructing road network
    print("4. Constructing adjacency matrix with selected cells...")
    adj = adjacency(edge_pairs, avg_speed_new, idx_dict, sigma2=sigma2)

    return adj, node_pos

def construct_road_network_from_grid_direction(n_row, n_col, file_dir, neigh_k, least_ratio, sigma2):

    print("0. Get occurrence matrix...")
    count_np_2channel_file = os.path.join(file_dir, '..', 'direction_count.npy')
    speed_2channel_file = os.path.join(file_dir, '..', 'direction_avg.npy')
    files = list_filenames(file_dir, excluded_dates=[])
    num_days = len(files)
    if os.path.exists(count_np_2channel_file):
        count_np = np.load(count_np_2channel_file)
        avg_speed = np.load(speed_2channel_file)
    else:
        count_np_2 = np.zeros((288, n_row, n_col), dtype=np.int)
        speed_sum = np.zeros_like(count_np_2, dtype=np.float32)
        for file in files:
            try:
                fr = h5py.File(os.path.join(file_dir, file), 'r')
                data = fr['array'].value
                fr.close()
            except:
                continue
            non_zero_pos = data[..., 2] > 0
            non_zeros = np.zeros(count_np_2.shape, dtype=np.int)
            non_zeros[non_zero_pos] = 1
            count_np_2 += non_zeros
            speed_sum += data[..., 2]
        count_np = np.sum(count_np_2, axis=0)
        speed_sum_sum = np.sum(speed_sum, axis=0)
        avg_speed = speed_sum_sum / count_np
        avg_speed = np.where(np.isnan(avg_speed), np.zeros_like(avg_speed), avg_speed)
        np.save(count_np_2channel_file, count_np)
        np.save(speed_2channel_file, avg_speed)

    # 2. select the cells that meet the threshold requirement
    print("1. Selecting cells with threshold...")
    # set a number threshold for holding data
    num_threshold = num_days * 288 * least_ratio
    selected_emerge = np.zeros_like(count_np, np.int)
    selected_emerge[count_np > num_threshold] = count_np[count_np > num_threshold]
    avg_speed_new = np.zeros_like(avg_speed, dtype=np.float32)
    # print("avg speed new shape is ", avg_speed_new.shape)
    # print("count np shape is ", count_np.shape)
    # print("avg_speed shape is ", avg_speed.shape)
    avg_speed_new[count_np > num_threshold] = avg_speed[count_np > num_threshold]

    # 1. construct a grid graph
    print("2. Constructing grid graph...")
    # z = grid(n_col, n_row)
    non_zeros_x, non_zeros_y = np.nonzero(selected_emerge)
    node_pos = np.stack([non_zeros_x, non_zeros_y], axis=1)
    _, edge_pairs = distance_sklearn_metrics(node_pos, k=neigh_k)

    # 3. get a correspondence of "pixel id"(row wise) with "graph node id"
    print("3. Constructing network with selected cells...")
    idx_dict, idx_revert_dict = construct_road_network(n_col, node_pos)

    # 4. construct a adjacency matrix of constructing road network
    print("4. Constructing adjacency matrix with selected cells...")
    adj = adjacency(edge_pairs, avg_speed_new, idx_dict, sigma2=sigma2)

    return adj, node_pos


def construct_road_network_from_grid_speed_indices(n_row, n_col, file_dir, neigh_k, least_ratio, sigma2, indices):

    print("0. Get occurrence matrix...")
    count_np_2channel_file = os.path.join(file_dir, '..', 'speed_count_indices.npy')
    speed_2channel_file = os.path.join(file_dir, '..', 'speed_avg_indices.npy')
    files = list_filenames(file_dir, excluded_dates=[])
    num_days = 0
    if os.path.exists(count_np_2channel_file):
        count_np = np.load(count_np_2channel_file)
        avg_speed = np.load(speed_2channel_file)
    else:
        count_np_2 = np.zeros((len(indices)*3, n_row, n_col), dtype=np.int)
        speed_sum = np.zeros_like(count_np_2, dtype=np.float32)
        for file in files:
            try:
                data = load_input_data(os.path.join(file_dir, file), indices)
                data = data.reshape(-1, n_row, n_col, 3)
                num_days += 1
            except:
                continue

            non_zero_pos = data[..., 1] > 0
            non_zeros = np.zeros(count_np_2.shape, dtype=np.int)
            non_zeros[non_zero_pos] = 1
            count_np_2 += non_zeros
            speed_sum += data[..., 1]
        count_np = np.sum(count_np_2, axis=0)
        speed_sum_sum = np.sum(speed_sum, axis=0)
        avg_speed = speed_sum_sum / count_np
        avg_speed = np.where(np.isnan(avg_speed), np.zeros_like(avg_speed), avg_speed)
        avg_speed[np.isinf(avg_speed)] = 0.

        np.save(count_np_2channel_file, count_np)
        np.save(speed_2channel_file, avg_speed)

    # 2. select the cells that meet the threshold requirement
    print("1. Selecting cells with threshold...")
    # set a number threshold for holding data
    num_threshold = num_days * len(indices) * 3 * least_ratio
    selected_emerge = np.zeros_like(count_np, np.int)
    selected_emerge[count_np > num_threshold] = count_np[count_np > num_threshold]
    avg_speed_new = np.zeros_like(avg_speed, dtype=np.float32)
    # print("avg speed new shape is ", avg_speed_new.shape)
    # print("count np shape is ", count_np.shape)
    # print("avg_speed shape is ", avg_speed.shape)
    avg_speed_new[count_np > num_threshold] = avg_speed[count_np > num_threshold]

    # 1. construct a grid graph
    print("2. Constructing grid graph...")
    # z = grid(n_col, n_row)
    non_zeros_x, non_zeros_y = np.nonzero(selected_emerge)
    node_pos = np.stack([non_zeros_x, non_zeros_y], axis=1)
    _, edge_pairs = distance_sklearn_metrics(node_pos, k=neigh_k)

    # 3. get a correspondence of "pixel id"(row wise) with "graph node id"
    print("3. Constructing network with selected cells...")
    idx_dict, idx_revert_dict = construct_road_network(n_col, node_pos)

    # 4. construct a adjacency matrix of constructing road network
    print("4. Constructing adjacency matrix with selected cells...")
    adj = adjacency(edge_pairs, avg_speed_new, idx_dict, sigma2=sigma2)

    return adj, node_pos

def construct_road_network_from_grid_direction_5channels(
        n_row, n_col, file_dir, neigh_k, least_ratio, folder='training'):

    print("0. Get occurrence matrix...")
    count_np_2channel_file = os.path.join(file_dir, '..', 'direction_data_time_{}.npy'.format(folder))
    files = list_filenames(file_dir, excluded_dates=[])
    if os.path.exists(count_np_2channel_file):
        count_np_2 = np.load(count_np_2channel_file)
    else:
        count_np_2 = np.zeros((288, n_row, n_col, 5), dtype=np.int)
        for file in files:
            try:
                fr = h5py.File(os.path.join(file_dir, file), 'r')
                # shape (288, 495, 436, 3)
                data = fr['array'].value
                fr.close()
            except:
                continue
            # data_direction = np.expand_dims(data[..., 2], axis=-1)
            # tem_data = data_direction.copy()
            # data_direction = (data_direction // 85).astype(np.int) + 1
            # data_direction[tem_data == 0] = 0
            # count_np_2[data_direction] = 1.0

            direction = np.expand_dims(data[..., 2], axis=-1)

            direction_shape = list(direction.shape)
            direction_shape[-1] = 5
            direction_zeros = np.zeros(direction_shape, np.int)
            ## count the number of directions
            direction_zeros[:, :, :, 0] = np.sum(direction == 0, axis=-1)
            direction_zeros[:, :, :, 1] = np.sum(direction == 1, axis=-1)
            direction_zeros[:, :, :, 2] = np.sum(direction == 85, axis=-1)
            direction_zeros[:, :, :, 3] = np.sum(direction == 170, axis=-1)
            direction_zeros[:, :, :, 4] = np.sum(direction == 255, axis=-1)

            count_np_2 += direction_zeros

        # count_np_2 = np.sum(count_np_2, axis=0)
        np.save(count_np_2channel_file, count_np_2)

    print("1. Selecting cells with threshold...")
    # 2. sum all the directions across the time axis
    direction_sum_all_time = np.sum(count_np_2, axis=0)
    num_threshold = len(files) * 288 *  least_ratio / 2

    # from left to right
    direction_sum_left_right = direction_sum_all_time[:, :, 2] + direction_sum_all_time[:, :, 4]
    # get the interested nodes meet the least ratio requirement
    non_zeros_x, non_zeros_y = np.where(direction_sum_left_right > num_threshold)
    node_pos_l2r = np.stack([non_zeros_x, non_zeros_y], axis=1)
    _, edge_pairs = distance_sklearn_metrics_direction(node_pos_l2r, k=neigh_k, metric='euclidean', ncol=n_col, l2r=True)
    # 3. get a correspondence of "pixel id"(row wise) with "graph node id"
    print("3. Constructing network with selected cells...")
    idx_dict, idx_revert_dict = construct_road_network(n_col, node_pos_l2r)
    # 4. construct a adjacency matrix of constructing road network
    print("4. Constructing adjacency matrix with selected cells...")
    adj_l2r = adjacency(edge_pairs, direction_sum_left_right, idx_dict, directed=True)

    # from right to left
    direction_sum_left_right = direction_sum_all_time[:, :, 1] + direction_sum_all_time[:, :, 3]
    # get the interested nodes meet the least ratio requirement
    non_zeros_x, non_zeros_y = np.where(direction_sum_left_right > num_threshold)
    node_pos_r2l = np.stack([non_zeros_x, non_zeros_y], axis=1)
    _, edge_pairs = distance_sklearn_metrics_direction(node_pos_r2l, k=neigh_k, metric='euclidean', ncol=n_col, l2r=False)
    # 3. get a correspondence of "pixel id"(row wise) with "graph node id"
    print("3. Constructing network with selected cells...")
    idx_dict, idx_revert_dict = construct_road_network(n_col, node_pos_r2l)
    # 4. construct a adjacency matrix of constructing road network
    print("4. Constructing adjacency matrix with selected cells...")
    adj_r2l = adjacency(edge_pairs, direction_sum_left_right, idx_dict, directed=True)

    return adj_l2r, adj_r2l, node_pos_l2r, node_pos_r2l

def construct_road_network_from_grid_direction_5channels_in_out_bound(
        n_row, n_col, file_dir, neigh_k, least_ratio, folder='training', node_pos=None):

    print("0. Get occurrence matrix...")
    count_np_2channel_file = os.path.join(file_dir, '..', 'direction_data_time_{}.npy'.format(folder))
    files = list_filenames(file_dir, excluded_dates=[])
    if os.path.exists(count_np_2channel_file):
        count_np_2 = np.load(count_np_2channel_file)
    else:
        count_np_2 = np.zeros((288, n_row, n_col, 5), dtype=np.int)
        for file in files:
            try:
                fr = h5py.File(os.path.join(file_dir, file), 'r')
                # shape (288, 495, 436, 3)
                data = fr['array'].value
                fr.close()
            except:
                continue
            # data_direction = np.expand_dims(data[..., 2], axis=-1)
            # tem_data = data_direction.copy()
            # data_direction = (data_direction // 85).astype(np.int) + 1
            # data_direction[tem_data == 0] = 0
            # count_np_2[data_direction] = 1.0

            direction = np.expand_dims(data[..., 2], axis=-1)

            direction_shape = list(direction.shape)
            direction_shape[-1] = 5
            direction_zeros = np.zeros(direction_shape, np.int)
            ## count the number of directions
            direction_zeros[:, :, :, 0] = np.sum(direction == 0, axis=-1)
            direction_zeros[:, :, :, 1] = np.sum(direction == 1, axis=-1)
            direction_zeros[:, :, :, 2] = np.sum(direction == 85, axis=-1)
            direction_zeros[:, :, :, 3] = np.sum(direction == 170, axis=-1)
            direction_zeros[:, :, :, 4] = np.sum(direction == 255, axis=-1)

            count_np_2 += direction_zeros

        # count_np_2 = np.sum(count_np_2, axis=0)
        np.save(count_np_2channel_file, count_np_2)

    print("1. Selecting cells with threshold...")
    # 2. sum all the directions across the time axis
    direction_sum_all_time = np.sum(count_np_2, axis=0)
    num_threshold = len(files) * 288 * (1 - least_ratio)
    # get the interested nodes meet the least ratio requirement
    non_zeros_x, non_zeros_y = np.where(direction_sum_all_time[..., 0] < num_threshold)

    if node_pos is None:
        print("0. Get occurrence matrix...")
        count_np_2channel_file = os.path.join(file_dir, '..', 'speed_count.npy')
        speed_2channel_file = os.path.join(file_dir, '..', 'speed_avg.npy')
        files = list_filenames(file_dir, excluded_dates=[])
        num_days = len(files)
        if os.path.exists(count_np_2channel_file):
            count_np = np.load(count_np_2channel_file)
            avg_speed = np.load(speed_2channel_file)
        else:
            count_np_2 = np.zeros((288, n_row, n_col), dtype=np.int)
            speed_sum = np.zeros_like(count_np_2, dtype=np.float32)
            for file in files:
                try:
                    fr = h5py.File(os.path.join(file_dir, file), 'r')
                    data = fr['array'].value
                    fr.close()
                except:
                    continue
                non_zero_pos = data[..., 1] > 0
                non_zeros = np.zeros(count_np_2.shape, dtype=np.int)
                non_zeros[non_zero_pos] = 1
                count_np_2 += non_zeros
                speed_sum += data[..., 1]
            count_np = np.sum(count_np_2, axis=0)
            speed_sum_sum = np.sum(speed_sum, axis=0)
            avg_speed = speed_sum_sum / count_np
            avg_speed = np.where(np.isnan(avg_speed), np.zeros_like(avg_speed), avg_speed)
            np.save(count_np_2channel_file, count_np)
            np.save(speed_2channel_file, avg_speed)

        # 2. select the cells that meet the threshold requirement
        print("1. Selecting cells with threshold...")
        # set a number threshold for holding data
        num_threshold = num_days * 288 * least_ratio
        selected_emerge = np.zeros_like(count_np, np.int)
        selected_emerge[count_np > num_threshold] = count_np[count_np > num_threshold]
        avg_speed_new = np.zeros_like(avg_speed, dtype=np.float32)
        # print("avg speed new shape is ", avg_speed_new.shape)
        # print("count np shape is ", count_np.shape)
        # print("avg_speed shape is ", avg_speed.shape)
        avg_speed_new[count_np > num_threshold] = avg_speed[count_np > num_threshold]
        non_zeros_x, non_zeros_y = np.nonzero(selected_emerge)

        node_pos = np.stack([non_zeros_x, non_zeros_y], axis=1)

    print("3. Constructing network with selected cells...")
    idx_dict, idx_revert_dict = construct_road_network(n_col, node_pos)

    _, edge_pairs = distance_sklearn_metrics_direction(node_pos, k=neigh_k, metric='euclidean', ncol=n_col, l2r=True)
    # 3. get a correspondence of "pixel id"(row wise) with "graph node id"
    # from left to right
    direction_sum_left_right = direction_sum_all_time[:, :, 2] + direction_sum_all_time[:, :, 4]
    # 4. construct a adjacency matrix of constructing road network
    print("4. Constructing left to right adjacency matrix with selected cells...")
    adj_l2r = adjacency(edge_pairs, direction_sum_left_right, idx_dict, directed=True)

    _, edge_pairs = distance_sklearn_metrics_direction(node_pos, k=neigh_k, metric='euclidean', ncol=n_col, l2r=False)
    # from right to left
    direction_sum_left_right = direction_sum_all_time[:, :, 1] + direction_sum_all_time[:, :, 3]
    # 4. construct a adjacency matrix of constructing road network
    print("4. Constructing right to left adjacency matrix with selected cells...")
    adj_r2l = adjacency(edge_pairs, direction_sum_left_right, idx_dict, directed=True)

    return adj_l2r, adj_r2l, node_pos


def construct_road_network_from_grid_condense(
        row_patch, col_patch, file_dir, neigh_k=8, least_ratio=0.033):

    print("1. Query a nodes from the validation data folder ")
    # Search for all h5 files
    p = Path(file_dir)
    assert (p.is_dir())
    files = p.glob('*.h5')
    data_all = []
    for h5dataset_fp in files:
        file_path = str(h5dataset_fp.resolve())
        with h5py.File(file_path, 'r') as f:
            data = f['array'][()]
            data_all.append(data)
    data_all = np.stack(data_all, axis=0)
    batch, timeslots, rows, cols, num_channels = data_all.shape
    data_patch = np.reshape(data_all, (batch, timeslots, rows//row_patch, row_patch,
                                       cols//col_patch, col_patch, num_channels))
    non_zeros = np.sum(data_patch > 0, axis=(0, 1, 3, 5, 6))
    total_num_counts = batch * timeslots * row_patch * col_patch * num_channels
    non_zeros_x, non_zeros_y = np.nonzero(non_zeros > total_num_counts * least_ratio)
    node_pos = np.stack([non_zeros_x, non_zeros_y], axis=1)
    print("The constructed Node Number is ", node_pos.shape)
    print("non_zeros shape is ", non_zeros.shape)

    print("2. Constructing network with selected cells...")
    idx_dict, idx_revert_dict = construct_road_network(cols//col_patch, node_pos)

    _, edge_pairs = distance_sklearn_metrics(node_pos, k=neigh_k, metric='euclidean')
    # 4. construct a adjacency matrix of constructing road network
    print("3. Constructing adjacency matrix with selected cells...")
    adj = adjacency(edge_pairs, non_zeros, idx_dict)

    return adj, node_pos

def get_date_from_file(file):
    date_str = file[:8]
    date_time = date(year=int(date_str[:4]), month=int(date_str[4:6]), day=int(date_str[6:]))
    dow = date_time.weekday()

    return dow

def construct_road_network_from_grid_daily_7channels(n_row, n_col, file_dir, folder='validation'):

    print("Constructing daily 7channels")
    count_np_2channel_file = os.path.join(file_dir, '..', 'daily_direction_5channel_{}.npy'.format(folder))
    speed_volume_2channel_file = os.path.join(file_dir, '..', 'daily_sv_2channel_{}.npy'.format(folder))
    files = list_filenames(file_dir, excluded_dates=[])
    if os.path.exists(count_np_2channel_file) and os.path.exists(speed_volume_2channel_file):
        count_np_2 = np.load(count_np_2channel_file)
        sv_np_2 = np.load(speed_volume_2channel_file)
    else:
        count_np_2 = np.zeros((7, 288, n_row, n_col, 5), dtype=np.int)
        sv_np_2 = np.zeros((7, 288, n_row, n_col, 2), dtype=np.float)
        count_dow = np.zeros(7, np.int)
        for file in files:
            try:
                fr = h5py.File(os.path.join(file_dir, file), 'r')
                # shape (288, 495, 436, 3)
                data = fr['array'].value
                fr.close()
            except:
                continue
            # data_direction = np.expand_dims(data[..., 2], axis=-1)
            # tem_data = data_direction.copy()
            # data_direction = (data_direction // 85).astype(np.int) + 1
            # data_direction[tem_data == 0] = 0
            # count_np_2[data_direction] = 1.0

            # get day of week from file name
            dow = get_date_from_file(file)

            direction = np.expand_dims(data[..., 2], axis=-1)
            direction_shape = list(direction.shape)
            direction_shape[-1] = 5
            direction_zeros = np.zeros(direction_shape, np.int8)
            ## count the number of directions
            direction_zeros[:, :, :, 0] = np.sum(direction == 0, axis=-1)
            direction_zeros[:, :, :, 1] = np.sum(direction == 1, axis=-1)
            direction_zeros[:, :, :, 2] = np.sum(direction == 85, axis=-1)
            direction_zeros[:, :, :, 3] = np.sum(direction == 170, axis=-1)
            direction_zeros[:, :, :, 4] = np.sum(direction == 255, axis=-1)
            count_np_2[dow, ...] += direction_zeros

            sv_np_2[dow, ...] = data[..., :2]
            count_dow[dow] += 1

        for i in range(7):
            sv_np_2[i, ...] = sv_np_2[i, ...] / count_dow[i]

        # count_np_2 = np.sum(count_np_2, axis=0)
        np.save(count_np_2channel_file, count_np_2)
        np.save(speed_volume_2channel_file, sv_np_2)

    return count_np_2, sv_np_2


def distance_scipy_spatial(z, k=4, metric='euclidean'):
    """Compute exact pairwise distances."""
    d = scipy.spatial.distance.pdist(z, metric)
    d = scipy.spatial.distance.squareform(d)
    # k-NN graph.
    idx = np.argsort(d)[:, 1:k+1]
    d.sort()
    d = d[:, 1:k+1]
    return d, idx

def distance_sklearn_metrics(z, k=4, metric='euclidean'):
    """Compute exact pairwise distances."""
    d = sklearn.metrics.pairwise.pairwise_distances(
            z, metric=metric, n_jobs=-2)
    # k-NN graph.
    idx = np.argsort(d)[:, 1:k+1]
    d.sort()
    d = d[:, 1:k+1]
    # set idx of non-neighbour nodes to -1
    non_neighbour = d > np.sqrt(2)
    idx[non_neighbour] = -1

    return d, idx

def distance_sklearn_metrics_direction(z, k=4, metric='euclidean', ncol=436, l2r=True):
    """Compute exact pairwise distances."""
    num_nodes = z.shape[0]
    d = sklearn.metrics.pairwise.pairwise_distances(
            z, metric=metric, n_jobs=-2)
    # k-NN graph.
    idx = np.argsort(d)[:, 1:k+1]
    d.sort()
    d = d[:, 1:k+1]
    # set idx of non-neighbour nodes to -1
    non_neighbour = d > np.sqrt(2)
    idx[non_neighbour] = -1
    # get the col number of the selected idx
    col_idx = z[idx][:, :, 1] % ncol
    col_nodes = z[range(num_nodes), 1] % ncol
    if l2r:
        wrong_direction = col_idx < np.expand_dims(col_nodes, axis=1)
        idx[wrong_direction] = -1
    else:
        wrong_direction = col_idx > np.expand_dims(col_nodes, axis=1)
        idx[wrong_direction] = -1

    return d, idx

def distance_lshforest(z, k=4, metric='cosine'):
    """Return an approximation of the k-nearest cosine distances."""
    assert metric is 'cosine'
    lshf = sklearn.neighbors.LSHForest()
    lshf.fit(z)
    dist, idx = lshf.kneighbors(z, n_neighbors=k+1)
    assert dist.min() < 1e-10
    dist[dist < 0] = 0
    return dist, idx

# TODO: other ANNs s.a. NMSLIB, EFANNA, FLANN, Annoy, sklearn neighbors, PANN


# def adjacency(dist, idx):
#     """Return the adjacency matrix of a kNN graph."""
#     M, k = dist.shape
#     assert M, k == idx.shape
#     assert dist.min() >= 0
#
#     # Weights.
#     sigma2 = np.mean(dist[:, -1])**2
#     dist = np.exp(- dist**2 / sigma2)
#
#     # Weight matrix.
#     I = np.arange(0, M).repeat(k)
#     J = idx.reshape(M*k)
#     V = dist.reshape(M*k)
#     W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))
#
#     # No self-connections.
#     W.setdiag(0)
#
#     # Non-directed graph.
#     bigger = W.T > W
#     W = W - W.multiply(bigger) + W.T.multiply(bigger)
#
#     assert W.nnz % 2 == 0
#     assert np.abs(W - W.T).mean() < 1e-10
#     assert type(W) is scipy.sparse.csr.csr_matrix
#     return W


def replace_random_edges(A, noise_level):
    """Replace randomly chosen edges by random edges."""
    M, M = A.shape
    n = int(noise_level * A.nnz // 2)

    indices = np.random.permutation(A.nnz//2)[:n]
    rows = np.random.randint(0, M, n)
    cols = np.random.randint(0, M, n)
    vals = np.random.uniform(0, 1, n)
    assert len(indices) == len(rows) == len(cols) == len(vals)

    A_coo = scipy.sparse.triu(A, format='coo')
    assert A_coo.nnz == A.nnz // 2
    assert A_coo.nnz >= n
    A = A.tolil()

    for idx, row, col, val in zip(indices, rows, cols, vals):
        old_row = A_coo.row[idx]
        old_col = A_coo.col[idx]

        A[old_row, old_col] = 0
        A[old_col, old_row] = 0
        A[row, col] = 1
        A[col, row] = 1

    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


def laplacian(W, normalized=True):
    """Return the Laplacian of the weigth matrix."""

    # Degree matrix.
    d = W.sum(axis=0)

    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr.csr_matrix
    return L


def lmax(L, normalized=True):
    """Upper-bound on the spectrum."""
    if normalized:
        return 2
    else:
        return scipy.sparse.linalg.eigsh(
                L, k=1, which='LM', return_eigenvectors=False)[0]

def fourier(L, algo='eigh', k=1):
    """Return the Fourier basis, i.e. the EVD of the Laplacian."""

    def sort(lamb, U):
        idx = lamb.argsort()
        return lamb[idx], U[:, idx]

    if algo is 'eig':
        lamb, U = np.linalg.eig(L.toarray())
        lamb, U = sort(lamb, U)
    elif algo is 'eigh':
        lamb, U = np.linalg.eigh(L.toarray())
    elif algo is 'eigs':
        lamb, U = scipy.sparse.linalg.eigs(L, k=k, which='SM')
        lamb, U = sort(lamb, U)
    elif algo is 'eigsh':
        lamb, U = scipy.sparse.linalg.eigsh(L, k=k, which='SM')

    return lamb, U



def lanczos(L, X, K):
    """
    Given the graph Laplacian and a data matrix, return a data matrix which can
    be multiplied by the filter coefficients to filter X using the Lanczos
    polynomial approximation.
    """
    M, N = X.shape
    assert L.dtype == X.dtype

    def basis(L, X, K):
        """
        Lanczos algorithm which computes the orthogonal matrix V and the
        tri-diagonal matrix H.
        """
        a = np.empty((K, N), L.dtype)
        b = np.zeros((K, N), L.dtype)
        V = np.empty((K, M, N), L.dtype)
        V[0, ...] = X / np.linalg.norm(X, axis=0)
        for k in range(K-1):
            W = L.dot(V[k, ...])
            a[k, :] = np.sum(W * V[k, ...], axis=0)
            W = W - a[k, :] * V[k, ...] - (
                    b[k, :] * V[k-1, ...] if k > 0 else 0)
            b[k+1, :] = np.linalg.norm(W, axis=0)
            V[k+1, ...] = W / b[k+1, :]
        a[K-1, :] = np.sum(L.dot(V[K-1, ...]) * V[K-1, ...], axis=0)
        return V, a, b

    def diag_H(a, b, K):
        """Diagonalize the tri-diagonal H matrix."""
        H = np.zeros((K*K, N), a.dtype)
        H[:K**2:K+1, :] = a
        H[1:(K-1)*K:K+1, :] = b[1:, :]
        H.shape = (K, K, N)
        Q = np.linalg.eigh(H.T, UPLO='L')[1]
        Q = np.swapaxes(Q, 1, 2).T
        return Q

    V, a, b = basis(L, X, K)
    Q = diag_H(a, b, K)
    Xt = np.empty((K, M, N), L.dtype)
    for n in range(N):
        Xt[..., n] = Q[..., n].T.dot(V[..., n])
    Xt *= Q[0, :, np.newaxis, :]
    Xt *= np.linalg.norm(X, axis=0)
    return Xt  # Q[0, ...]


def rescale_L(L, lmax=2):
    """Rescale the Laplacian eigenvalues in [-1,1]."""
    M, M = L.shape
    I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
    L /= lmax / 2
    L -= I
    return L


def chebyshev(L, X, K):
    """Return T_k X where T_k are the Chebyshev polynomials of order up to K.
    Complexity is O(KMN)."""
    M, N = X.shape
    assert L.dtype == X.dtype

    # L = rescale_L(L, lmax)
    # Xt = T @ X: MxM @ MxN.
    Xt = np.empty((K, M, N), L.dtype)
    # Xt_0 = T_0 X = I X = X.
    Xt[0, ...] = X
    # Xt_1 = T_1 X = L X.
    if K > 1:
        Xt[1, ...] = L.dot(X)
    # Xt_k = 2 L Xt_k-1 - Xt_k-2.
    for k in range(2, K):
        Xt[k, ...] = 2 * L.dot(Xt[k-1, ...]) - Xt[k-2, ...]
    return Xt

# The following code is copied from DCRNN

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))

def laplacian(W, normalized=True):
    """Return the Laplacian of the weigth matrix."""

    # Degree matrix.
    d = W.sum(axis=0)

    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr.csr_matrix
    return L

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
        rows, cols = adj_mx.nonzero()
        adj_mx[cols, rows] = adj_mx[rows, cols]

    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = sp.linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I

    return L.astype(np.float32)


def calculate_gcn_semi_laplacian(adj_mx, undirected=True):
    if undirected:
        # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
        rows, cols = adj_mx.nonzero()
        adj_mx[cols, rows] = adj_mx[rows, cols]

    adj = sp.coo_matrix(adj_mx)
    adj += sp.eye(adj.shape[0])
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    L = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    L = sp.csr_matrix(L)

    return L.astype(np.float32)

def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1:
                print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            else:
                print(name, ':', num_param)
            total_param += num_param
    return total_param