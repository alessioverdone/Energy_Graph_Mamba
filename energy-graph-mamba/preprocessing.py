import os
import torch
import chardet
from geopy.distance import geodesic
import json
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import networkx as nx


def get_coordinates(path, light_db_index=0):
    """
    path (str): Path where there are time series txt files
    light_db_index: Index to decide the number of PV plants to use
    """
    list_of_time_series_data = os.listdir(path)
    if light_db_index != 0:
        list_of_time_series_data = list_of_time_series_data[:light_db_index]
    list_of_coordinates = []
    for index, file in enumerate(list_of_time_series_data):
        line_list = file.rstrip('\n').split('_')
        lat = line_list[1]
        long = line_list[2]
        list_of_coordinates.append([lat, long])
    print(list_of_coordinates)

    return list_of_coordinates


def get_distance_matrix(list_of_coordinates, threshold=9999):
    """
    list_of_coordinates (list): list_of_coordinates of each PV plants for build the graph
    threshold (int): kilometers threshold to decide edges
    """
    size = len(list_of_coordinates)
    distance_matrix = torch.zeros((size, size))
    edges = []
    for index_i, coord_i in enumerate(list_of_coordinates):
        for index_j, coord_j in enumerate(list_of_coordinates):
            if index_i == index_j:
                distance_matrix[index_i, index_j] = 0
                continue

            dist = float(geodesic(coord_i, coord_j).km)
            if dist <= threshold:

                # OLD
                distance_matrix[index_i, index_j] = dist
                edges.append([index_i, index_j])

                # # START NEW - Modifica per non fare edge doppi
                # if [index_j, index_i] not in edges:
                #     edges.append([index_i, index_j])
                #     distance_matrix[index_i, index_j] = dist
                # else:
                #     distance_matrix[index_i, index_j] = 0
                # # END NEW
            else:
                distance_matrix[index_i, index_j] = 0
    return distance_matrix, edges


def scaling_matrix(m):
    """
    m: Target distance matrix
    """
    scaler = MinMaxScaler()
    scaler.fit(m)
    m = scaler.transform(m)
    return m


def visualize_matrix(matrix):
    """
    matrix: Target distance matrix
    """
    plt.matshow(matrix)
    plt.show()


def visualize_graph(edges):
    g = nx.Graph()
    for elem in edges:
        g.add_edge(elem[0], elem[1], color='g')

    pos = nx.spring_layout(g)  # spring_layout, random_layout, kamada_kawai_layout
    nx.draw(g, pos,
            with_labels=True,
            node_color='lightgreen')
    plt.show()


def get_min_max_dist(m):
    min_d = torch.min(m, dim=0)
    max_d = torch.min(m, dim=0)
    return min_d, max_d


def remove_zero_weight(weights):
    new_list = list()
    for list_i in weights:
        for j in list_i:
            if j != 0.0:
                new_list.append(j)
    print("Edges: " + str(len(new_list)))
    return new_list


def create_weigths_and_edges_from_station_coordinates(path_folder, light_db_index):
    """
    path_folder (str): Directory with single .txt PV files
    light_db_index (int): Index to decide the number of PV plants to use
    """
    list_of_coordinates = get_coordinates(path_folder, light_db_index)
    threshold = 50
    distance_matrix, edges = get_distance_matrix(list_of_coordinates, threshold)
    max_value = torch.max(distance_matrix)
    distance_matrix = distance_matrix / max_value
    # distance_matrix = scaling_matrix(distance_matrix)  # if it needed to scale the matrix
    visualize_matrix(distance_matrix)
    # min_dist, max_dist = get_min_max_dist(distance_matrix)
    print(distance_matrix)
    visualize_graph(edges)
    return distance_matrix, edges


def process_txt_data_multivariate_time(path_folder, filename):
    """
    path_folder (str): Directory with single .txt PV files
    filename (str): single PV file
    """
    path = os.path.join(path_folder, filename)
    list_power = []
    list_temperature = []
    list_wind = []
    list_month = []
    list_hour = []
    with open(path, 'rb') as file:
        result = chardet.detect(file.read())
    with open(path, 'r', encoding=result['encoding']) as f:
        cont = 0
        start_flag = False
        for line in f:
            line_list = line.rstrip('\n').split(',')
            if not (start_flag):  # Start time series condition
                if line_list[0] == 'time':
                    start_flag = True
                    continue
                else:
                    continue
            if line_list[0] == 'P':  # End time series condition
                list_power = list_power[:-1]
                list_temperature = list_temperature[:-1]
                list_wind = list_wind[-1]
                cont -= 1
                break
            cont += 1
            # Da qui in giù lavori con la lista
            if len(line_list) > 6:
                list_power.append(float(line_list[1]))
                list_temperature.append(float(line_list[4]))
                list_wind.append(float(line_list[5]))
                time = line_list[0]
                month = time[4:6]
                hour = time[9:11]
                list_month.append(float(month))
                list_hour.append(float(hour))

        f.close()
        return torch.tensor(list_power), torch.tensor(list_temperature), torch.tensor(list_wind), \
            torch.tensor(list_month), torch.tensor(list_hour)


def create_aggregate_tensor_multivariate_time(path_folder, light_db_index=0):
    """
    path_folder (str): Directory with single .txt PV files
    light_db_index (int): Index to decide the number of PV plants to use
    """
    list_of_time_series_data = os.listdir(path_folder)
    if light_db_index != 0:
        list_of_time_series_data = list_of_time_series_data[:light_db_index]
    start = True
    for index, file in enumerate(list_of_time_series_data):
        data, temperature, wind, month, hour = process_txt_data_multivariate_time(path_folder, file)
        print(data.shape, temperature.shape, wind.shape, month.shape, hour.shape)
        if start:
            matrix_values = torch.zeros((data.shape[0], len(list_of_time_series_data)))
            matrix_values_temp = torch.zeros((temperature.shape[0], len(list_of_time_series_data)))
            matrix_values_wind = torch.zeros((wind.shape[0], len(list_of_time_series_data)))
            matrix_values_month = torch.zeros((month.shape[0], len(list_of_time_series_data)))
            matrix_values_hour = torch.zeros((hour.shape[0], len(list_of_time_series_data)))
            start = False
        matrix_values[:, index] = data
        matrix_values_temp[:, index] = temperature
        matrix_values_wind[:, index] = wind
        matrix_values_month[:, index] = month
        matrix_values_hour[:, index] = hour

    all_matrix = [matrix_values, matrix_values_temp, matrix_values_wind, matrix_values_month, matrix_values_hour]

    print("Done!")
    return all_matrix


def create_json_database_with_weight_and_treshold_multivariate_time(all_matrix_list, weights, edges):
    """
    all_matrix_list (list): list of features data
    weights: edges data weights
    edges: created edges
    """
    output_data = all_matrix_list[0]
    temp_data = all_matrix_list[1]
    wind_data = all_matrix_list[2]
    month_data = all_matrix_list[3]
    hour_data = all_matrix_list[4]
    final_dict = {"block": output_data.tolist(),
                  "block_temp": temp_data.tolist(),
                  "block_wind": wind_data.tolist(),
                  "block_month": month_data.tolist(),
                  "block_hour": hour_data.tolist(),
                  "time_periods": 0,
                  "weights": [],
                  "edges": []}
    final_dict["time_periods"] = len(final_dict["block"])
    final_dict["weights"] = remove_zero_weight(weights.tolist())
    final_dict["edges"] = edges
    return final_dict


def main_weight_multivariate_time():
    """
    Main function for generating .json file in PyTorch Spatio-Temporal dataset format

    """
    light_db_index = 0  # 0 == all PV plants, n == n plants
    path_folder = "Txt_files"
    all_matrix_list = create_aggregate_tensor_multivariate_time(path_folder, light_db_index)
    weights, edges = create_weigths_and_edges_from_station_coordinates(path_folder, light_db_index)
    db = create_json_database_with_weight_and_treshold_multivariate_time(all_matrix_list, weights, edges)
    return db


if __name__ == '__main__':
    # Generate .json dataset from PVGIS .txt (or .csv) files
    dataset_name = os.path.join("../data", "num_nodes_multivariate.json")
    json_data = main_weight_multivariate_time()

    # Save data.json
    with open(dataset_name, 'w+') as f:
        json.dump(json_data, f)
