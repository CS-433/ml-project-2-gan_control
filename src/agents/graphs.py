import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

import matplotlib.pyplot as plt
import networkx as nx


def draw_graph(feature_matrix, graph, node_id_to_vehicle_id_mapping, min_x=-50, max_x=50, min_y=-50, max_y=50, save_name=''):
    """
    Draws the graph and print its nodes' feature information
    :param save_name: The filename of the figure in case it needs to be exported
    :param max_y: The maximum y-coordinate to plot
    :param min_y: The minimum y-coordinate to plot
    :param max_x: The maximum x-coordinate to plot
    :param min_x: The minimum x-coordinate to plot
    :param feature_matrix: The features of every vehicle
    :param graph: The graph that was created by the GraphFactory
    :param node_id_to_vehicle_id_mapping: The mapping from node ids to vehicle ids
    :return: Nothing, but plots the graph and prints its nodes' features
    """
    # Compute the positions to plot the graph on
    positions = {
        n_id: [(feature_matrix[v_id][0] - min_x) / (max_x - min_x), (feature_matrix[v_id][1] - min_y) / (max_y - min_y)]
        for n_id, v_id in node_id_to_vehicle_id_mapping.items()
    }

    # We add "Vehicle" in front of the label to help match the nodes in the graph
    for k, v in node_id_to_vehicle_id_mapping.items():
        node_id_to_vehicle_id_mapping[k] = "Vehicle " + str(v)

    g = to_networkx(graph, to_undirected=True)
    nx.draw(g, pos=positions, labels=node_id_to_vehicle_id_mapping)

    if save_name != '':
        plt.savefig(save_name)

    plt.show()

    for i, features in enumerate(graph.x):
        print(node_id_to_vehicle_id_mapping[i], features)


def create_adjacency_matrix(feature_matrix, verbose=False, max_dx=10, max_dy=30):
    """
    Default "adjacency matrix" creation function. Each of these functions MUST follow the same method signature
    :param max_dy: The maximum y-distance between two neighboring vertices
    :param max_dx: The maximum x-distance between two neighboring vertices
    :param verbose: Whether to print the breath-first search process
    :param feature_matrix: Shape (#vehicles, #features) where the first index is the truck
    :return: A list of edge tuples, the vehicle ids in the subgraph, and a mapping from node ids to vehicle ids
    """

    edge_list = set()
    connected_idxs = {0}

    # We need to map the vehicle ids to node ids. Notice that the index of the truck node stays 0!
    vehicle_id_to_node_id_mapping = {0: 0}
    map_ctr = 1

    # O(|V|^2). We find the whole subgraph connected to the truck using breadth-first search
    q = [0]
    while len(q) > 0:
        vehicle_id = q.pop(0)

        if verbose:
            print(vehicle_id)

        for other_vehicle_id in range(len(feature_matrix)):
            if vehicle_id == other_vehicle_id:
                continue

            # NOTE: THIS ASSUMED THAT THE FIRST TWO FEATURES ARE THE X AND Y POSITIONS OF THE CAR
            if abs(feature_matrix[vehicle_id][0] - feature_matrix[other_vehicle_id][0]) <= max_dx and \
                    abs(feature_matrix[vehicle_id][1] - feature_matrix[other_vehicle_id][1]) <= max_dy:

                if other_vehicle_id not in vehicle_id_to_node_id_mapping.keys():
                    vehicle_id_to_node_id_mapping[other_vehicle_id] = map_ctr
                    map_ctr += 1

                if verbose:
                    print("\t" + str(other_vehicle_id), end=' ')

                # It's a directed graph, so we add edges in both directions
                edge_list.add(
                    (vehicle_id_to_node_id_mapping[vehicle_id], vehicle_id_to_node_id_mapping[other_vehicle_id]))
                edge_list.add(
                    (vehicle_id_to_node_id_mapping[other_vehicle_id], vehicle_id_to_node_id_mapping[vehicle_id]))

                # If it's within the range, and we haven't seen it yet, we want to explore the other vehicle
                # to see if it's connected more vehicles in the subgraph of the truck
                if other_vehicle_id not in connected_idxs:
                    if verbose:
                        print(True)
                    q.append(other_vehicle_id)
                    connected_idxs.add(other_vehicle_id)
                else:
                    if verbose:
                        print(False)

    return torch.tensor(list(edge_list)), torch.tensor(list(connected_idxs)), \
           {k: v for v, k in vehicle_id_to_node_id_mapping.items()}


class GraphFactory:
    """
    This class follows the Factory design pattern to generate graphs
    """

    @staticmethod
    def create_graph(feature_matrix, adjacency_matrix_function=create_adjacency_matrix):
        """
        Returns the graph representation with node features from "feature_matrix" and edges from "adjacency_matrix_function"
        :param feature_matrix: Shape (#vehicles, #features) where the first index is the truck
        :param adjacency_matrix_function: function to decide when an edge exists between nodes
        :return: The constructed graph representation
        """
        edge_tuple_list, used_vehicle_idxs, node_id_to_vehicle_id_mapping = adjacency_matrix_function(feature_matrix)

        used_vehicle_features = feature_matrix[list(node_id_to_vehicle_id_mapping.values())]

        # Creates a homogeneous graph, more information on
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs
        generated_graph = Data(x=torch.tensor(used_vehicle_features, dtype=torch.float32),
                               edge_index=edge_tuple_list.t().contiguous())

        return generated_graph, node_id_to_vehicle_id_mapping


if __name__ == '__main__':
    # This is just an example graph

    # Just for drawing on correct positions
    min_x, max_x, min_y, max_y = 50, 150, 50, 150

    feature_matrix = np.array([
        [100, 100],  # Vehicle 0
        [111, 111],  # Vehicle 1
        [90, 90],  # Vehicle 2
        [80, 90],  # Vehicle 3
        [80, 80],  # Vehicle 4
        [100, 110],  # Vehicle 5
        [60, 70],  # Vehicle 6
        [95, 95],  # Vehicle 7
        [120, 100]
    ])

    graph, node_id_to_vehicle_id_mapping = GraphFactory.create_graph(feature_matrix)

    #draw_graph(feature_matrix, graph, node_id_to_vehicle_id_mapping, min_x, max_x, min_y, max_y)