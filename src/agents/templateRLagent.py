"""
Placeholder class for communication with code base
Allows the current decision maker to be overriden with your RL input

 - Decision encoding: [0,1,2,NaN] = [left change, right change, no change, let MPC decide]
"""

from agents.graphs import GraphFactory, draw_graph, create_adjacency_matrix


# Notes:
# - Feel free to edit this file as appropriate, changing template names requires changes throughout code base

class RLAgent:
    """
    RL agents class:
    Decides the appropriate choice of MPC pathplanner

    Methods:
    - featchVehicleFeatures: Fetches the current vehicle states and classes
    - getDecision: Returns the appropriate trajectory option to decision master

    Variables:
    - vehicleFeatures: Vehicle states at the current time prior to executing the optimal control action
    - decision: Current decision made by the RL agents
    """

    def __init__(self):
        self.node_features = []
        self.decision = float('nan')

    def fetchVehicleFeatures(self, features):
        # Fetches the most recent vehicle features, automatically refreshes each simulation step
        # The features are as follows:
        # size(features) = (#features, 1, #vehicles + 1) where "+1" comes from the truck
        # features[:, :, 0] is the truck
        # features are as follows: [px, py, v, theta, vehicle_type]
        # My thought is that the "1" was added for a batch size, but we can get rid of it for now

        # Now we have (#vehicles, #features) where features = (px, py, v, theta, vehicle_type)
        # And the truck is index 0
        self.node_features = features[:, 0, :].T

        # Now we turn the absolute position in to relative coordinates with respect to the truck
        truck_x, truck_y = self.node_features[0, :2]
        self.node_features[:, 0] -= truck_x
        self.node_features[:, 1] -= truck_y

    def getDecision(self):
        # Constructs the graph
        max_dist_edge_creator = lambda feature_matrix: create_adjacency_matrix(feature_matrix, max_dx=100, max_dy=100)
        graph, node_id_to_vehicle_id_mapping = GraphFactory.create_graph(self.node_features,
                                                                         adjacency_matrix_function=max_dist_edge_creator)

        # This is for visualizing the graph that was created along with its features
        draw_graph(self.node_features, graph, node_id_to_vehicle_id_mapping, min_x=-150, max_x=150, min_y=-10, max_y=10)
        print(self.node_features)

        return self.decision
