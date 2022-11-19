"""
Placeholder class for communication with code base
Allows the current decision maker to be overriden with your RL input

 - Decision encoding: [0,1,2,NaN] = [left change, right change, no change, let MPC decide]
"""

import numpy as np


# Notes:
# - Feel free to edit this file as appropriate, changing template names requires changes troughout code base

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
        self.vehicleFeatures = []
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
        self.vehicleFeatures = features[:, 0, :].T

        print(self.vehicleFeatures.shape, self.vehicleFeatures)

    def getDecision(self):
        # Returns final decision for RL agents
        return self.decision
