"""
Placeholder class for communication with code base
Allows the current decision maker to be overriden with your RL input

 - Decision encoding: [0,1,2,NaN] = [left change, right change, no change, let MPC decide]
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from agents.graphs import GraphFactory, create_adjacency_matrix


class ReplayBuffer:
    """
    Experience replay for the Deep Q-Learning agent
    """

    def __init__(self, max_mem: int, device):
        self.device = device

        # The maximum size of the replay buffer
        self.mem_size = max_mem

        # Counts what memory address we are currently at
        self.mem_counter = 0

        self.state_mem = np.empty(shape=(self.mem_size,), dtype=object)
        self.action_mem = torch.zeros(self.mem_size, dtype=torch.int32)
        self.reward_mem = torch.zeros(self.mem_size, dtype=torch.float32)
        self.new_state_mem = np.empty(shape=(self.mem_size,), dtype=object)
        self.terminal_mem = torch.zeros(self.mem_size, dtype=torch.int32)

    def sample_transitions(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)

        # Samples uniformly random batches of size batch_size (with replacement)
        batch_idxs = np.randint(high=max_mem, size=(batch_size,))

        # Extract the tensors with the transitions and put them onto the computation device
        states = self.state_mem[batch_idxs]
        actions = self.action_mem[batch_idxs].to(self.device)
        rewards = self.reward_mem[batch_idxs].to(self.device)
        new_states = self.new_state_mem[batch_idxs]
        terminals = self.terminal_mem[batch_idxs].to(self.device)

        return states, actions, rewards, new_states, terminals

    def store_transition(self, state, action, reward, state_, terminal):
        idx = self.mem_counter % self.mem_size
        self.mem_counter += 1

        self.state_mem[idx] = state
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.new_state_mem[idx] = state_
        # If terminal -> 0, else 1 (used as multiplicative factor, since terminal states have a value of 0)
        self.terminal_mem[idx] = 1 - int(terminal)

    def reset(self):
        self.mem_counter = 0

        self.state_mem = np.empty(shape=(self.mem_size,), dtype=object)
        self.action_mem = torch.zeros(self.mem_size, dtype=torch.int32)
        self.reward_mem = torch.zeros(self.mem_size, dtype=torch.float32)
        self.new_state_mem = np.empty(shape=(self.mem_size,), dtype=object)
        self.terminal_mem = torch.zeros(self.mem_size, dtype=torch.bool)


# TODO: This is just an example configuration, we need to see what we actually want to do!
class GATQNetwork(torch.nn.Module):
    """
    Defines the neural network used by the Deep Q-Learning agent
    """

    def __init__(self, num_node_features, hidden_dim_size, num_actions):
        super(GATQNetwork, self).__init__()

        self.gat1 = GATConv(in_channels=num_node_features, out_channels=num_node_features)
        self.dense_1 = torch.nn.Linear(num_node_features, hidden_dim_size)
        self.output_layer = torch.nn.Linear(hidden_dim_size, num_actions)

    def forward(self, state):
        x, edge_index = state.x, state.edge_index

        x = F.relu(self.gat1(x, edge_index))

        # Pick the truck feature embeddings (which we will use to put through the network)
        x = x[0]

        # Put it through some final dense network
        x = F.relu(self.dense_1(x))
        q_values = self.output_layer(x)

        return q_values


def change_to_relative_pos(feature_matrix):
    feature_matrix = copy.deepcopy(feature_matrix)
    # We turn the absolute position in to relative coordinates with respect to the truck
    truck_x, truck_y = feature_matrix[0, :2]
    feature_matrix[:, 0] -= truck_x
    feature_matrix[:, 1] -= truck_y

    return feature_matrix


class DQNAgent:
    """
    RL agents class:
    Decides the appropriate choice of MPC path-planner

    Methods:
    - featchVehicleFeatures: Fetches the current vehicle states and classes
    - getDecision: Returns the appropriate trajectory option to decision master

    Variables:
    - vehicleFeatures: Vehicle states at the current time prior to executing the optimal control action
    - decision: Current decision made by the RL agents
    """

    def __init__(self, device, num_node_features, n_actions, gamma, target_copy_delay, learning_rate, batch_size, epsilon,
                 epsilon_dec=1e-3, epsilon_min=0.01, memory_size=1_000_000, file_name='models/dqn_model.pt'):
        """
        Args:
            device: CPU or GPU to put the data on (used for computations)
            input_shape: The dimensionality of the observation space
            n_actions: The number of possible actions
            gamma: The discount factor
            target_copy_delay: The number of iterations before synchronizing the policy- to the target network
            learning_rate: The learning rate
            batch_size: The number of samples to use in one iteration of mini-batch gradient descent
            epsilon: Random action probability (epsilon-greedy policy)
            epsilon_dec: The decrement in epsilon after one minibatch learning iteration
            epsilon_min: The minimum random action probability
            memory_size: The maximum number of transitions in the replay buffer
            file_name: The file name for when saving the model
        """
        self.device = device

        # The algorithm we use to decide how to put edges in our graph
        self.edge_creator = lambda feature_matrix: create_adjacency_matrix(feature_matrix, max_dx=100, max_dy=100)

        self.actions = [i for i in range(n_actions)]

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min

        self.C = target_copy_delay
        self.num_iters_until_net_sync = self.C

        self.model_file = file_name

        # Instantiate a replay buffer
        self.replay_buffer = ReplayBuffer(memory_size, device)

        # Build the Deep Q-Network (neural network)
        self.policy_net = GATQNetwork(
            num_node_features=num_node_features,
            hidden_dim_size=10,
            num_actions=n_actions
        ).to(device)
        self.target_net = copy.deepcopy(self.policy_net)

        # Mean squared error loss between target and predicted Q-values
        self.loss = torch.nn.SmoothL1Loss(beta=0.5)  # nn.MSELoss()

        # Optimizer used to update the network parameters
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def store_transition(self, state_features, action, reward, next_state_features, terminal_state):
        # TODO: We will need to store these vehicle features from inside the training loop
        # Fetches the most recent vehicle features, automatically refreshes each simulation step
        # The features are as follows:
        # size(features) = (#features, 1, #vehicles + 1) where "+1" comes from the truck
        # features[:, :, 0] is the truck
        # features are as follows: [px, py, v, theta, vehicle_type]
        # My thought is that the "1" was added for a batch size, but we can get rid of it for now

        # Now we have (#vehicles, #features) where features = (px, py, v, theta, vehicle_type)
        # And the truck is index 0. We also change the coordinate system to be centered at the truck
        state_features = change_to_relative_pos(state_features[:, 0, :].T)
        next_state_features = change_to_relative_pos(next_state_features[:, 0, :].T)

        # Constructs the actual state graphs
        state, _ = GraphFactory.create_graph(state_features, adjacency_matrix_function=self.edge_creator)
        next_state, _ = GraphFactory.create_graph(next_state_features, adjacency_matrix_function=self.edge_creator)

        self.replay_buffer.store_transition(state, action, reward, next_state, terminal_state)

    def choose_action(self, state_features):
        """
        Chooses an action according to an epsilon-greedy policy

        Args:
            state_features: measures information about the state that the agent observes

        Returns:
            The action to perform
        """
        state_features = change_to_relative_pos(state_features[:, 0, :].T)
        state, _ = GraphFactory.create_graph(state_features, adjacency_matrix_function=self.edge_creator)
        state = state.to(device=self.device)

        action = np.random.choice(self.actions)

        if np.random.random() >= self.epsilon:
            with torch.no_grad():
                q_pred = self.policy_net.forward(state)
                # Choose the action by taking the largest Q-value
                action = torch.argmax(q_pred).item()
        return action

    def learn(self):
        """
        Performs one iteration of minibatch gradient descent
        """
        # Return if the buffer does not yet contain enough samples
        if self.replay_buffer.mem_counter < self.batch_size:
            return

        # Sample from the replay buffer
        states, actions, rewards, new_states, terminals = self.replay_buffer.sample_transitions(self.batch_size)

        # Predict the Q-values in the current state, and in the new state (after taking the action)
        # Unfortunately we cannot do this in parallel
        q_preds = torch.zeros(size=(self.batch_size,), dtype=torch.float32, device=self.device)
        for i, state in enumerate(states):
            state = state.to(device=self.device)
            q_preds[i] = self.policy_net.forward(state)

        # If C = 0, we would update the network at every iteration, which would be crazy inefficient
        # It equals training without a target network
        q_targets = torch.zeros(size=(self.batch_size,), dtype=torch.float32, device=self.device)
        if self.C == 0:
            for i, state in enumerate(new_states):
                state = state.to(device=self.device)
                q_targets[i] = self.policy_net.forward(state)
        else:
            for i, state in enumerate(new_states):
                state = state.to(device=self.device)
                q_targets[i] = self.target_net.forward(state)

        # For every sampled transition, set the target for the action that was taken as
        # defined earlier: r + gamma * max_a' Q(s', a')
        y = rewards + self.gamma * terminals * torch.max(q_targets, dim=1)[0]

        # Get a list of indices [0, 1, ..., batch_size-1]
        batch_idxs = torch.arange(self.batch_size, dtype=torch.long)

        # Perform one iteration of minibatch gradient descent: reset the gradients, compute the loss, clamp the
        # gradients, and update
        self.optimizer.zero_grad()

        output = self.loss(y, q_preds[batch_idxs, actions.long()]).to(self.device)
        output.backward()

        # Clamp the gradients in a range between -1 and 1
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0)

        self.optimizer.step()

        # Decrease the random action probability if its minimum has not yet been reached
        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

        # Copy the policy network weights to the target network every C iterations
        if self.C != 0:
            if self.num_iters_until_net_sync == 0:
                self.num_iters_until_net_sync = self.C
                self.target_net.load_state_dict(self.policy_net.state_dict())
            else:
                self.num_iters_until_net_sync -= 1

    def save_model(self, file_path=''):
        """
        Stores the Deep Q-Network state in a file

        Args:
            file_path: Path and file name to store the model to
        """
        if file_path == '':
            file_path = self.model_file
        torch.save(self.policy_net.state_dict(), file_path)

    def load_model(self, file_path=''):
        """
        Loads the Deep Q-Network state from a file

        Args:
            file_path: Path to the file of the model to load
        """
        if file_path == '':
            file_path = self.model_file
        self.policy_net.load_state_dict(torch.load(file_path))
