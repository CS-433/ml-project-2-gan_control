"""
Placeholder class for communication with code base
Allows the current decision maker to be overriden with your RL input

 - Decision encoding: [0,1,2,NaN] = [left change, right change, no change, let MPC decide]
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

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
        self.lane_mask = torch.zeros((self.mem_size, 2), dtype=torch.bool)

    def sample_transitions(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)

        # Samples uniformly random batches of size batch_size (with replacement)
        batch_idxs = np.random.randint(low=0, high=max_mem, size=(batch_size,))

        # Extract the tensors with the transitions and put them onto the computation device
        states = self.state_mem[batch_idxs]
        actions = self.action_mem[batch_idxs].to(self.device)
        rewards = self.reward_mem[batch_idxs].to(self.device)
        new_states = self.new_state_mem[batch_idxs]
        terminals = self.terminal_mem[batch_idxs].to(self.device)
        lane_masks = self.lane_mask[batch_idxs].to(self.device)

        return states, actions, rewards, new_states, terminals, lane_masks

    def store_transition(self, state, action, reward, state_, terminal, lane_mask):
        idx = self.mem_counter % self.mem_size
        self.mem_counter += 1

        self.state_mem[idx] = state
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.new_state_mem[idx] = state_
        # If terminal -> 0, else 1 (used as multiplicative factor, since terminal states have a value of 0)
        self.terminal_mem[idx] = 1 - int(terminal)
        self.lane_mask[idx] = lane_mask

    def reset(self):
        self.mem_counter = 0

        self.state_mem = np.empty(shape=(self.mem_size,), dtype=object)
        self.action_mem = torch.zeros(self.mem_size, dtype=torch.int32)
        self.reward_mem = torch.zeros(self.mem_size, dtype=torch.float32)
        self.new_state_mem = np.empty(shape=(self.mem_size,), dtype=object)
        self.terminal_mem = torch.zeros(self.mem_size, dtype=torch.bool)
        self.lane_mask = torch.zeros((self.mem_size, 2), dtype=torch.bool)


# TODO: This is just an example configuration, we need to see what we actually want to do!
class GATQNetwork(torch.nn.Module):
    """
    Defines the neural network used by the Deep Q-Learning agent
    """

    def __init__(self, num_node_features, hidden_dim_size, num_actions):
        super(GATQNetwork, self).__init__()

        # Note: we already add self loops in the graph factory
        self.gat1 = GATv2Conv(in_channels=num_node_features, out_channels=2 * num_node_features, add_self_loops=False)
        self.gat2 = GATv2Conv(in_channels=2 * num_node_features, out_channels=4 * num_node_features, add_self_loops=False)

        self.dense_1 = torch.nn.Linear(4 * num_node_features, hidden_dim_size)
        self.output_layer = torch.nn.Linear(hidden_dim_size, num_actions)

    def forward(self, state):
        x, edge_index = state.x, state.edge_index

        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))

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


def rescale_speeds_relative_to_limit(feature_matrix, speed_lim):
    feature_matrix = copy.deepcopy(feature_matrix)
    feature_matrix[:, 2] = (feature_matrix[:, 2] - speed_lim) / speed_lim

    return feature_matrix


def preprocess_state_features(state_features, speed_lim):
    # preprocesses the state feature matrix to prepare it for graph creation.
    # returns the updated state_feature matrix, and the associated lane mask

    def _lane_num_to_lane_mask(lane_num, leftmost_lane=1, rightmost_lane=-1):
        # function to convert lane number to a lane mask. returns a torch
        # BoolTensor of length 2, where the first and second values indicate if
        # the left and right lane changes should be masked respectively
        ## PROBABLY SHOULD MOVE THIS FUNCTION ELSEWHERE BUT NOT SURE THE BEST
        ## PLACE TO PUT IT
        return torch.BoolTensor([lane_num == leftmost_lane,
                                 lane_num == rightmost_lane])

    # size(features) = (#features, 1, #vehicles + 1) where "+1" comes from the truck
    # features[:, :, 0] is the truck
    # change the coordinate system to be centered at the truck
    # Now we have (#vehicles, #features) where features = (px, py, v, theta, vehicle_type, lane_num)
    state_features = change_to_relative_pos(state_features[:, 0, :].T)

    state_features = rescale_speeds_relative_to_limit(state_features, speed_lim)

    # get mask for disallowed lane changes
    lane_mask = _lane_num_to_lane_mask(state_features[0, -1])
    # remove lane number and vehicle type features
    state_features = state_features[:, :-2]

    return state_features, lane_mask


def get_best_valid_action(q_vals, lane_mask):
    # returns the action with highest q value (along with the q value itself)
    # after removing invalid actions according to the lane mask

    # 0 is left, 1 is right, 2 is continue straight. straight is always
    # an option so this decision should never be masked (for out simple
    # scenario at least where lanes don't reduce etc.)
    decision_mask = torch.cat((lane_mask, torch.BoolTensor([False])))

    if decision_mask[0] == True or decision_mask[1] == True:
        print('', end='')

    # bit dodgy but set q vals for any invalid decision to less than min
    masked_q_vals = torch.where(decision_mask, q_vals.min() - .1, q_vals)

    best_action = torch.argmax(masked_q_vals).item()
    q_max = torch.max(masked_q_vals)

    # print(lane_mask)
    # print(q_vals)
    # print(decision_mask)
    # print(masked_q_vals)
    # print(best_action)
    # print(q_max)
    # print('\n\n')

    return best_action, q_max


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

    def __init__(self, device, num_node_features, n_actions, speed_lim,
                 gamma=0.9, target_copy_delay=0, learning_rate=10e-3,
                 batch_size=32, epsilon=0.01, epsilon_dec=1e-3, epsilon_min=0.01,
                 memory_size=1_000_000, file_name='out/models/dqn_model.pt'):
        """
        Args:
            device: CPU or GPU to put the data on (used for computations)
            input_shape: The dimensionality of the observation space
            n_actions: The number of possible actions
            speed_lim: Speed limit of highway/road
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

        self.speed_lim = speed_lim

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
        self.loss = torch.nn.MSELoss()  # torch.nn.SmoothL1Loss(beta=0.5)

        # Optimizer used to update the network parameters
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def store_transition(self, state_features, action, reward, next_state_features, terminal_state):
        # TODO: We will need to store these vehicle features from inside the training loop
        # Fetches the most recent vehicle features, automatically refreshes each simulation step
        # The features are as follows:
        # size(features) = (#features, 1, #vehicles + 1) where "+1" comes from the truck
        # features[:, :, 0] is the truck
        # features are as follows: [px, py, v, theta, vehicle_type, lane_num]
        # My thought is that the "1" was added for a batch size, but we can get rid of it for now

        state_features, lane_mask = preprocess_state_features(state_features, self.speed_lim)
        next_state_features, _ = preprocess_state_features(next_state_features, self.speed_lim)

        # Constructs the actual state graphs
        state, _ = GraphFactory.create_graph(state_features, adjacency_matrix_function=self.edge_creator)
        next_state, _ = GraphFactory.create_graph(next_state_features, adjacency_matrix_function=self.edge_creator)

        self.replay_buffer.store_transition(state, action, reward, next_state, terminal_state, lane_mask)

    def choose_action(self, state_features):
        """
        Chooses an action according to an epsilon-greedy policy

        Args:
            state_features: measures information about the state that the agent observes

        Returns:
            The action to perform
        """
        state_features, lane_mask = preprocess_state_features(state_features, self.speed_lim)
        decision_mask = lane_mask.tolist()
        # 0 is left, 1 is right, 2 is continue straight. straight is always
        # an option so this decision should never be masked (for out simple
        # scenario at least where lanes don't reduce etc.)
        decision_mask.append(False)

        state, _ = GraphFactory.create_graph(state_features, adjacency_matrix_function=self.edge_creator)
        state = state.to(device=self.device)

        possible_actions = [a for i, a in enumerate(self.actions) if not decision_mask[i]]
        action = np.random.choice(possible_actions)

        if np.random.random() >= self.epsilon:
            with torch.no_grad():
                q_pred = self.policy_net(state)
                # Choose the action by taking the largest Q-value
                action, _ = get_best_valid_action(q_pred, lane_mask)
        return action

    def learn(self):
        """
        Performs one iteration of minibatch gradient descent
        """
        # Return if the buffer does not yet contain enough samples
        if self.replay_buffer.mem_counter < self.batch_size:
            return

        # Sample from the replay buffer
        states, actions, rewards, new_states, terminals, lane_masks = self.replay_buffer.sample_transitions(
            self.batch_size)

        # Predict the Q-values in the current state, and in the new state (after taking the action)
        # Unfortunately we cannot do this in parallel
        q_preds = torch.zeros(size=(self.batch_size, len(self.actions)), dtype=torch.float32, device=self.device)
        for i, state in enumerate(states):
            state = state.to(device=self.device)
            q_preds[i] = self.policy_net(state)

        # If C = 0, we would update the network at every iteration, which would be crazy inefficient
        # It equals training without a target network
        q_targets = torch.zeros(size=(self.batch_size,), dtype=torch.float32, device=self.device)
        if self.C == 0:
            for i, state in enumerate(new_states):
                state = state.to(device=self.device)
                _, max_q = get_best_valid_action(self.policy_net(state),
                                                 lane_masks[i])
                q_targets[i] = max_q
        else:
            for i, state in enumerate(new_states):
                state = state.to(device=self.device)
                _, max_q = get_best_valid_action(self.target_net(state),
                                                 lane_masks[i])
                q_targets[i] = max_q

        # For every sampled transition, set the target for the action that was taken as
        # defined earlier: r + gamma * max_a' Q(s', a')
        y = rewards + self.gamma * terminals * q_targets

        # Get a list of indices [0, 1, ..., batch_size-1]
        batch_idxs = torch.arange(self.batch_size, dtype=torch.long)

        # Perform one iteration of minibatch gradient descent: reset the gradients, compute the loss, clamp the
        # gradients, and update
        self.optimizer.zero_grad()

        output = self.loss(y, q_preds[batch_idxs, actions.long()]).to(self.device)
        output.backward()

        # Clamp the gradients in a range between -1 and 1 (Only with Huber loss, for now we use MSE)
        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0)

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


if __name__ == '__main__':
    # This is just an example graph

    # Just for drawing on correct positions
    min_x, max_x, min_y, max_y = 50, 150, 50, 150

    feature_matrix = np.array([
        [100.0, 100],  # Vehicle 0
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

    # draw_graph(feature_matrix, graph, node_id_to_vehicle_id_mapping, min_x, max_x, min_y, max_y)

    nn = GATQNetwork(2, 10, 4)

    with torch.no_grad():
        print(nn(graph))
