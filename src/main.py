import argparse
import time
import json
import os

# Packages
import torch
from casadi import *
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Classes and helpers
from vehicleModelGarage import vehBicycleKinematic
from scenarios import trailing, simpleOvertake
from traffic import vehicleSUMO, combinedTraffic
from controllers import makeController, makeDecisionMaster
from helpers import *

from agents.templateRLagent import DQNAgent

np.random.seed(1)

# ----------------------------------------------------------
# Constants
# ----------------------------------------------------------

device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters required to be included in the hyperparam config file
# dictionary values indicate the type of value required for each hyperparam
REQ_HYPERPARAMS = {"gamma": "float", "target_copy_delay": "int",
                   "learning_rate": "float", "batch_size": "int",
                   "epsilon": "float", "epsilon_dec": "float",
                   "epsilon_min": "float", "memory_size": "int"}
REQ_HYPERPARAM_DESC = ', '.join(['%s (%s)' % (k, v)
                                 for k, v in REQ_HYPERPARAMS.items()])

num_node_features = 4
n_actions = 3

# Set Gif-generation
makeMovie = False
directory = "sim_render"

# ----------------------------------------------------------
# Parse command line argments
# ----------------------------------------------------------

ap = argparse.ArgumentParser(description='Train DeepQN RL agent for lane changing decisions')

ap.add_argument("-H", "--hyperparam_config", required=True,
                help=("Path to json file containing the hyperparameters for " +
                      "training the GNN. Must define the following " +
                      "hyperparameters: " + REQ_HYPERPARAM_DESC))

ap.add_argument("-l", "--log_dir", default='../out/runs',
                help=("Directory in which to store logs. Default ../out/runs"))

ap.add_argument("-e", "--num_episodes", type=int, default=100,
                help=("Number of episodes to run simulation for. Default 100"))

ap.add_argument("-d", "--max_dist", type=int, default=500,
                help=("Goal distance for vehicle to travel. Simulation " +
                      "terminates if this is reached. Default 500m"))

ap.add_argument("-t", "--time_step", type=float, default=0.2,
                help=("Simulation time step. Default 0.2s"))

ap.add_argument("-f", "--controller_frequency", type=int, default=5,
                help=("Controller update frequency. Updates at each f " +
                      "timesteps. Default 5"))

ap.add_argument("-s", "--speed_limit", type=int, default=60,
                help=("Highway speed limit (km/h). Default 60"))

ap.add_argument("-N", "--horizon_length", type=int, default=12,
                help=("MPC horizon length. Default 12 "))

ap.add_argument("-T", "--simulation_time", type=int, default=100,
                help=("Maximum total simulation time (s). Default 100"))

ap.add_argument("-E", "--exp_id", default='',
                help=("Optional ID for the experiment"))

args = ap.parse_args()

# -----------------------------------------------------
# validate and parse hyperparameter configuration file

hp_file = args.hyperparam_config
assert os.path.exists(hp_file), ("ERROR: hyperparam config file not found.")
# try loading the json file
try:
    with open(hp_file, 'r', encoding='utf-8') as f:
        hps = json.load(f)
# if the json couldn't be decoded, exit and inform user that config file
# wasn't a valid json file
except json.JSONDecodeError:
    sys.exit("Hyperparameter config file is not a valid json file.")
# cmake sure it contains all required params and nothing else
hyperparams = {}
for hp in REQ_HYPERPARAMS.keys():
    assert (hp in hps), hp + " hyperparameter missing from config file"
    hyperparams[hp] = hps[hp]

# ------------------
# configure logging

log_dir = args.log_dir
assert os.path.exists(log_dir), "Invalid log directory (must already exist)"

# append timestamp to exp ID in case multiple of the same ID is run
exp_id = args.exp_id + '_' + str(time.time())

exp_log_dir = os.path.join(log_dir, exp_id)
if not os.path.exists(exp_log_dir):
    os.makedirs(exp_log_dir)
    print(f"Created experiment log directory at {exp_log_dir}")

# create a directory inside experiment directory to store tensorboard logs
tensorboard_log_dir = os.path.join(exp_log_dir, 'tensorboard_logs')
if not os.path.exists(tensorboard_log_dir):
    os.makedirs(tensorboard_log_dir)
    print(f"Created tensorboard log directory at {tensorboard_log_dir}")

# this is done to allow us to log all script parameters and hyperparameters
# together
args.hyperparam_config = hyperparams

# ----------------------
# System initialization

dt = args.time_step  # Simulation time step (Impacts traffic model accuracy)
f_controller = args.controller_frequency  # Controller update frequency, i.e updates at each t = dt*f_controller
N = args.horizon_length  # MPC Horizon length

ref_vx = args.speed_limit / 3.6  # Highway speed limit in (m/s)

tsim = args.simulation_time  # Maximum total simulation time in seconds

N_episodes = args.num_episodes  # Number of scenarios run created
dist_max = args.max_dist  # Goal distance for the vehicle to travel. If reached, epsiode terminates

# -------------------------- Initilize RL agent object ----------------------------------

RL_Agent = DQNAgent(device, num_node_features, n_actions, ref_vx, **hyperparams)

# -------------------------- Set logging ----------------------------------

# Settings for Tensorboard, which records the training and outputs results
writer = SummaryWriter(log_dir=tensorboard_log_dir)

# save all input arguments to a json log file for reproducibility
param_log_file = os.path.join(exp_log_dir, 'experiment_parameters.json')
with open(param_log_file, 'w', encoding='utf-8') as f:
    json.dump(vars(args), f, ensure_ascii=False, indent=4)
print(f"Saved input parameter log file to {param_log_file}")

# ----------------- Ego Vehicle Dynamics and Controller Settings ------------------------
vehicleADV = vehBicycleKinematic(dt, N)

vehWidth, vehLength, L_tract, L_trail = vehicleADV.getSize()
nx, nu, nrefx, nrefu = vehicleADV.getSystemDim()

# Integrator
int_opt = 'rk'
vehicleADV.integrator(int_opt, dt)
F_x_ADV = vehicleADV.getIntegrator()

# Set Cost parameters
Q_ADV = [0, 40, 3e2, 5, 5]  # State cost, Entries in diagonal matrix
R_ADV = [5, 5]  # Input cost, Entries in diagonal matrix
q_ADV_decision = 50

vehicleADV.cost(Q_ADV, R_ADV)
vehicleADV.costf(Q_ADV)
L_ADV, Lf_ADV = vehicleADV.getCost()

# ------------------ Problem definition ---------------------
scenarioTrailADV = trailing(vehicleADV, N, lanes=3)
scenarioADV = simpleOvertake(vehicleADV, N, lanes=3)
roadMin, roadMax, laneCenters = scenarioADV.getRoad()

# -------------------- Traffic Set up -----------------------
# * Be carful not to initilize an unfeasible scenario where a collsion can not be avoided
# # Initilize ego vehicle
vx_init_ego = 55 / 3.6  # Initial velocity of the ego vehicle
vehicleADV.setInit([0, laneCenters[0]], vx_init_ego)

# # Initilize surrounding traffic
# Lanes [0,1,2] = [Middle,left,right]
vx_ref_init = 50 / 3.6  # (m/s)
advVeh1 = vehicleSUMO(dt, N, [30, laneCenters[1]], [0.9 * vx_ref_init, 0], type="normal")
advVeh2 = vehicleSUMO(dt, N, [45, laneCenters[0]], [0.8 * vx_ref_init, 0], type="passive")
advVeh3 = vehicleSUMO(dt, N, [100, laneCenters[2]], [0.85 * vx_ref_init, 0], type="normal")
advVeh4 = vehicleSUMO(dt, N, [-20, laneCenters[1]], [1.2 * vx_ref_init, 0], type="aggressive")
advVeh5 = vehicleSUMO(dt, N, [40, laneCenters[2]], [1.2 * vx_ref_init, 0], type="aggressive")

# # Combine choosen vehicles in list
vehList = [advVeh1, advVeh2, advVeh3, advVeh4, advVeh5]

# # Define traffic object
leadWidth, leadLength = advVeh1.getSize()
traffic = combinedTraffic(vehList, vehicleADV, ref_vx, f_controller)
traffic.setScenario(scenarioADV)
Nveh = traffic.getDim()

# -----------------------------------------------------------------
# -----------------------------------------------------------------
#      Formulate optimal control problem using opti framework
# -----------------------------------------------------------------
# -----------------------------------------------------------------
dt_MPC = dt * f_controller
# Version = [trailing,leftChange,rightChange]
opts1 = {"version": "leftChange", "solver": "ipopt", "integrator": "rk"}
MPC1 = makeController(vehicleADV, traffic, scenarioADV, N, opts1, dt_MPC)
MPC1.setController()
# MPC1.testSolver(traffic)
changeLeft = MPC1.getFunction()

opts2 = {"version": "rightChange", "solver": "ipopt", "integrator": "rk"}
MPC2 = makeController(vehicleADV, traffic, scenarioADV, N, opts2, dt_MPC)
MPC2.setController()
# MPC2.testSolver(traffic)
changeRight = MPC2.getFunction()

opts3 = {"version": "trailing", "solver": "ipopt", "integrator": "rk"}
MPC3 = makeController(vehicleADV, traffic, scenarioTrailADV, N, opts3, dt_MPC)
MPC3.setController()
trailLead = MPC3.getFunction()

print("Initilization succesful.")

# Initilize Decision maker
decisionMaster = makeDecisionMaster(vehicleADV, traffic, [MPC1, MPC2, MPC3],
                                    [scenarioTrailADV, scenarioADV], RL_Agent)

decisionMaster.setDecisionCost(q_ADV_decision)  # Sets cost of changing decision

# # -----------------------------------------------------------------
# # -----------------------------------------------------------------
# #                         Simulate System
# # -----------------------------------------------------------------
# # -----------------------------------------------------------------

# Constants over all episodes
Nsim = int(tsim / dt)
tspan = np.linspace(0, tsim, Nsim)

refxADV = [0, laneCenters[1], ref_vx, 0, 0]
refxT_in, refxL_in, refxR_in = vehicleADV.setReferences(laneCenters)

refu_in = [0, 0, 0]
refxT_out, refu_out = scenarioADV.getReference(refxT_in, refu_in)
refxL_out, refu_out = scenarioADV.getReference(refxL_in, refu_in)
refxR_out, refu_out = scenarioADV.getReference(refxR_in, refu_in)

refxADV_out, refuADV_out = scenarioADV.getReference(refxADV, refu_in)

# # Store variables
X = np.zeros((nx, Nsim * N_episodes, 1))
U = np.zeros((nu, Nsim * N_episodes, 1))

X_pred = np.zeros((nx, N + 1, Nsim * N_episodes))

X_traffic = np.zeros((4, Nsim * N_episodes, Nveh))
X_traffic_ref = np.zeros((4, Nsim * N_episodes, Nveh))
X_traffic[:, 0, :] = traffic.getStates()

x_lead = DM(Nveh, N + 1)
traffic_state = np.zeros((5, N + 1, Nveh))

feature_map = np.zeros((6, Nsim, Nveh + 1))
i_crit = 0
traffic.reset()

overall_iters = 0

import matplotlib

matplotlib.use('qt5Agg')
matplotlib.pyplot.ion()

# # Episode iteration
for j in range(0, N_episodes):
    print("Episode: ", j + 1)
    # # Initialize simulation
    x_iter = DM(int(nx), 1)
    x_iter[:], u_iter = vehicleADV.getInit()
    vehicleADV.update(x_iter, u_iter)

    eps_iters = 0

    # # Simulation loop
    i = i_crit
    runSimulation = True

    # For the experience replay of the RL agent
    previous_state = None
    action = None
    reward = 0

    while runSimulation:

        # Update feature map for RL agent
        feature_map_i = createFeatureMatrix(vehicleADV, traffic)
        feature_map[:, i:] = feature_map_i

        # Get current traffic state
        x_lead[:, :] = traffic.prediction()[0, :, :].transpose()
        traffic_state[:2, :, ] = traffic.prediction()[:2, :, :]

        # Record the velocity of the vehicle
        writer.add_scalars('Overall/Velocity', {'Velocity': feature_map_i[2][0][0],
                                                'Maximum Velocity': ref_vx}, overall_iters)

        writer.add_scalars('Episode_' + str(j) + '/Velocity', {'Velocity': feature_map_i[2][0][0],
                                                               'Maximum Velocity': ref_vx}, eps_iters)
        # Initialize master controller
        if (i - i_crit) % f_controller == 0:
            # print("----------")
            # print('Step: ', i-i_crit)
            decisionMaster.storeInput([x_iter, refxL_out, refxR_out, refxT_out, refu_out, x_lead, traffic_state])

            fig = plotScene(feature_map_i, scenarioADV, vehicleADV, vehList)
            plt.show()
            plt.pause(0.02)

            # Experience replay for the RL agent, only None if we are at the first iteration
            if previous_state is not None:
                reward /= f_controller
                RL_Agent.store_transition(previous_state, action, reward, feature_map_i, terminal_state=False)

            # Update reference based on current lane
            refxL_out, refxR_out, refxT_out = decisionMaster.updateReference()

            writer.add_scalar('Overall/Rewards', reward, overall_iters)
            writer.add_scalar('Episode_' + str(j) + '/Rewards', reward, eps_iters)

            RL_Agent.learn()

            # Compute optimal control action
            x_test, u_test, X_out, selected_action = decisionMaster.chooseController(feature_map_i)
            u_iter = u_test[:, 0]

            # Update the state for the RL agent, reset the reward to be accumulated
            previous_state, action, reward = feature_map_i, selected_action, 0

        # TODO: We can add the reward to the reward function in this loop
        # This just adds the difference of the velocity with the speed limit onto the reward sum
        reward += min(0, feature_map_i[2][0][0] - ref_vx)

        # Update traffic and store data
        X[:, i] = x_iter
        U[:, i] = u_iter
        X_pred[:, :, i] = X_out
        x_iter = F_x_ADV(x_iter, u_iter)

        try:
            traffic.update()
            vehicleADV.update(x_iter, u_iter)
        except:
            print('Simulation finished: Crash occured')
            break

        # Termination conditions
        if (i - i_crit == Nsim - 1):
            # Simulation did not finish
            runSimulation = False
            print('Simulation finished: Max Iterations')
        elif (x_iter[0].full().item() > dist_max):
            # Simulation finished succesfully
            runSimulation = False
            print('Simulation finished: Max distance reached')
        elif (x_iter[1].full().item() > roadMax) or (x_iter[1].full().item() < roadMin):
            # Truck de-rails from road
            runSimulation = False
            print('Simulation finished: Derail Occured')
        else:
            i += 1

        traffic.tryRespawn(x_iter[0])
        X_traffic[:, i, :] = traffic.getStates()
        X_traffic_ref[:, i, :] = traffic.getReference()

        eps_iters += 1
        overall_iters += 1

    writer.add_scalar('Overall/Episode_Iterations', eps_iters, j)
    writer.add_scalar('Overall/Episode_Distances', x_iter[0].full().item(), j)

    i_crit = i

    # Prepare for next simulation
    traffic.reset()
    X_traffic[:, i, :] = traffic.getStates()
    X_traffic_ref[:, i, :] = traffic.getReference()

print("Simulation finished")

RL_Agent.save_model(os.path.join(exp_log_dir, 'final_model.pt'))

# -----------------------------------------------------------------
# -----------------------------------------------------------------
#                    Plotting and data extraction
# -----------------------------------------------------------------
# -----------------------------------------------------------------

# Creates animation of traffic scenario

# if makeMovie:
#     borvePictures(X, X_traffic, X_traffic_ref, vehList, X_pred, vehicleADV, scenarioADV, traffic, i_crit, f_controller,
#                   directory)
#
# features2CSV(feature_map, Nveh, Nsim)
