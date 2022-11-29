# Packages
import torch
from casadi import *
import numpy as np

# Classes and helpers
from vehicleModelGarage import vehBicycleKinematic
from scenarios import trailing, simpleOvertake
from traffic import vehicleSUMO, combinedTraffic
from controllers import makeController, makeDecisionMaster
from helpers import *

from agents.templateRLagent import DQNAgent

# Set Gif-generation
makeMovie = False
directory = r"C:\Phd\Student_Projects\GNN_RL_EPFL\Latest_code_local\simRes.gif"

# System initialization 
dt = 0.2  # Simulation time step (Impacts traffic model accuracy)
f_controller = 5  # Controller update frequency, i.e updates at each t = dt*f_controller
N = 12  # MPC Horizon length

ref_vx = 60 / 3.6  # Highway speed limit in (m/s)

# -------------------------- Initilize RL agent object ----------------------------------
# The agent is feed to the decision maker, changing names requries changing troughout code base
N_episodes = 10  # Number of scenarios run created
dist_max = 500  # Goal distance for the vehicle to travel. If reached, epsiode terminates

# Settings for the RL agent
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_node_features = 6
n_actions = 3
gamma = 0.9
target_copy_delay = 0
learning_rate = 10E-3
batch_size = 32
epsilon = 0.01

RL_Agent = DQNAgent(device, num_node_features, n_actions, gamma, target_copy_delay, learning_rate, batch_size, epsilon)

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
tsim = 100  # Maximum total simulation time in seconds
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

# # Episode iteration
for j in range(0, N_episodes):
    print("Episode: ", j + 1)
    # # Initialize simulation
    x_iter = DM(int(nx), 1)
    x_iter[:], u_iter = vehicleADV.getInit()
    vehicleADV.update(x_iter, u_iter)

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

        # Initialize master controller
        if (i - i_crit) % f_controller == 0:
            # print("----------")
            # print('Step: ', i-i_crit)
            decisionMaster.storeInput([x_iter, refxL_out, refxR_out, refxT_out, refu_out, x_lead, traffic_state])

            # Experience replay for the RL agent, only None if we are at the first iteration
            if previous_state is not None:
                RL_Agent.store_transition(previous_state, action, reward, feature_map_i, terminal_state=False)

            # Update reference based on current lane
            refxL_out, refxR_out, refxT_out = decisionMaster.updateReference()

            RL_Agent.learn()

            # Compute optimal control action
            x_test, u_test, X_out, selected_action = decisionMaster.chooseController(feature_map_i)
            u_iter = u_test[:, 0]

            # Update the state for the RL agent, reset the reward to be accumulated
            previous_state, action, reward = feature_map_i, selected_action, 0

        # TODO: We can add the reward to the reward function in this loop
        # This example just add a reward of 1 at every iteration
        reward += 1

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
            print('Simulation finished: Crash Occured')
        else:
            i += 1

        traffic.tryRespawn(x_iter[0])
        X_traffic[:, i, :] = traffic.getStates()
        X_traffic_ref[:, i, :] = traffic.getReference()

    i_crit = i
    # Prepare for next simulation
    traffic.reset()
    X_traffic[:, i, :] = traffic.getStates()
    X_traffic_ref[:, i, :] = traffic.getReference()

print("Simulation finished")

# -----------------------------------------------------------------
# -----------------------------------------------------------------
#                    Plotting and data extraction
# -----------------------------------------------------------------
# -----------------------------------------------------------------

# Creates animation of traffic scenario

if makeMovie:
    borvePictures(X, X_traffic, X_traffic_ref, vehList, X_pred, vehicleADV, scenarioADV, traffic, i_crit, f_controller,
                  directory)

features2CSV(feature_map, Nveh, Nsim)
