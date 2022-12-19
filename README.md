# Machine Learning Project 2: Autonomous Lane Changing using Deep Reinforcement Learning with Graph Neural Networks

![](https://github.com/BorveErik/Autonomous-Truck-Sim/blob/RL_training_env/simRes.gif)

**Authors:** Arvind Satish Menon, Lars C.P.M. Quaedvlieg, and Somesh Mehra

**Supervisor:** [Erik Börve](mailto://borerik@chalmers.se) 

**Group:** GAN_CONTROL

## Important Links
- Final report: https://www.overleaf.com/read/sbtzctmpbqpy
- In-depth project description: https://docs.google.com/document/d/1_oW5013IwnaLW3alfvVjh7CEu5wkTPQh/edit

## Project introduction

In recent years, autonomous vehicles have garnered significant attention due to their potential to improve the safety, 
efficiency, and accessibility of transportation. One important aspect of autonomous driving is the ability to make 
lane-changing decisions, which requires the vehicle to predict the intentions and behaviors of other road users and to 
evaluate the safety and feasibility of different actions. 

In this project, we propose a graph neural network (GNN) architecture combined with reinforcement learning (RL) and a model 
predictive controller to solve the problem of autonomous lane changing. By using GNNs, it is possible to learn a control 
policy that takes into account the complex and dynamic relationships between the vehicles, rather than just considering 
local features or patterns. More specifically, we employ Deep Q-learning with Graph Attention Networks for the agent.

**Please note that not all the code is the work of this project group**. We will use a basis provided by our supervisor
to build upon. For an idea of this basis, please utilize [this repository](https://github.com/BorveErik/Autonomous-Truck-Sim).
However, we will also mention our contributions to every individual file further down this README.

From the repository linked above:

>  This project provides an implementation of an autonomous truck in a multi-lane highway scenario. The controller utilizes
 non-linear optimal control to compute multiple feasible trajectories, of which the most cost-efficent is choosen. The 
 simulation is set up to be suitable for RL training.

## Getting Started

Clone the project in your local machine.

### Requirements

Locate the repository and run:
  ```sh
  pip install -r requirements.txt
  ```

**Additionally**, you need to install Pytorch and Pytorch Geometric separately. please note that for installing 
[Pytorch](https://pytorch.org/get-started/locally/) and [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html),
please refer to the references installation guides for CPU or GPU and ensure that a *compatible version* of Pytorch has 
been installed for Pytorch Geometric.

| Package             | Use                         |
|---------------------|-----------------------------|
| Pytorch (Geometric) | Automatic Differentiation   |
| Networkx            | Graph representation        |
| Casadi              | Nonlinear optimization      |
| Numpy               | Numerical computations      |
| Matplotlib          | Plotting and visualizations |
| PyQt5               | Plotting and visualizations |
| Tensorboard         | Experiment visualizations   |

### Usage

The RL model can be trained using a certain configuration (see format below) via the "main.py" file. This is also where 
simulations are configured, including e.g., designing traffic scenarios and setting up the optimal controllers.

Next, the "inference.py" file allows you to perform inference using certain configurations of the environment and a
trained agent model.


## Structure of the repository

    .
    ├── out               # Any output files generated from the code
    ├── res               # Any resources that can be or are used in the code (e.g. configurations)
    ├── src               # Source code of the project
    ├── .gitignore        # GitHub configuration settings
    ├── README.md         # Description of the repository
    └── requirements.txt  # Python packages to install before running the code

### Out
    .
    ├── out                   
    │   ├── models  # Any Pytorch models to store after training, which can then be loaded later
    │   └── runs    # Files generated by TensorBoard to visualize the experiments and inference
    └── ...

> This is the directory in which all the output files generated from the code are stored

### Res
    .
    ├── ...
    ├── res                   
    │   ├── model_hyperparams             # Directory storing a configuration for the RL model hyperparameters
    │   │   └── example_hyperparams.json  # Example configuration file for the hyperparameters
    │   ├── ex2.csv                       # Example file with raw data generated in the simulation (only used for intuition)
    │   ├── metaData_ex2.txt              # Metadata for ex2.csv
    │   └── simRes.gif                    # Example GIF of one episode in the simulation
    └── ...

> This is the directory where any resources that can be or are used in the code (e.g. configurations)

### Src
    .
    ├── ...
    ├── src  
    │   ├── agents                 # The package created for storing anything related to the RL agent
    │   │   ├── __init__.py        # Creates the Python package
    │   │   ├── graphs.py          # File containing class for creating the graph datastructures
    │   │   └── rlagent.py         # File containing anything from the network architecture to the Deep Q-learning agent
    │   ├── controllers.py         # Generates optimal controller based on specified scenario, and optimizes the trajectory choice, returns optimal policy
    │   ├── helpers.py             # Contains assisting functions, e.g., for data extraction and plotting
    │   ├── inference.py           # Performing inference with a trained RL agent
    │   ├── main.py                # Setting up and running simulations to train the RL agent
    │   ├── scenarios.py           # Formulates constraints for the different scenarios considered in the optimal controllers
    │   ├── traffic.py             # Combined traffic: Used to communicate with all vehicles in traffic scenario, creates a vehicle with specified starting position, velocity and class.
    │   └── vehicleModelGarage.py  # Contains truck models that can be utilized in the simulation
    └── ...

> This is the directory that contains all the source code for this project

### Arguments for executable files

@TODO: Somesh
@TODO: Could you also mention how you can reproduce the experiments with the seed? And which seed we used I guess

#### main.py

```bash
python main.py -H ...
```

#### inference.py

```bash
python inference.py -H ...
```


### Who did what?

Since we used a code template from our supervisor to build upon, it is important to note who made the contributions (our
group or our supervisor).

| File                  | Who                        |
|-----------------------|----------------------------|
| graph.py              | Project group              |
| rlagent.py            | Project group              |
| controllers.py        | Supervisor                 |
| helpers.py            | Supervisor                 |
| inference.py          | Project group              |
| main.py               | Project group & Supervisor |
| scenarios.py          | Supervisor                 |
| traffic.py            | Supervisor                 |
| vehicleModelGarage.py | Supervisor                 |

In main.py, our supervisor made everything that has to do with the raw simulation and the controller. On top of that, we
added everything to do with the RL Agent, the argument parsing (thus the models, hyperparameters and configurations), and
TensorBoard.