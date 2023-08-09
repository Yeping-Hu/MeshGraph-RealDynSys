# Graph Learning in Physical-informed Mesh-reduced Space for Real-world Dynamic Systems

This Git repository contains codes for the 'Graph Learning in Physical-informed Mesh-reduced Space for Real-world Dynamic Systems' paper that is published in 2023 SIGKDD. The goal of this work is to reduce computational costs of learning-based simulator while preseving crucial dynamic properties. Specifically, this work uses physical-informed prior (PiP) information to learn and predict dynamic systems in a reduced mesh space. A two-stage graph-based model is proposed. In the first stage, we learn a subgraph autoencoder to summarize the information in a mesh-reduced space using physical-informed priors. In the second stage, we learn a dynamics predictor to predict subgraph evolution. We demonstrate the effectiveness of our model on two fluid flow datasets: lid-driven cavity flow data and cylinder flow data. This code is also applicable for other dynamic systems by modifying the PiP module, along with corresponding data.

Authors: [Yeping Hu](https://yeping-hu.github.io/){:target="_blank"} (hu25@llnl.gov), Bo Lei(lei4@llnl.gov), Victor M. Castillo(castillo3@llnl.gov)

Affiliation: Lawrence Livermore National Laboratory, Livermore, CA, USA

## Table of Contents
- [1. Setup](#1-setup)
- [2. Datasets](#2-datasets)
  - [2.1 Lid-driven Cavity Flow Data](#21-lid-driven-cavity-flow-data)
  - [2.2 Cylinder Flow Data](#22-cylinder-flow-data)
  - [2.3 Out-of-Distribution Cylinder Flow Data](#23-out-of-distribution-cylinder-flow-data)
  - [2.4 Customized Data](#24-customized-data)
- [3. Training Mesh Graph Autoencoder](#3-training-mesh-graph-autoencoder)
  - [3.1 Lid-driven Cavity Flow](#31-lid-driven-cavity-flow)
  - [3.2 Cylinder Flow](#32-cylinder-flow)
- [4. Training Graph Predictor at Mesh-reduced Space](#4-training-graph-predictor-at-mesh-reduced-space)
  - [4.1 Full Graph](#41-full-graph)
  - [4.2 Subgraph](#42-subgraph)
- [5. Evaluation](#5-evaluation)

## 1. Setup
Set up a virtual environment and install the required packages:
```shell
pip install -r requirements.txt
```
## 2. Datasets
### 2.1 Lid-driven Cavity Flow Data
Our lid-driven cavity flow dataset is generated using OpenFOAM. The dataset is stored in 
`./data/cavity_flow_all.pkl`. The pickle file is a python Dictionary which stores the generated 
cases with different settings. The Data structure is:
```
{
  'case_1': {
    'properties': specs of this case (U: lid velocity, Re: Reynolds number)
    'mesh_pos': [N_node x 2] array that stores 2D location of each node,
    'velocity': [N_node x 2] array that stores velocity at each node in xy direction,
    'node_type': [N_node x 1] array that stores the type of each node
    'cells': [N_mesh x 3] array that stores the index of all triangular meshes,
    'vortex': a dictionary that stores all vortex center in this case and its size
  },
  'case_2': {...},
}
```

### 2.2 Cylinder Flow Data
The cylinder flow dataset we used to develop our model is provided by DeepMind. Download the dataset:
```shell
bash ./data/download_dataset.sh cylinder_flow ./data
```
The dataset is stored in `./data/cylinder_flow/` with `.tfrecord` format. We convert the data to 
`.pkl` format for customized trajectory length and $\Delta$t. The original data consists of 1000 
training, 100 validation and 100 test trajectories with 600 time steps. The following command converts the data and produces the trajectories 
ending at 300 time steps with time interval of 1: 
```shell
python dataset.py --end 300 --interval 1
```
Each pickle file is a python List with each element being a dictionary storing the data of a trajectory.
The data structure for each trajectory is:
```
{
  'mesh_pos': [N_node x 2] array that stores 2D location of each node,
  'velocity': [N_timestep x N_node x 2] array that stores velocity at each node and time step in xy direction,
  'node_type': [N_node x 1] array that stores the type of each node
  'cells': [N_mesh x 3] array that stores the index of all triangular meshes,
}
```

### 2.3 Out-of-Distribution Cylinder Flow Data
To analyze model generalizability, we use OpenFOAM to simulate unseen test cases with horizontal and vertical placement
of two cylinders. The data can be found at `./data/cfd_two_cylinder.pkl`. The data structure is the same as the
regular cylinder flow data.

### 2.4 Customized Data
For customized data, make sure the data is stored in the same format as the above datasets.
Specifically, each entry should be a dictionary with the following keys: `mesh_pos`, `velocity`, `node_type`, `cells`.
To load train/test datasets, please follow `load_cavity_data` and `load_cfd_traj` in `./core/dataset.py` as examples.

---

## 3. Training Mesh Graph Autoencoder

In this first stage, an autoencoder is learned to summarize information in a mesh-reduced space.
Configurations and hyperparameters are stored in `.yaml` file under the directory of `configs/cavity` and
`configs/cylinder_recon`.

### 3.1 Lid-driven Cavity Flow
Lid-driven cavity flow data has highly varying mesh structures and we select half of the nodes in the mesh to form the 
mesh-reduced space. The configuration using PiP MeshGraphSelector and TIN MeshGraphReverser is `chybrid_r05.yaml`. 
Training script:
```
python train_cavity.py  --config chybrid_r05.yaml
```
GMR_GMUS is the baseline model using a random MeshGraphSelector. The configuration is `crandom_r05.yaml`.


### 3.2 Cylinder Flow
For cylinder flow data, we select 512 nodes in the mesh to form the mesh-reduced space. The configuration using 
PiP MeshGraphSelector and TIN MeshGraphReverser is `300_hybrid512_bc.yaml`. Training script:
```
python train_cylinder_recon.py  --config 300_hybrid512_bc.yaml --mode train
```
The configuration for GMR_GMUS is `300_random512_bc.yaml`.

## 4. Training Graph Predictor at Mesh-reduced Space
In this stage, we train a dynamics predictor to predict the evolution of the flow field.

### 4.1 Full Graph
The full graph dynamics predictor is a single-stage MeshGraphNet model similar to the model used 
in the DeepMind paper. The configuration is 
`./configs/cylinder_rollout/300fg_e15_NI5e-3.yaml`.
To train a full graph dynamics predictor, run the following command:
```
python train_rollout_fg.py --config 300fg_e15_NI5e-3.yaml
```
### 4.2 Subgraph

The configuration for training a 300-step subgraph dynamics predictor is `./configs/cylinder_sg/300_h512z4_dnet.yaml`. 
It is build on the subgraph autoencoder trained using the configuration `300_hybrid512_bc.yaml`.
It follows the following steps:
- Train the subgraph autoencoder (done in the previous step)
- Obtain latent representations z of the training data and the indices of the selected nodes

```shell
python train_cylinder_recon.py --config 300_hybrid512_bc.yaml --mode save_z
```
- Connect the subgraph using the selected nodes using Delaunay triangulation (MeshGraphConnect)

```shell
python train_cylinder_recon.py --config 300_hybrid512_bc.yaml --mode connect
```

- Train the z predictor using the latent representations z and the subgraph

```shell
python train_rollout_sg.py --config 300_h512z4_dnet.yaml --mode train
```
The configuration for GMR_GMUS is `300_r512z4_dnet.yaml`.

## 5. Evaluation
The test result can be generated running training python scripts with `--mode test`. Reconstructed or predicted 
velocity fields are saved accordingly. To run test on out-of-distribution cylinder flow data, add flag `--ood`
in command. The functionalities in `core/metric.py` can be used to compute
vortex-related metrics including VDR (vortex detection rate) and MDDV (mean distance to detected vortices).

##
LLNL-CODE-852129



