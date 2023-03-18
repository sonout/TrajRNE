<!-- ABOUT THE PROJECT -->
<!-- ## About The Project -->
# TrajRNE
The source code for the paper: “Road Representation Learning with Vehicle Trajectories” accepted in PAKDD 2023 by Stefan Schestakov, Paul Heinemeyer and Elena Demidova.

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
To install the required packages we recommend using [Conda](https://docs.conda.io/en/latest/). Our used environment can be easily installed with conda.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/...
   ```
2. Install conda environment
   ```sh
   conda config --env --set channel_priority strict
   conda env create -f environment.yml
   ```
3. Activate the environment
   ```sh
   conda activate road-emb
   ```

<!-- USAGE EXAMPLES -->


## Usage

1. Generate road network with 
   ```sh
   python generate_road_network.py
   ```
2. Generate training data with
   ```sh
   python generate_train_data.py
   ```
3. Train Model with 
   ```sh
   python train.py
   ```
   The final Road Embeddings will be saved in the data/ folder. 





