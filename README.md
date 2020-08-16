## Trajectory Path Prediction [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Our proposed model predicts trajectory paths of novel compounds using RNN-based algorithms.

The presented work demonstrates the training of recurrent neural networks (RNNs) from distributions of atom coordinates in solid state structures that were obtained using ab initio molecular dynamics (AIMD) simulations. AIMD simulations on solid state structures are treated as a multi-variate time-series problem. By referring interactions between atoms over the simulation time to temporary correlations among them, RNNs find patterns in the multi-variate time-dependent data, which enable forecasting trajectory paths and potential energy profiles. Two types of RNNs, namely gated recurrent unit and long short-term memory networks, are considered. The model is described and compared against a baseline AIMD simulation. Findings demonstrate that both networks can potentially be harnessed for accelerated statistical sampling in computational materials research.

### Results

*Predictions of the trajectory of the an atom in the studied compound and total potential energy by (a) GRU and (b) LSTM compared to AIMD baseline:*
<img src="https://raw.githubusercontent.com/Mehrdad93/mehrdad93.github.io/master/images/predict.png" width="200" height="400" />

*Density plot of coordinates of (a) all atoms (b) randomly chosen atom i, over the AIMD simulation run:*
<img src="https://raw.githubusercontent.com/Mehrdad93/mehrdad93.github.io/master/images/Density.png" width="200" height="400" />

### Contributors
> Mehrdad Mokhtari;
> ‪Mohammad J. Eslamibidgoli‬;
> Michael H. Eikerling

More details can be found in the following file:
> Trajectory_prediction_RNN.pdf
