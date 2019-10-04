# Trajectory-path-prediction
Predicting trajectory paths of novel compounds employing RNN-based models

The presented work demonstrates the training of recurrent neural networks (RNNs) from distributions of atom coordinates in solid state structures that were obtained using ab initio molecular dynamics (AIMD) simulations. AIMD simulations on solid state structures are treated as a multi-variate time-series problem. By referring interactions between atoms over the simulation time to temporary correlations among them, RNNs find patterns in the multi-variate time-dependent data, which enable forecasting trajectory paths and potential energy profiles. Two types of RNNs, namely gated recurrent unit and long short-term memory networks, are considered. The model is described and compared against a baseline AIMD simulation. Findings demonstrate that both networks can potentially be harnessed for accelerated statistical sampling in computational materials research.

More details can be found in the following file:
> 
