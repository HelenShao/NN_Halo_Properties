# NN_Halo_Properties
In conjunction with the autoencoder project, using neural networks to determine which halo properties can be expressed
in terms of others. This will help reduce the input parameters for the symbolic regression. 

Similar to training an autoencoder, this uses a dynamic model that is optimized by optuna (n_layers, n_neurons per layer, 
lr, wd)

Important Features:
1. Saliency Maps - determine which of the input properties contribute most to the final output from model
2. Model Predictions vs True Inputs - for each property, there is a model prediction based on the other <11 input properties 
