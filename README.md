# 3D PCA modeling in python

## Duffing Model in 3-Dimensional State
This is a python version based on the 3D duffing repo from victormeloasm at https://github.com/victormeloasm/3DDuffingOscillator

## Lorenz Model

## Neural ODEs
This uses a neural network to solve. This requires torch to work as we are training an NN, but shouldn't be too costly and also utilizes the torchdiffeq package for solving: https://github.com/electro-phys/torchdiffeq

### Vector Field
This uses the output of the neural ODE, howeverm you can specify any equation that you want, but this requiries manually adding it to the code.
