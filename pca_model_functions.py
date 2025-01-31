import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint as torch_odeint
from sklearn.decomposition import PCA

def duffing_oscillation_test():
    # Parameters
    amp = 0.42
    b = 0.5
    alpha = -1.0
    beta = 1.0
    w = 1.0

    # Time step and initial condition
    tspan = np.arange(0, 500, 0.1)
    x10 = 0.5021
    x20 = 0.17606
    y0 = [x10, x20]

    # Duffing oscillator equations
    def f(y, t, b, alpha, beta, amp, w):
        x1, x2 = y
        dx1 = x2
        dx2 = -b * x2 - alpha * x1 - beta * x1**3 + amp * np.sin(w * t)
        return [dx1, dx2]

    # Solve the Duffing oscillator
    y = odeint(f, y0, tspan, args=(b, alpha, beta, amp, w))
    x1 = y[:, 0]
    x2 = y[:, 1]

    # Create a colored 3D trajectory using lines
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colormap = plt.cm.jet(np.linspace(0, 1, len(tspan)))
    
    for i in range(len(tspan) - 1):
        ax.plot(x1[i:i+2], x2[i:i+2], tspan[i:i+2], color=colormap[i], linewidth=1.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Time')
    plt.title('Colored Duffing Oscillator Trajectory')
    plt.colorbar(plt.cm.ScalarMappable(cmap='jet'), ax=ax)
    plt.show()
    return


def lorenz_attractor_test():
    # Parameters
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    # Time step and initial condition
    tspan = np.arange(0, 50, 0.01)
    x10 = 1.0
    x20 = 1.0
    x30 = 1.0
    y0 = [x10, x20, x30]

    # Lorenz attractor equations
    def f(y, t, sigma, rho, beta):
        x1, x2, x3 = y
        dx1 = sigma * (x2 - x1)
        dx2 = x1 * (rho - x3) - x2
        dx3 = x1 * x2 - beta * x3
        return [dx1, dx2, dx3]

    # Solve the Lorenz attractor
    y = odeint(f, y0, tspan, args=(sigma, rho, beta))
    x1 = y[:, 0]
    x2 = y[:, 1]
    x3 = y[:, 2]

    # Create a colored 3D trajectory using lines
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colormap = plt.cm.jet(np.linspace(0, 1, len(tspan)))
    
    for i in range(len(tspan) - 1):
        ax.plot(x1[i:i+2], x2[i:i+2], x3[i:i+2], color=colormap[i], linewidth=1.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Colored Lorenz Attractor Trajectory')
    plt.colorbar(plt.cm.ScalarMappable(cmap='jet'), ax=ax)
    plt.show()
    return


def duffing_oscillation_pca(pca_data):
    # Parameters
    amp = 0.42
    b = 0.5
    alpha = -1.0
    beta = 1.0
    w = 1.0

    # Time step and initial condition
    tspan = np.arange(0, len(pca_data), 1)
    x10, x20, x30 = pca_data[0]
    y0 = [x10, x20, x30]

    # Duffing oscillator equations adapted for 3D PCA
    def f(y, t, b, alpha, beta, amp, w):
        x1, x2, x3 = y
        dx1 = x2
        dx2 = -b * x2 - alpha * x1 - beta * x1**3 + amp * np.sin(w * t)
        dx3 = -x3 + x1 * x2  # Adding a third dimension
        return [dx1, dx2, dx3]

    # Solve the Duffing oscillator
    y = odeint(f, y0, tspan, args=(b, alpha, beta, amp, w))
    x1 = y[:, 0]
    x2 = y[:, 1]
    x3 = y[:, 2]

    # Create a colored 3D trajectory using lines
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colormap = plt.cm.jet(np.linspace(0, 1, len(tspan)))
    
    for i in range(len(tspan) - 1):
        ax.plot(x1[i:i+2], x2[i:i+2], x3[i:i+2], color=colormap[i], linewidth=1.5)
    
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    plt.title('Colored Duffing Oscillator Trajectory on PCA Data')
    plt.colorbar(plt.cm.ScalarMappable(cmap='jet'), ax=ax)
    plt.show()
    return

def lorenz_attractor_pca(pca_data):

    # Parameters for the Lorenz attractor
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    # Time step and initial condition based on the PCA data
    tspan = np.arange(0, len(pca_data), 0.1)
    x10, x20, x30 = pca_data[0]
    y0 = [x10, x20, x30]

    # Lorenz attractor equations
    def f(y, t, sigma, rho, beta):
        x1, x2, x3 = y
        dx1 = sigma * (x2 - x1)
        dx2 = x1 * (rho - x3) - x2
        dx3 = x1 * x2 - beta * x3
        return [dx1, dx2, dx3]

    # Solve the Lorenz attractor
    y = odeint(f, y0, tspan, args=(sigma, rho, beta))
    x1 = y[:, 0]
    x2 = y[:, 1]
    x3 = y[:, 2]

    # Create a colored 3D trajectory using lines
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colormap = plt.cm.jet(np.linspace(0, 1, len(tspan)))
    
    for i in range(len(tspan) - 1):
        ax.plot(x1[i:i+2], x2[i:i+2], x3[i:i+2], color=colormap[i], linewidth=1.5)
    
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    plt.title('Colored Lorenz Attractor Trajectory on PCA Data')
    plt.colorbar(plt.cm.ScalarMappable(cmap='jet'), ax=ax)
    plt.show()
    return



class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 50),
            nn.Tanh(),
            nn.Linear(50, 3)
        )

    def forward(self, t, y):
        return self.net(y)

def neural_ode_pca(pca_data, epochs=1000, lr=0.01):
    # Convert PCA data to PyTorch tensors
    pca_data = torch.tensor(pca_data, dtype=torch.float32)

    # Define initial condition and time points
    y0 = pca_data[0]
    t = torch.linspace(0, 1, pca_data.shape[0])

    # Initialize the ODE function and optimizer
    ode_func = ODEFunc()
    optimizer = optim.Adam(ode_func.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_y = torch_odeint(ode_func, y0, t)
        loss = torch.mean((pred_y - pca_data) ** 2)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    # Plot the results
    pred_y = pred_y.detach().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pred_y[:, 0], pred_y[:, 1], pred_y[:, 2], label='Predicted')
    ax.plot(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], label='Actual')
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    plt.legend()
    plt.show()
    return ode_func


def generate_vector_field(ode_func, grid_size=20, pca_range=(150, 150)):
    x = np.linspace(pca_range[0], pca_range[1], grid_size)
    y = np.linspace(pca_range[0], pca_range[1], grid_size)
    z = np.linspace(pca_range[0], pca_range[1], grid_size)
    X, Y, Z = np.meshgrid(x, y, z)

    U, V, W = np.zeros(X.shape), np.zeros(Y.shape), np.zeros(Z.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                point = torch.tensor([X[i, j, k], Y[i, j, k], Z[i, j, k]], dtype=torch.float32)
                vector = ode_func(0, point).detach().numpy()
                U[i, j, k], V[i, j, k], W[i, j, k] = vector

    return X, Y, Z, U, V, W

from scipy.optimize import fsolve
def find_fixed_points(ode_func, initial_guesses):
    def equations(state):
        return ode_func(0, torch.tensor(state, dtype=torch.float32)).detach().numpy()

    fixed_points = [fsolve(equations, guess) for guess in initial_guesses]
    return fixed_points

def generate_vector_field_2d(ode_func, grid_size=20, pca_range=(-20, 20)):
    x = np.linspace(pca_range[0], pca_range[1], grid_size)
    y = np.linspace(pca_range[0], pca_range[1], grid_size)
    X, Y = np.meshgrid(x, y)

    U, V = np.zeros(X.shape), np.zeros(Y.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = torch.tensor([X[i, j], Y[i, j]], dtype=torch.float32)
            vector = ode_func(0, point).detach().numpy()
            U[i, j], V[i, j] = vector

    return X, Y, U, V

def jacobian(ode_func, state):
    state = torch.tensor(state, dtype=torch.float32)
    state.requires_grad_(True)
    vector = ode_func(0, state)
    jacobian_matrix = torch.autograd.functional.jacobian(lambda s: ode_func(0, s), state)
    return jacobian_matrix.detach().numpy()

def check_stability(ode_func, fixed_points):
    stability = []
    for fp in fixed_points:
        jacobian_matrix = jacobian(ode_func, fp)
        eigvals = np.linalg.eigvals(jacobian_matrix)
        stability.append(('Stable' if np.all(eigvals < 0) else 'Unstable', eigvals))
    return stability