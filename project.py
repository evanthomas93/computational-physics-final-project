import numpy as np
import matplotlib.pyplot as plt

# store psi as Nx1 array, full solution stored as NxM (N positions, M timesteps)
m = 1
omega = 1
hbar = 1
C_n = [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)] # initial values of first three eigenstates
average_n = (np.dot(C_n,np.arange(1,4)))/3.0 # average energy level, used to get classical turning point 
N = 201
M = 100
x_classical_turning_point = np.sqrt((2*average_n+1)*(hbar/(m*omega)))
x_max = x_classical_turning_point*5.0
x_vec = np.linspace(-x_max, x_max, N)
h =(2*x_max)/(N-1)

def generate_initial_state():
    # get initial psi
    return psi

def euler_step(psi, dpsi_dt_function, dt):
    pass
def runge_kutta_step(psi, dpsi_dt_function, dt):
    pass
def crank_nicholson_step(psi, dpsi_dt_function, dt):
    pass
    # might need a separate version for damped and undamped

# V(x) of harmonic potential
def harmonic_potential(x):
    pass

def dpsi_dt_undamped(psi, x, t):
    pass
    
def dpsi_dt_damped(psi, x, t, alpha):
    pass

#def time_dependent_solution():

def eigenstate_solution():
    psi = np.zeros((N,M))
    for i in range(N):
        psi[:,0] += C_n[0]*(m*omega/np.pi/hbar)**0.25*np.exp(-m*omega*x_vec**2.0/(2*hbar))
        psi[:,0] += C_n[1]*(4/np.pi*(m*omega/hbar)**3)**0.25*x_vec*np.exp(-m*omega*x_vec**2.0/(2*hbar))
        psi[:,0] += C_n[2]*(m*omega/4/np.pi/hbar)**0.25*(2*m*omega/hbar*x_vec**2.0-1)*np.exp(-m*omega*x_vec**2.0/2/hbar)
    return psi # N x M for N points, M timesteps

# solve time dependent equation for all three timestepping methods
# compare to eigenstate solution

# solve time dependent equation for damped case

# plot