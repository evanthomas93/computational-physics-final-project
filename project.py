import numpy as np
import matplotlib.pyplot as plt

# store psi as Nx1 array, full solution stored as NxM (N positions, M timesteps)
m = 1
hBar = 1.055e-34 # Reduced Planck's Constant in J * s
omega = 1
hbar = 1
C_n = [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)] # initial values of first three eigenstates
average_n = (np.dot(C_n,np.arange(1,4)))/3.0 # average energy level, used to get classical turning point 
N = 201
M = 100
T = 10*2*np.pi/omega # run for 10 times the period of the harmonic oscillator
x_classical_turning_point = np.sqrt((2*average_n+1)*(hbar/(m*omega)))
x_max = x_classical_turning_point*5.0
x_vec = np.linspace(-x_max, x_max, N)
t_vec = np.linspace(0,T,M)
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

def dpsi_dt_undamped(psi, x, t):
    # Second derivative of the wavefunction psi
    d2psi_dx2 = np.zeros_like(psi)
    for i in range(1, len(x) - 1):  # Exclude boundaries for derivative calculation
        d2psi_dx2[i] = (psi[i+1] - 2*psi[i] + psi[i-1]) / (h**2)
    
    dpsidt = (-1j / hBar) * (- (hBar**2 / (2 * m)) * d2psi_dx2 + (0.5 * m * omega**2 * x**2) * psi)

    return dpsidt

def dpsi_dt_damped(psi, x, t, alpha):
    d2psi_dx2 = np.zeros_like(psi)
    for i in range(1, len(x) - 1):  
        d2psi_dx2[i] = (psi[i+1] - 2*psi[i] + psi[i-1]) / (h**2)
    
    kinetic_term = (- (hBar**2 / (2 * m)) * d2psi_dx2) * np.exp(-alpha * t) # The kinetic term of the full derivative
    potential_term = (0.5 * m * omega**2 * x**2 * psi) * np.exp(alpha * t) # The potential term of the full derivative

    dpsidt = (-1j / hBar) * (kinetic_term + potential_term)

    return dpsidt

#def time_dependent_solution():
def E_n(n):
    return (n+0.5)*hbar*omega
def eigenstate_solution():
    psi = np.zeros((N,M),np.complex128)
    for i in range(M):
        t = t_vec[i]
        psi[:,i] += np.exp(-1.0j*E_n(0)*t/hbar)*C_n[0]*(m*omega/np.pi/hbar)**0.25*np.exp(-m*omega*x_vec**2.0/(2*hbar))
        psi[:,i] += np.exp(-1.0j*E_n(1)*t/hbar)*C_n[1]*(4/np.pi*(m*omega/hbar)**3)**0.25*x_vec*np.exp(-m*omega*x_vec**2.0/(2*hbar))
        psi[:,i] += np.exp(-1.0j*E_n(2)*t/hbar)*C_n[2]*(m*omega/4/np.pi/hbar)**0.25*(2*m*omega/hbar*x_vec**2.0-1)*np.exp(-m*omega*x_vec**2.0/2/hbar)
    return psi # N x M for N points, M timesteps

# solve time dependent equation for all three timestepping methods
# compare to eigenstate solution

# solve time dependent equation for damped case

# plot