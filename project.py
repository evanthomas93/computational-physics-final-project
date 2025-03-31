import numpy as np
import matplotlib.pyplot as plt

# store psi as Nx1 array, full solution stored as NxM (N positions, M timesteps)
m = 1
hBar = 1.055e-34 # Reduced Planck's Constant in J * s
omega = 1
C_n = [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)] # initial values of first three eigenstates
average_n = (np.dot(C_n,np.arange(1,4)))/3.0 # average energy level, used to get classical turning point 
N = 201
M = 3000
T = 3*2*np.pi/omega # run for 10 times the period of the harmonic oscillator
x_classical_turning_point = np.sqrt((2*average_n+1)*(hBar/(m*omega)))
x_max = x_classical_turning_point*5.0
x_vec = np.linspace(-x_max, x_max, N)
t_vec = np.linspace(0,T,M)
dt = t_vec[1]-t_vec[0]
h =(2*x_max)/(N-1)

def generate_initial_state():
    # get initial psi
    return psi

def euler_step(psi, dpsi_dt_function, t, dt):
    return psi + dpsi_dt_function(psi, x_vec, t)*dt
def runge_kutta_step(psi, dpsi_dt_function, t, dt):
    k1 = dt * dpsi_dt_function(psi, x_vec, t)
    k2 = dt * dpsi_dt_function((psi + (0.5 * k1)), x_vec, (t + (0.5 * dt)))
    k3 = dt * dpsi_dt_function((psi + (0.5 * k2)), x_vec, (t + (0.5 * dt)))
    k4 = dt * dpsi_dt_function((psi + k3), x_vec, (t + dt))

    psi_next = psi + ((k1 + (2 * k2) + (2 * k3) + k4) / 6)

    return psi_next

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
    return (n+0.5)*hBar*omega
def eigenstate_solution():
    psi = np.zeros((N,M),np.complex128)
    for i in range(M):
        t = t_vec[i]
        psi[:,i] += np.exp(-1.0j*E_n(0)*t/hBar)*C_n[0]*(m*omega/np.pi/hBar)**0.25*np.exp(-m*omega*x_vec**2.0/(2*hBar))
        psi[:,i] += np.exp(-1.0j*E_n(1)*t/hBar)*C_n[1]*(4/np.pi*(m*omega/hBar)**3)**0.25*x_vec*np.exp(-m*omega*x_vec**2.0/(2*hBar))
        psi[:,i] += np.exp(-1.0j*E_n(2)*t/hBar)*C_n[2]*(m*omega/4/np.pi/hBar)**0.25*(2*m*omega/hBar*x_vec**2.0-1)*np.exp(-m*omega*x_vec**2.0/2/hBar)
    return psi # N x M for N points, M timesteps

psi_eigenstates = eigenstate_solution()

psi_undamped_euler = np.zeros_like(psi_eigenstates, np.complex128)
psi_undamped_euler[:,0] = psi_eigenstates[:,0]
psi_undamped_rk4 = psi_undamped_euler.copy()
for i in range(M):
    print(i)
    if i > 0:
        psi_undamped_euler[0,i-1] = 0.0
        psi_undamped_euler[-1,i-1] = 0.0
        psi_undamped_rk4[0,i-1] = 0.0
        psi_undamped_rk4[-1,i-1] = 0.0
        psi_undamped_euler[:,i] = euler_step(psi_undamped_euler[:,i-1], dpsi_dt_undamped, t_vec[i], dt)
        psi_undamped_rk4[:,i] = runge_kutta_step(psi_undamped_rk4[:,i-1], dpsi_dt_undamped, t_vec[i], dt)
    #plt.plot(np.real(psi_undamped_euler[:,i]))
    #plt.plot(np.imag(psi_undamped_euler[:,i]))
    #plt.plot(np.abs(psi_undamped_rk4[:,i])**2)
    #plt.xlabel("x (m)")
    #plt.ylabel("$|\Psi|^2$")
    #plt.ylim(0, 8e16)
    #plt.show(block=False)
    #plt.savefig("euler/{}.png".format(i))
    #plt.clf()

# solve time dependent equation for all three timestepping methods
# compare to eigenstate solution

# solve time dependent equation for damped case

# plot