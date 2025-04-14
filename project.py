import numpy as np
import matplotlib.pyplot as plt

# store psi as Nx1 array, full solution stored as NxM (N positions, M timesteps)
m = 1
hBar = 1.055e-34 # Reduced Planck's Constant in J * s
omega = 1
C_n = [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)] # initial values of first three eigenstates
average_n = (np.dot(C_n,np.arange(1,4)))/3.0 # average energy level, used to get classical turning point 
N = 201
M = 10000
T = 10*2*np.pi/omega # run for 10 times the period of the harmonic oscillator
x_classical_turning_point = np.sqrt((2*average_n+1)*(hBar/(m*omega)))
x_max = x_classical_turning_point*5.0
x_vec = np.linspace(-x_max, x_max, N)
t_vec = np.linspace(0,T,M)
dt = t_vec[1]-t_vec[0]
h =(2*x_max)/(N-1)
alpha = 0.1

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

def dpsi_dt_damped(psi, x, t):
    d2psi_dx2 = np.zeros_like(psi)
    for i in range(1, len(x) - 1):  
        d2psi_dx2[i] = (psi[i+1] - 2*psi[i] + psi[i-1]) / (h**2)
    
    kinetic_term = (- (hBar**2 / (2 * m)) * d2psi_dx2) * np.exp(-alpha * t) # The kinetic term of the full derivative
    potential_term = (0.5 * m * omega**2 * x**2 * psi) * np.exp(alpha * t) # The potential term of the full derivative

    dpsidt = (-1j / hBar) * (kinetic_term + potential_term)

    return dpsidt

def rms_error(P1, P2):
    error = np.zeros(M)
    Delta_P = P1 - P2
    for i in range(M):
        error[i] = np.sqrt(np.sum( (Delta_P[:,i])**2 ))
    return error

def expected_position(psi):
    x_bar = np.zeros(M)
    a = x_vec[0]
    b = x_vec[-1]
    k = np.arange(1,)
    for i in range(M):
        f = np.abs(psi[:,i])**2*np.transpose(x_vec)
        psi_i = psi[:,i]
        x_bar[i] = h/3*( f[0]+f[-1]+4*np.sum(f[1:-1:2])+2*np.sum(f[2:-2:2]) )
    return x_bar

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

print("Computing Undamped Exact Solution ...")
psi_eigenstates = eigenstate_solution()

print("Computing Undamped Numerical Solutions ...")
psi_undamped_euler = np.zeros_like(psi_eigenstates, np.complex128)
psi_undamped_euler[:,0] = psi_eigenstates[:,0]
psi_undamped_rk4 = psi_undamped_euler.copy()
for i in range(M):
    psi_undamped_euler[0,i-1] = 0.0
    psi_undamped_euler[-1,i-1] = 0.0
    psi_undamped_rk4[0,i-1] = 0.0
    psi_undamped_rk4[-1,i-1] = 0.0
    psi_undamped_euler[:,i] = euler_step(psi_undamped_euler[:,i-1], dpsi_dt_undamped, t_vec[i], dt)
    psi_undamped_rk4[:,i] = runge_kutta_step(psi_undamped_rk4[:,i-1], dpsi_dt_undamped, t_vec[i], dt)
P_eigenstates = np.abs(psi_eigenstates)**2
P_undamped_euler = np.abs(psi_undamped_euler)**2
P_undamped_rk4 = np.abs(psi_undamped_rk4)**2
error_undamped_euler = rms_error(P_eigenstates, P_undamped_euler)
error_undamped_rk4 = rms_error(P_eigenstates, P_undamped_rk4)
x_bar_eigenstates = expected_position(psi_eigenstates)
x_bar_undamped_euler = expected_position(psi_undamped_euler)
x_bar_undamped_rk4 = expected_position(psi_undamped_rk4)

print("Computing Damped Numerical Solution ...")
psi_damped = np.zeros_like(psi_eigenstates, np.complex128)
psi_damped[:,0] = psi_eigenstates[:,0]
for i in range(1,M):
    psi_damped[0,i-1] = 0.0
    psi_damped[-1,i-1] = 0.0
    psi_damped[:,i] = runge_kutta_step(psi_damped[:,i-1], dpsi_dt_damped, t_vec[i], dt)
P_damped = np.abs(psi_damped)**2
x_bar_damped = expected_position(psi_damped)
    
# solve time dependent equation for damped case

# plot:

# plotting the exact, Euler and RK4 methods for the undamped case:
# First, we plot the probability densities, |ψ(x)|², versus time using the three mathematical methods
plt.figure(figsize = (16, 5))
timeIndecies = [0, M//3, 2 * M//3, M - 1] # We choose four indecies, but could we chose more...?

for i, idx in enumerate(timeIndecies):
    plt.subplot(1, 4, i + 1)
    plt.plot(x_vec, P_eigenstates[:, idx], color = '#001f3f', label = 'Exact') # Plotting Exact
    plt.plot(x_vec, P_undamped_euler[:, idx], color = '#CFB53B', label = 'Euler') # Plotting Euler
    plt.plot(x_vec, P_undamped_rk4[:, idx], color = '#DCDCDC', label = "RK4") # Plotting RK4

    plt.title(f'Time = {t_vec[idx]:.2e} s')
    plt.xlabel('x')
    plt.ylabel('|ψ(x)|²')
    plt.grid(True)
    if i == 0:
        plt.legend()
    
plt.tight_layout()
plt.suptitle('Undamped Harmonic Oscillator: |ψ(x)|² at Different Times', y = 1.05)
plt.show()
plt.savefig('probDensitiesVsPos')

# Then, we plot the expected position, <x>, versus time using the three mathematical methods
plt.figure(figsize = (10, 5))
plt.plot(t_vec, x_bar_eigenstates, color = '#001f3f', label = 'Exact') # Plotting Exact
plt.plot(t_vec, x_bar_undamped_euler, color = '#CFB53B', label = 'Euler') # Plotting Euler
plt.plot(t_vec, x_bar_undamped_rk4, color = '#DCDCDC', label = "RK4" ) # Plotting RK4

plt.xlabel('Time (s)')
plt.ylabel(r'$\langle x  \rangle$')
plt.suptitle('Undamped Harmonic Oscillator: <x> at Different Times')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('expectedPosVsTime')

# Then we plot the RMS error versus time for the Euler's method and RK4 Method
plt.figure(figsize = (10, 6))
plt.plot(t_vec, error_undamped_euler, color = '#CFB53B', label = 'Euler RMS Error') # Plotting Euler RMS Error
plt.plot(t_vec, error_undamped_rk4, color = '#DCDCDC', label = 'Euler RMS Error') # Plotting Euler RMS Error
plt.xlabel('Time (s)')
plt.ylabel('RMS Error')
plt.suptitle('Undamped Harmonic Oscillator: RMS Errors at Different Times')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig('rmsErrorVsTime')