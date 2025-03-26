# store psi as Nx1 array

def generate_initial_state():
    # get initial psi
    return psi

def euler_step(psi, dpsi_dt_function, dt):

def runge_kutta_step(psi, dpsi_dt_function, dt):

def crank_nicholson_step(psi, dpsi_dt_function, dt):
    # might need a separate version for damped and undamped

# V(x) of harmonic potential
def harmonic_potential(x):

def dpsi_dt_undamped(psi, x, t):

def dpsi_dt_damped(psi, x, t, alpha):

#def time_dependent_solution():

def eigenstate_solution():
    # get full solution by evolving eigenstates directly
    return psi # N x M for N points, M timesteps



# solve time dependent equation for all three timestepping methods
# compare to eigenstate solution

# solve time dependent equation for damped case

# plot