import numpy as np
from pollution import stable_time_step

def create_grid(P0, Z0, domain_size, params):
    """
    Simulate the predator-prey / advection-diffusion model in a 2D grid

    parameters
    ----------
    P0 (ndarray) : Initial prey density array
    Z0 (ndarray) : Initial predator density array
    domain_size (tuple) : Physical size of the spatial domain (x, z)
    vx ((nz, nx+1) ndarray) : horizontal current velocity at x-faces
    Dx ((nz, nx+1) ndarray) : horizontal diffusion coeffcient at x-faces
    Dz ((nz+1, nx) ndarray) : horizontal diffusion coeffcient at z-faces
    dt (float) : timestep. not checked for stability
    """
    current_velocity = params['current_velocity']
    diffusion = params['diffusion']
    
    # The grid is constructed here. I use a similar approach to the ap/d lab (see pollution.py, line 47) i.e. the grid size is determined by the initial conditions for prey
    nz, nx = P0.shape

    # As requested by Dr Cornfold, I haved added a check that P0 and Z0 are the same shape
    if P0.shape != Z0.shape:
        raise ValueError(f"P0 and Z0 must have the same shape. Got P0.shape={P0.shape}, Z0.shape={Z0.shape}")
    
    # Cell dimensions is also determined by my spatial domain size (see pollution.py, line 123)
    
    dx = domain_size[0] / nx  
    dz = domain_size[1] / nz

    # I save the intial conditions, so they can be used again (see pollution.py, line 48)
    P = P0.copy()
    Z = Z0.copy()

    # Current speed at cell x - faces (see pollution.py, line 137)
    vx = np.zeros((nz, nx + 1)) + current_velocity
    # Diffusion at cell x-faces (see pollution.py, line 139)
    Dx = np.full((nz, nx + 1), diffusion)
    # Diffusion at cell z-faces (see pollution.py, line 141)
    Dz = np.full((nz + 1, nx), diffusion)

    # I then determine a stable time step (see pollution.py, line 143)
    dt = stable_time_step(vx, Dx, Dz, dx, dz) * 10
    print(f'Stable timestep: {dt:.5f}s') # I wanted to print something as confirmation and we no longer have the sim_time defined!

    return {
    'nz': nz, 'nx': nx, 'dx': dx, 'dz': dz,
    'vx': vx, 'Dx': Dx, 'Dz': Dz, 'dt': dt,
    'P': P, 'Z': Z
}

# Parameters Dictionary
params = {
    'alpha': 5.2,           # Prey reproduction rate
    'beta': 3,           # Predation rate
    'delta': 6.2,          # Predator reproduction rate
    'gamma': 6.2,           # Predator death rate
    'K': 1.0,               # Prey carrying capacity
    'current_velocity': 0.0,  # Advection velocity
    'diffusion': 0.00,      # Diffusion coefficient
    'v_m_prey': 0.0,        # Prey migration velocity
    'v_m_pred': 0.0,        # Predator migration scaling
    'time_lag': 0.0         # Time lag for predator migration
}