import numpy as np
import matplotlib.pyplot as plt
from pollution import euler_advection_diffusion_timestep, stable_time_step

def lotka_volterra_sources(P, Z, params):
    """
    My version of the Lotka-Volterra (LV going forward) model
    """

    # Defining the parameters (from the dictionary) for the function
    alpha = params['alpha']  
    beta = params['beta']
    delta = params['delta']
    gamma = params['gamma']
    K = params['K']
    
    # Lotka-Volterra Model
    prey_growth = alpha * P * (1.0 - P / K)  # Logistic model for prey growth with carrying capacity (see Part 4.2 of the LV lab practical)
    prey_death = -beta * P * Z               # Prey death due to predation
    predator_gain = delta * P * Z            # Predator gain from prey
    predator_death = -gamma * Z              # Predator natural decay

    # I then define two varibales, akin to src in the lab, for both prey and predators which will act as the source term when inputting into the advection / diffusion model
    src_prey = prey_growth + prey_death
    src_pred = predator_gain + predator_death
    
    return src_prey, src_pred

def predator_prey_advection_diffusion_step(P, Z, vx, Dx, Dz, params, dt, dx, dz):
    """ 
    Function for integrating the LV model with the atmospheric pollution / dispersal (ap/d going forward), and reactions
    """
    
    # Calculate the LV source terms that will be passed into the ap/d model
    src_prey, src_pred = lotka_volterra_sources(P, Z, params)

    # From ap/d lab, calling the pollution.py function and passing through source terms to update prey and predator population densitities
    P_updated = euler_advection_diffusion_timestep(P, vx, Dx, Dz, src_prey, dt, dx, dz)
    Z_updated = euler_advection_diffusion_timestep(Z, vx, Dx, Dz, src_pred, dt, dx, dz)

    return P_updated, Z_updated

    # P_updated and Z_updated reflect the new population densities after diffusion and advection. This therefore feeds into predatiion as in the next iteration of calulations, the P and Z densities are different than what would have been observed without passing this function

def run_simulation(P0, Z0, domain_size, params, timesteps, save_every):
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
    timesteps (int) : Total number of time steps to simulate
    save_every (int) : Frequency of saving or visualising results
    """
    # Defining the parameters (from the dictionary) for the function (again!)
    current_velocity = params['current_velocity']
    diffusion = params['diffusion']
    
    # The grid is constructed here. I use a similar approach to the ap/d lab (see pollution.py, line 47) i.e. the grid size is determined by the initial conditions for prey
    nz, nx = P0.shape
    
    # Cell dimensions is also determined by my spatial domain size (see pollution.py, line 123)
    
    dx = domain_size[0] / nx  
    dz = domain_size[1] / nz

    # I save the intial conditions, so they can be used again (see pollution.py, line 48)
    P = P0.copy()
    Z = Z0.copy()

    # Wind speed at cell x - faces (see pollution.py, line 137)
    vx = np.zeros((nz, nx + 1)) + current_velocity
    # Diffusion at cell x-faces (see pollution.py, line 139)
    Dx = np.full((nz, nx + 1), diffusion)
    # Diffusion at cell z-faces (see pollution.py, line 141)
    Dz = np.full((nz + 1, nx), diffusion)

    # I then determine a stable time step (see pollution.py, line 143)
    dt = stable_time_step(vx, Dx, Dz, dx, dz)
    print(f'Stable timestep: {dt:.5f}s') # I wanted to print something as confirmation and we no longer have the sim_time defined!

    # Simulation actually starts running
    for t in range(timesteps):
        P, Z = predator_prey_advection_diffusion_step(P, Z, vx, Dx, Dz, params, dt, dx, dz)

        # Plotting function periodically
        if t % save_every == 0:
            print(f'Timestep {t}')
            plt.imshow(P, cmap='Greens', origin='lower', extent=[0, domain_size[0], 0, domain_size[1]])
            plt.colorbar(label='Prey Density')
            plt.title(f'Time = {t * dt:.2f}s')
            plt.show()

    return P, Z


# Parameters Dictionary
params = {
    'alpha': 0.1,    # Prey reproduction rate (birth - non-predation death)
    'beta': 0.02,    # Predation rate i.e. each predator eats ? prey per unit time
    'delta': 0.01,   # Predator reproduction rate i.e. predators produce ? offspring per prey eaten per unit time
    'gamma': 0.1,    # Death rate of predators in absence of prey
    'K': 1.0,        # Carrying capacity (for prey only)
    'current_velocity': 0.2, # Advection coefficient i.e. constant current velocity
    'diffusion': 0.01 # Diffusion coefficient i.e. spreading out of species
}

# Run the model
if __name__ == '__main__':
    P0 = np.random.uniform(0.4, 0.6, size=(100, 100))  # Random initial prey density
    Z0 = np.random.uniform(0.2, 0.4, size=(100, 100))  # Random initial predator density
    final_P, final_Z = run_simulation(P0, Z0, domain_size=(10, 10), params=params, timesteps=500, save_every=50)
