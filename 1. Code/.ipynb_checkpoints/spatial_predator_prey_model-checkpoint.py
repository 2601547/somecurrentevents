import numpy as np
import matplotlib.pyplot as plt
from pollution import euler_advection_diffusion_timestep, stable_time_step
from shared_functions import params, create_grid

def lotka_volterra_sources(P, Z, params, cell_area):
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
    prey_growth = alpha * P #*  (1.0 - P / K)  # Logistic model for prey growth with carrying capacity (see Part 4.2 of the LV lab practical)
    prey_death = -beta * P * Z               # Prey death due to predation
    predator_gain = delta * beta * P * Z     # Predator gain from prey
    predator_death = -gamma * Z              # Predator natural decay

    # I then define two varibales, akin to src in the lab, for both prey and predators which will act as the source term when inputting into the advection / diffusion model
    src_prey = prey_growth + prey_death
    src_pred = predator_gain + predator_death
    
    return src_prey, src_pred

def predator_prey_advection_diffusion_step(P, Z, vx, Dx, Dz, params, dt, dx, dz, cell_area):
    """ 
    Function for integrating the LV model with the atmospheric pollution / dispersal (ap/d going forward), and reactions

    returns
    ------
    P_updated (ndarray): Updated prey density following effects of diffusion / advection
    Z_updated (ndarray): Updated predator density following effects of diffusion / advection
    
    """
    
    # Calculate the LV source terms that will be passed into the ap/d model
    src_prey, src_pred = lotka_volterra_sources(P, Z, params, cell_area)

    # From ap/d lab, calling the pollution.py function and passing through source terms to update prey and predator population densitities
    P_updated = euler_advection_diffusion_timestep(P, vx, Dx, Dz, src_prey, dt, dx, dz)
    Z_updated = euler_advection_diffusion_timestep(Z, vx, Dx, Dz, src_pred, dt, dx, dz)

    return P_updated, Z_updated

    # P_updated and Z_updated reflect the new population densities after diffusion and advection. This therefore feeds into predatiion as in the next iteration of calulations, the P and Z densities are different than what would have been observed without passing this function

