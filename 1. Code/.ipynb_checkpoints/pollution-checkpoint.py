# This code was provided in the "atmospheric pollution / dispersal" lab. Since I will be using some custom functions from it to integrate into my own model, I have saved a copy in this project folder. Besides this comment, no other ammendments have been made to the provided file. 

# Okay nvm there was an error so made the following change:
#     "fig.colorbar(im, cax=cbar_ax, label='$C$ ($\mu$ g / m$^3$)')",  became
#     "fig.colorbar(im, cax=cbar_ax, label=r'$C$ ($\mu$ g / m$^3$)')"

import numpy as np
import matplotlib.pyplot as plt

def stable_time_step(vx, Dx, Dz, dx, dz):
    
    dx2, dz2 = dx*dx, dz*dz
    dt_diff = dx2 * dz2 / (2 * (np.max(Dx)+np.max(Dz)) * (dx2 + dz2 ) + 1.0e-10) # stable time step. 
    dt_cfl = dx / (np.max(vx) + 1.0e-10) # advection stable step
    dt = min(dt_cfl, dt_diff)*0.5

    return dt

def euler_advection_diffusion_timestep(c0, vx, Dx, Dz, src, dt, dx, dz, inversion=False, upwind=True):
    """ 
    Evolve the advection diffusion equation for pollution concentration 
    by one time step. Spatial discretization using the finite volume method
    on a rectangular mesh of rectangular cells. Forward Euler timestep
     
    Advection is limited to wind parallel to the x-axis
    
    Note the time-step and advection schemes here are for illustration. Real
    codes use better schemes (usually :)
    
    parameters
    ----------
    c0 ((nz, nx) ndarray) : cell-centered concentration at the start of the time step
    vx ((nz, nx+1) ndarray) : horizontal wind velocity at x-faces
    Dx ((nz, nx+1) ndarray) : horizontal diffusion coeffcient at x-faces
    Dz ((nz+1, nx) ndarray) : horizontal diffusion coeffcient at z-faces
    src: concentration source per unit time
    dt (float) : timestep. not checked for stability
    
    inversion: affects the boundary condition at the top boundary: 
               when true, pollution cannot cross the inversion.
               when false, concentration set to 0 at the top
    
    upwind: choice of advection schemne. When true, first-order upwind.
    when false, central difference
    
    returns
    ------
    c (ndarray): updated concentration
    
    """
    nz, nx = c0.shape
    c = c0.copy()
    
    #storage for face-centered fluxes. index j correspomnd to 'w' face of cell j
    Fx = np.zeros((nz, nx + 1))
    Fz = np.zeros((nz +1 , nx))
    
    #Diffusive fluxes (per unit volume) 
    Fx[:, 1:nx] = - Dx[:, 1:nx] * (c[:,1:nx] - c[:,0:nx-1]) / (dx*dx)
    Fz[1:nz, :] = - Dz[1:nz, :] * (c[1:nz,:] - c[0:nz-1,:]) / (dz*dz)
    
    #Advective fluxes
    if upwind:
       Fx[:, 1:nx] += np.where(vx[:,1:nx] > 0, vx[:,1:nx] * c[:,0:nx-1], -vx[:,1:nx]*c[:,1:nx]) / dx     
    else:
       Fx[:, 1:nx] += vx[:,1:nx] * (c[:,0:nx-1] + c[:,1:nx]) * 0.5 / dx
    
                           
    c[1:nz-1, 1:nx-1] = c0[1:nz-1, 1:nx-1]  + dt * (
            - (Fx[1:nz-1, 2:nx] - Fx[1:nz-1, 1:nx-1] )
            - (Fz[2:nz, 1:nx-1] - Fz[1:nz-1, 1:nx-1] )
            +  src[1:nz-1, 1:nx-1] )
    
    #upper atmosphere boundary
    if (inversion):
        c[-1, 1:-1] = c[-2, 1:-1] 
        ...
    else:
        c[-1, 1:-1] = 0
        
    #ground boundary - no deposition
    c[0, 1:-1] = c[1, 1:-1] 
        
    #left boundary. assumes upstream!
    c[1:-1, 0] = 0
    
    #right boundary. assumes downstream!
    c[:, -1] = c[:, -2]
    
    return c

def euler_advection_diffusion(c0, vx, Dx, Dz, src, tspan, dx, dz, inversion=False, upwind=True):
    """ 
    Evolve the advection diffusion equation for pollution concentration 
    from time tspan[0] to tspan[1]. Spatial discretization using the finite volume method
    on a rectangular mesh of rectangular cells. Forward Euler timestep.
    Use as m,any timesteps as needed given stability requirements 
    
    return: c(t = tspan[1])
    """
    
    dt_stable = stable_time_step(vx, Dx, Dz, dx, dz)
    
    time = tspan[0]

    c = c0.copy()
    nstep = (tspan[1] - tspan[0]) / dt_stable
    step = 0
    while (time < tspan[1]) and (step < nstep + 1) :
        dt = min(dt_stable, tspan[1] - time)
        c = euler_advection_diffusion_timestep(c, vx, Dx, Dz, src, dt, dx, dz, inversion=inversion, upwind=upwind)
        time += dt
        step += 1
        
    if time < tspan[1]:
        print ('eee', time, tspan[1])
    return c
    
        
def transport_pollution(sim_time, dx_per_km, width_height_km, source_x_z, source_emission_rate, diffusion, wind_speed,inversion=False, upwind=True):
    """
    Simulate pollution transport from a point source
    
    """
    
    # cell dimensions in x-, z- directions, m
    dx = 1.0 / dx_per_km 
    dz = 0.1 * dx 
    
    #mesh
    nx, nz = int(width_height_km[0]/dx) + 2, int(width_height_km[1]/dz) + 2
    x = np.linspace(-dx/2, width_height_km[0] + dx/2, nx)
    z = np.linspace(-dz/2, width_height_km[1] + dz/2, nz)

    #point source position and rate of emission -. ug / m^3
    i_src, j_src = np.argmin(np.abs(z - source_x_z[1])), np.argmin(np.abs(x - source_x_z[0]))
    src = np.zeros((nz,nx))
    src[i_src, j_src] = source_emission_rate/(dx*dz*1.e+6)

    #wind speed at cell x - faces
    vx = np.zeros((nz, nx + 1)) + wind_speed
    #diffusion at cell x-faces
    Dx =  np.full((nz, nx + 1), diffusion)  
    #diffusion at cell z-faces
    Dz =  np.full((nz + 1, nx), diffusion)
    
    dt_stable = stable_time_step(vx, Dx, Dz, dx, dz)
    print (f'{sim_time} second simulation requires ~ {int(sim_time/dt_stable)} steps: stable timestep dt = {dt_stable}')

    #initial condition
    conc0 = np.zeros((nz, nx))
    
    # output evry minute of timesteps
    dt = 60
    # store concentration every minute for 60 minutes
    nstore = int(sim_time/dt)
    #progress log interval
    progress = int(nstore/10)
    
    conc_store = np.zeros((nz,nx,nstore))
    time_store = np.zeros((nstore))
    mstore = 0
    time = 0.0
    for k in range(nstore):
        tspan = (time, time + dt)
        conc = euler_advection_diffusion(conc0, vx, Dx, Dz, src, tspan, dx, dz, inversion=inversion, upwind=upwind)
        #conc = ivp_advection_diffusion(conc0, vx, Dx, Dz, src, tspan, dx, dz, inversion=inversion, upwind=upwind)
        conc0[:,:] = conc[:,:]
        time += dt
        conc_store[:,:,k] = conc[:,:]
        time_store[k] = time
        if (k%progress == 0):
            print (f'progress: time = {time} / {sim_time}')
    print (f'{sim_time} second simulation complete')                    
    return x, z, time_store, conc_store                          
    
def plot_pollution(x, z, time, conc , source_x_z):
    nx = x.shape[0]
    nz = z.shape[0]
    nt = time.shape[0]
    
    dz, dx = z[1] - z[0], x[1] - x[0]
    i_src, j_src = np.argmin(np.abs(z - source_x_z[1])), np.argmin(np.abs(x - source_x_z[0]))
    
    fig = plt.figure(figsize=(12,6))
    m_map = [0, int(nt/2), nt-1]
    for p, m in enumerate(m_map):
        ax = fig.add_subplot(2,3,p+1)
        im = ax.pcolormesh(x[1:nx-1], z[1:nz-1],  conc[1:nz-1,1:nx-1,m], cmap='hot_r', vmin=0, vmax=10)
        if (p == 0):
            ax.set_ylabel('z (km)')
        else:
            ax.set_yticks([])
        ax.set_xlabel('x (km)')
        ax.text(0.9*np.max(x), 0.9*np.max(z), f't = {time[m]:.1f} s', horizontalalignment='right')    
    
    fig.subplots_adjust(right=0.85, wspace=0.2, hspace=0.25)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=r'$C$ ($\mu$ g / m$^3$)')

    ax = fig.add_subplot(2,2,3)
    nx4 = int(nx/4)
    for ix in [1, nx4, 2*nx4, 3*nx4, nx-2]:
        ax.plot(time, conc[1,ix,:], 'o-', ms=2,  label = f'x = {x[ix]:2.2f} km',lw=1)
        ax.legend(fontsize='xx-small', loc = 'upper right')
        ax.set_xlabel(r'time, $t$ (s)')
        ax.set_ylabel(r'Ground $C$ ($\mu$ g / m$^3$)')
    
    ax = fig.add_subplot(2,2,4)
    nt4 = int(nt/4)
    for it in [0, nt4, 2*nt4, 3*nt4, nt-1]:
        ax.plot(x, conc[1,:,it], 'o-', ms=2,   label = f't = {time[it]:2.2f} s',lw=1)
        ax.legend(fontsize='xx-small', loc = 'upper right')
        
        #label the stack
        ax.axvline(x[j_src]-dx/2., lw=0.5, color='k')
        ax.axvline(x[j_src]+dx/2., lw=0.5, color='k')
        ax.arrow(x[j_src]+2.5*dx, 0.9*np.max(conc), -2.0*dx, 0., length_includes_head=True)
        #ax.arrow(x[j_src]-4.5*dx, 0.9*np.max(conc), 4.0*dx, 0., length_includes_head=True, head_width=10)
        ax.text(x[j_src]+3.0*dx, 0.9*np.max(conc),'source cell', verticalalignment='center')
        
        
        ax.set_xlabel(r'$x$ (km)')
        ax.set_ylabel(r'Ground $\mu$ g / m$^3$)')