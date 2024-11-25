import numpy as np
import matplotlib.pyplot as plt


def stable_time_step(vx, Dx, Dz, dx, dz):
    """
    Calculate a stable time step for the advection-diffusion equation.
    """
    dx2, dz2 = dx * dx, dz * dz
    max_diffusion = max(np.max(Dx), np.max(Dz), 1.0e-10)
    max_velocity = max(np.max(vx), 1.0e-10)
    diffusive_dt = dx2 * dz2 / (2 * max_diffusion * (dx2 + dz2))
    cfl_dt = dx / max_velocity
    return min(cfl_dt, diffusive_dt) * 0.5


def euler_advection_diffusion_timestep(c0, vx, Dx, Dz, src, dt, dx, dz, bc="zero", upwind=True):
    """
    Evolve the advection-diffusion equation for pollution concentration by one time step.

    Parameters:
    ----------
    bc: str
        Boundary condition type. Options: "zero" (Dirichlet), "reflecting", "periodic".
    """
    nz, nx = c0.shape
    c = c0.copy()
    Fx = np.zeros((nz, nx + 1))
    Fz = np.zeros((nz + 1, nx))

    # Diffusive fluxes
    dcdx = (c[:, 1:] - c[:, :-1]) / dx
    dcdy = (c[1:, :] - c[:-1, :]) / dz
    Fx[:, 1:nx] = -Dx[:, 1:nx] * dcdx
    Fz[1:nz, :] = -Dz[1:nz, :] * dcdy

    # Advective fluxes
    if upwind:
        Fx[:, 1:nx] += np.where(vx[:, 1:nx] > 0, vx[:, 1:nx] * c[:, :-1], vx[:, 1:nx] * c[:, 1:]) / dx
    else:
        Fx[:, 1:nx] += vx[:, 1:nx] * (c[:, 1:] + c[:, :-1]) * 0.5 / dx

    # Update concentration
    c[1:nz-1, 1:nx-1] += dt * (
        -(Fx[1:nz-1, 2:nx] - Fx[1:nz-1, 1:nx-1]) / dx
        -(Fz[2:nz, 1:nx-1] - Fz[1:nz-1, 1:nx-1]) / dz
        + src[1:nz-1, 1:nx-1]
    )

    # Boundary conditions
    if bc == "zero":
        c[0, :] = 0  # Ground boundary
        c[-1, :] = 0  # Top boundary
        c[:, 0] = 0  # Left boundary
        c[:, -1] = 0  # Right boundary
    elif bc == "reflecting":
        c[0, :] = c[1, :]  # Reflect at ground
        c[-1, :] = c[-2, :]  # Reflect at top
        c[:, 0] = c[:, 1]  # Reflect at left
        c[:, -1] = c[:, -2]  # Reflect at right
    elif bc == "periodic":
        c[0, :] = c[-2, :]  # Wrap around at ground
        c[-1, :] = c[1, :]  # Wrap around at top
        c[:, 0] = c[:, -2]  # Wrap around at left
        c[:, -1] = c[:, 1]  # Wrap around at right

    return c


def create_grid(P0, Z0, domain_size, params):
    """
    Create a computational grid and calculate initial conditions.
    """
    if P0.shape != Z0.shape:
        raise ValueError(f"P0 and Z0 must have the same shape. Got P0.shape={P0.shape}, Z0.shape={Z0.shape}")

    nz, nx = P0.shape
    dx = domain_size[0] / nx
    dz = domain_size[1] / nz

    vx = np.full((nz, nx + 1), params['current_velocity'])
    Dx = np.full((nz, nx + 1), params['diffusion'])
    Dz = np.full((nz + 1, nx), params['diffusion'])

    dt = stable_time_step(vx, Dx, Dz, dx, dz)
    print(f"Stable timestep: {dt:.5f}s")

    return {'nz': nz, 'nx': nx, 'dx': dx, 'dz': dz, 'vx': vx, 'Dx': Dx, 'Dz': Dz, 'dt': dt, 'P': P0.copy(), 'Z': Z0.copy()}


def lotka_volterra_sources(P, Z, params):
    """
    Calculate the source terms for the Lotka-Volterra equations.
    """
    alpha, beta, delta, gamma, K = params['alpha'], params['beta'], params['delta'], params['gamma'], params['K']
    prey_growth = alpha * P * (1.0 - P / K)  # Logistic growth
    prey_death = -beta * P * Z
    predator_gain = delta * beta * P * Z
    predator_death = -gamma * Z
    return prey_growth + prey_death, predator_gain + predator_death


def predator_prey_advection_diffusion_step(P, Z, vx, Dx, Dz, params, dt, dx, dz, bc="zero"):
    """
    Integrate the Lotka-Volterra equations with advection-diffusion effects.
    """
    src_prey, src_pred = lotka_volterra_sources(P, Z, params)
    P_updated = euler_advection_diffusion_timestep(P, vx, Dx, Dz, src_prey, dt, dx, dz, bc=bc)
    Z_updated = euler_advection_diffusion_timestep(Z, vx, Dx, Dz, src_pred, dt, dx, dz, bc=bc)
    return P_updated, Z_updated


# Parameters dictionary
params = {
    'alpha': 5.2, 'beta': 3.0, 'delta': 6.2, 'gamma': 6.2,
    'K': 1.0, 'current_velocity': 0.1, 'diffusion': 0.01
}

# Test Framework
def test_conservation_of_mass():
    nx, nz = 20, 20
    P0 = np.random.rand(nz, nx)
    Z0 = np.random.rand(nz, nx)
    domain_size = (10, 10)
    grid = create_grid(P0, Z0, domain_size, params)
    P, Z = predator_prey_advection_diffusion_step(
        grid['P'], grid['Z'], grid['vx'], grid['Dx'], grid['Dz'], params, grid['dt'], grid['dx'], grid['dz'], bc="reflecting"
    )
    print("Test passed!")

# Run test
test_conservation_of_mass()
