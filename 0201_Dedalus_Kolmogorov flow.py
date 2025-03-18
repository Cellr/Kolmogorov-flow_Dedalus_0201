import numpy as np
import dedalus.public as d3
import logging

logger = logging.getLogger(__name__)

Lx, Ly = 4*np.pi, 2*np.pi
Nx, Ny = 256, 128
dealias = 1
stop_sim_time = 1.5e5
timestepper = d3.RK222
max_timestep = 1
dtype = np.float64

coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=dealias)

tau_psi=dist.Field(name='tau_psi')
psi_prime = dist.Field(name='psi_prime', bases=(xbasis, ybasis))
psi_bar = dist.Field(name='psi_bar', bases=(xbasis, ybasis))
costerm = dist.Field(name='costerm', bases=(xbasis, ybasis))

x_grid, y_grid = dist.local_grids(xbasis, ybasis)
ex, ey = coords.unit_vector_fields(dist)

alpha=Ly/Lx
n=4
Rec=2**0.25*n**(3/2)*(1+3*alpha**2/(4*n**2))

chi=1       # ???
nv=1/10     # ???
Re=10
#nv=(1/(chi**0.5*(Ly/2/np.pi)**1.5))**0.5
#Re=np.sqrt(chi)/nv*(Ly/2/np.pi)**1.5

def jacobian(f1, f2):
    return (d3.Differentiate(f1, coords['x']) * d3.Differentiate(f2, coords['y']) -
            d3.Differentiate(f1, coords['y']) * d3.Differentiate(f2, coords['x']))

problem = d3.IVP([psi_prime,tau_psi],namespace=locals())
problem.namespace.update({'t':problem.time})
problem.add_equation("d3.TimeDerivative(lap(psi_prime))- nv * lap(lap(psi_prime)) +tau_psi=-jacobian(psi_prime, lap(psi_bar)) -jacobian(psi_bar, lap(psi_prime))-jacobian(psi_prime, lap(psi_prime)) +costerm -nv*lap(lap(psi_bar))")
problem.add_equation("integ(psi_prime)=0")

solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

#psi_bar['g'] = - np.cos(n * y_grid)/(nv*n**3)
psi_bar['g'] = (Re / n**2) * np.sin(n * y_grid)
psi = psi_bar + psi_prime
vorticity = -d3.Laplacian(d3.Laplacian(psi))
costerm['g'] = n * np.cos(n * y_grid)
#psi_prime['g'] += 0.1 * np.sin(2*np.pi*x_grid/Lx) * np.exp(-(y_grid-0.5)**2/0.01)
#psi_prime['g'] += 0.1 * np.sin(2*np.pi*x_grid/Lx) * np.exp(-(y_grid+0.5)**2/0.01)

# initial random
# Equation 3.1
A = 0.1 * (Re / n**2)
k_min, k_max = 2.5, 9.5
kx_values = np.fft.fftfreq(Nx, d=Lx/Nx) * 2 * np.pi
ky_values = np.fft.fftfreq(Ny, d=Ly/Ny) * 2 * np.pi
kx_grid, ky_grid = np.meshgrid(kx_values, ky_values, indexing='ij')
k_magnitude = np.sqrt(kx_grid**2 + ky_grid**2)
valid_modes = (k_magnitude > k_min) & (k_magnitude < k_max)
random_phases = np.exp(1j * 2 * np.pi * np.random.rand(*kx_grid.shape))
#enst = np.sum((d3.Laplacian(psi_prime))['g']**2)
psi_prime_hat = np.zeros_like(kx_grid, dtype=np.complex128)
psi_prime_hat[valid_modes] = A * random_phases[valid_modes]
psi_prime_initial = np.fft.ifft2(psi_prime_hat).real
psi_prime['g'] = psi_prime_initial
#psi_prime['g'] *= np.sqrt(1/enst)
psi_prime['g'] += 0.05 * np.exp(-((x_grid - Lx/4)**2 + (y_grid - Ly/2)**2) / 0.01)   # Eq 3.2

# Symmetric
psi_prime['g'] = 0.5 * (psi_prime['g'] - np.flipud(psi_prime['g']))
psi_prime['g'] = 0.5 * (psi_prime['g'] - np.fliplr(np.roll(psi_prime['g'], shift=int(Nx/4), axis=1)))

def enforce_symmetries(field):
    field['g'] = 0.5 * (field['g'] - np.flipud(field['g']))
    field['g'] = 0.5 * (field['g'] - np.fliplr(np.roll(field['g'], shift=int(Nx/4), axis=1)))

y_target = 21 * np.pi / 32
vorticity_y_slice = d3.Interpolate(vorticity, coords['y'], y_target)

snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=10)
snapshots.add_task(psi, name='psi')
snapshots.add_task(vorticity, name='vorticity')
snapshots.add_task(vorticity_y_slice, name='vorticity_y_slice')

flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(psi, name='psi')

try:
    logger.info('Starting main loop')
    fixed_timestep = 0.075
    while solver.proceed:
        timestep = fixed_timestep
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            enforce_symmetries(psi_prime)
            max_psi = np.sqrt(flow.max('psi'))
            logger.info('Iteration=%i, Time=%e, dt=%e, max(w)=%f' %(solver.iteration, solver.sim_time, timestep, max_psi))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
