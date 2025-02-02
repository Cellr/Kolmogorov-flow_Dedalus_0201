import numpy as np
import dedalus.public as d3
import logging
#import math

logger = logging.getLogger(__name__)

Lx, Ly = 4, 1
Nx, Ny = 256, 64
dealias = 3/2
stop_sim_time = 2
timestepper = d3.RK222
max_timestep = 1e-2
dtype = np.float64

coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=dealias)

#tau=dist.VectorField(coords, name='tau', bases=(xbasis, ybasis))
#Re = dist.Field(name='Re', bases=(xbasis, ybasis))
u = dist.VectorField(coords, name='u', bases=(xbasis, ybasis))
#omega = dist.Field(name='omega', bases=(xbasis, ybasis))
#y_field = dist.Field(name='y_field', bases=(xbasis, ybasis))
costerm = dist.Field(name='costerm', bases=(xbasis, ybasis))

x, y = dist.local_grids(xbasis, ybasis)
ex, ey = coords.unit_vector_fields(dist)
n=4
nv=10        #???
chi=2       #???
#costerm=n*np.cos(n*coords['y'])
y_grid = dist.local_grid(ybasis)
Re=np.sqrt(chi)/nv*(Ly/2/np.pi)**1.5

#def gradiant(f):
#    return (d3.Differentiate(f, coords['x']) , d3.Differentiate(f, coords['y']) )
#def gradient_tensor(f):
#    fx, fy = f['g'][0], f['g'][1]
#    return [[d3.Differentiate(f['g'][0], coords['x']), d3.Differentiate(f['g'][0], coords['y'])],
#            [d3.Differentiate(f['g'][1], coords['x']), d3.Differentiate(f['g'][1], coords['y'])]]
#def laplacian(f):
#    return (d3.Differentiate(d3.Differentiate(f, coords['x']), coords['x']) + d3.Differentiate(d3.Differentiate(f, coords['y']), coords['y']))
def curl(f):
    return(d3.Differentiate(f@ey, coords['x'])-d3.Differentiate(f@ex, coords['y']))
def divergence(f):
    return(d3.Differentiate(f@ex, coords['x'])+d3.Differentiate(f@ey, coords['y']))
#def curl(f):
#    return(d3.Differentiate(f['g'][1], coords['x'])-d3.Differentiate(f['g'][0], coords['y']))

problem = d3.IVP([u],namespace=locals())
problem.add_equation(" d3.TimeDerivative(curl(u)) = - d3.DotProduct(u, d3.Gradient(curl(u))) + 1/Re*d3.Laplacian(curl(u)) + costerm ")
problem.add_equation(" divergence(u) = 0 ")
#problem.add_equation(" tau = 0")
#problem.add_equation(" d3.TimeDerivative(omega) = - d3.DotProduct(u, d3.Gradient(curl(u))) + 1/Re*d3.Laplacian(curl(u)) + n*np.cos(n*y_field) ")
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

omega=curl(u)
#costerm['g']['0'] = 0
costerm['g'] = n * np.cos(n * y_grid) 
#y_field['g']=coords['y']
u['g'][0] += 2 * np.sin(2*np.pi/Lx*x/Lx)

#flow = d3.GlobalFlowProperty(solver, cadence=10)
#flow.add_property(u, name='u')
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(omega, name='omega')

try:
    logger.info('Starting main loop')
    fixed_timestep = 1e-2
    while solver.proceed:
        timestep = fixed_timestep
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_omega = np.sqrt(flow.max('omega'))
            logger.info('Iteration=%i, Time=%e, dt=%e, max(w)=%f' %(solver.iteration, solver.sim_time, timestep, max_omega))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
