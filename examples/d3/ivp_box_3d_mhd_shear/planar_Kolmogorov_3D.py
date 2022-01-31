import numpy as np
from mpi4py import MPI
import time
from dedalus import public as de
from dedalus.extras import flow_tools
import logging
logger = logging.getLogger(__name__)

# Parameters
Lx, Ly, Lz = (2.0*np.pi, 2.0*np.pi, 2.0*np.pi)
Nx, Ny, Nz = (32, 32, 32)
HB = 0.5
MA = 1.0/np.sqrt(HB)
Pm = 0.1
Reynolds = 10
mReynolds = Reynolds*Pm
init_dt = 0.001 * Lx / (Nx)
# dt_0 = 0.05 * Lx / Nx  # 1e-4
dt_max = 100.0*init_dt

# simulation stop conditions
stop_sim_time = np.inf  # stop time in simulation time units
stop_wall_time = 2.5*60.*60.  # stop time in terms of wall clock
stop_iteration = 32000  # stop time in terms of iteration count


def filter_field(field, frac=0.5):
    """
    Taken from Dedalus example notebook on Taylor-Couette flow. This is meant to filter out small-scale noise in
    the initial condition, which can cause problems.
    :param field:
    :param frac:
    :return:
    """
    dom = field.domain
    local_slice = dom.dist.coeff_layout.slices(scales=dom.dealias)
    coeff = []
    for i in range(dom.dim)[::-1]:
        coeff.append(np.linspace(0, 1, dom.global_coeff_shape[i], endpoint=False))
    cc = np.meshgrid(*coeff)

    field_filter = np.zeros(dom.local_coeff_shape, dtype='bool')
    for i in range(dom.dim):
        field_filter = field_filter | (cc[i][local_slice] > frac)
    field['c'][field_filter] = 0j


# Create bases and domain
start_init_time = time.time()  # start a timer to see how long things take
x_basis = de.Fourier('x', Nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier('y', Ny, interval=(0, Ly), dealias=3/2)
z_basis = de.Fourier('z', Nz, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64, mesh=None)

# 3D MHD in terms of magnetic vector potential in Coulomb gauge (div(A) = 0)
# Drawing heavily from advice from Ben Brown/examples he and the other Dedalus developers shared on github,
# particularly https://github.com/DedalusProject/dedalus_scaling/blob/master/RB_mhd_3d.py
problem = de.IVP(domain, variables=['u', 'v', 'w', 'p', 'Ax', 'Ay', 'Az', 'phi'], time='t')
problem.parameters['MA2inv'] = 1.0/(MA**2.0)  # 99% sure that this is just H_B^*
problem.parameters['MA'] = MA
problem.parameters['Reinv'] = 1.0/Reynolds
problem.parameters['Rminv'] = 1.0/mReynolds
problem.parameters['Lx'] = Lx
problem.parameters['Ly'] = Ly
problem.parameters['Lz'] = Lz

problem.substitutions['Bx'] = "(dy(Az) - dz(Ay))"
problem.substitutions['By'] = "(dz(Ax) - dx(Az))"
# problem.substitutions['Bz'] = "(1.0 + dx(Ay) - dy(Ax))"
problem.substitutions['Bz'] = "(dx(Ay) - dy(Ax))"
problem.substitutions['Bz_tot'] = "(1.0 + Bz)"
problem.substitutions['Jx'] = "(dy(Bz) - dz(By))"
problem.substitutions['Jy'] = "(dz(Bx) - dx(Bz))"
problem.substitutions['Jz'] = "(dx(By) - dy(Bx))"
# Here, Ox, Oy, Oz are the x, y, z components of vorticity Omega = curl(u)
problem.substitutions['Ox'] = "(dy(w) - dz(v))"
problem.substitutions['Oy'] = "(dz(u) - dx(w))"
problem.substitutions['Oz'] = "(dx(v) - dy(u))"
problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Ly/Lz'

# Note the pressure term in this formulation is really p + u^2/2
problem.add_equation("dt(u) - Reinv*(dx(dx(u)) + dy(dy(u)) + dz(dz(u))) + dx(p) - MA2inv*Jy = v*Oz - w*Oy + MA2inv*(Jy*Bz - Jz*By)")
problem.add_equation("dt(v) - Reinv*(dx(dx(v)) + dy(dy(v)) + dz(dz(v))) + dy(p) + MA2inv*Jx = w*Ox - u*Oz + MA2inv*(Jz*Bx - Jx*Bz)")
problem.add_equation("dt(w) - Reinv*(dx(dx(w)) + dy(dy(w)) + dz(dz(w))) + dz(p) = u*Oy - v*Ox + MA2inv*(Jx*By - Jy*Bx) + sin(x)")
# What's commented out here: old code where the momentum advection term was v dot grad v, as opposed to what's above
#problem.add_equation("dt(u) - Reinv*(dx(dx(u)) + dy(dy(u)) + dz(dz(u))) + dx(p) - MA2inv*Jy = - u*dx(u) - v*dy(u) - w*dz(u) + MA2inv*(Jy*Bz - Jz*By)")
#problem.add_equation("dt(v) - Reinv*(dx(dx(v)) + dy(dy(v)) + dz(dz(v))) + dy(p) + MA2inv*Jx = - u*dx(v) - v*dy(v) - w*dz(u) + MA2inv*(Jz*Bx - Jx*Bz)")
#problem.add_equation("dt(w) - Reinv*(dx(dx(w)) + dy(dy(w)) + dz(dz(w))) + dz(p) = - u*dx(w) - v*dy(w) - w*dz(w) + MA2inv*(Jx*By - Jy*Bx) + sin(x)")

# Induction equations but for A. Note that if div(A) = 0 then curl(curl(A)) = -Laplacian(A)
# problem.add_equation("dt(Ax) - Rminv*(dx(dx(Ax)) + dy(dy(Ax)) + dz(dz(Ax))) + dx(phi) - v = v*Bz - w*By")
# problem.add_equation("dt(Ay) - Rminv*(dx(dx(Ay)) + dy(dy(Ay)) + dz(dz(Ay))) + dy(phi) + u = w*Bx - u*Bz")
# problem.add_equation("dt(Az) - Rminv*(dx(dx(Az)) + dy(dy(Az)) + dz(dz(Az))) + dz(phi) = u*By - v*Bx")
problem.add_equation("dt(Ax) + Rminv*Jx + dx(phi) - v = v*Bz - w*By")
problem.add_equation("dt(Ay) + Rminv*Jy + dy(phi) + u = w*Bx - u*Bz")
problem.add_equation("dt(Az) + Rminv*Jz + dz(phi) = u*By - v*Bx")

# div(u) = 0 and div(A) = 0 reduce to 0 = 0 for the kx=ky=kz Fourier mode. Thus, closing the system at that k requires
# specifying p and phi instead
problem.add_equation("dx(u) + dy(v) + dz(w) = 0", condition="(nx!=0) or (ny!=0) or (nz!=0)")
problem.add_equation("p=0", condition="(nx==0) and (ny==0) and (nz==0)")
problem.add_equation("dx(Ax) + dy(Ay) + dz(Az) = 0", condition="(nx!=0) or (ny!=0) or (nz!=0)")
problem.add_equation("phi=0", condition="(nx==0) and (ny==0) and (nz==0)")

# Build solver
solver = problem.build_solver(de.timesteppers.SBDF3)  # RK443)
logger.info('Solver built')

# Initial conditions
x = domain.grid(0)
y = domain.grid(1)
z = domain.grid(2)
psi = domain.new_field(name='psi')
u = solver.state['u']
v = solver.state['v']
w = solver.state['w']
#for f in [psi, u, v, w]:
    #f.set_scales(domain.dealias, keep_data=False)


# Noise initial conditions
# Random perturbations, initialized globally for same results in parallel
pert = 1e-4
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]
psi['g'] = pert * noise * np.ones_like(z)
# filter_field(psi)
psi.set_scales(1/2, keep_data=True)
psi['c']
psi['g']
psi.set_scales(1, keep_data=True)
psi.differentiate('z', out=u)
psi['g'] = -1.0*np.copy(psi['g'])
psi.differentiate('x', out=w)
w.set_scales(1, keep_data=True)
w['g'] = w['g'] + np.sin(x)

# Integration parameters
solver.stop_sim_time = stop_sim_time
solver.stop_wall_time = stop_wall_time
solver.stop_iteration = 1000

# Analysis
dumps = solver.evaluator.add_file_handler('dumps', iter=5000, max_writes=20)
dumps.add_system(solver.state)

snap = solver.evaluator.add_file_handler('snapshots', iter=100, max_writes=1000)
scalar = solver.evaluator.add_file_handler('scalar', iter=10, max_writes=10000)
for task_name in ["u", "v", "w", "Bx", "By", "Bz_tot"]:
    snap.add_task("interp(" + task_name + ", x=0)", scales=1, name=task_name + " side")
    snap.add_task("interp(" + task_name + ", y=0)", scales=1, name=task_name + " front")
    snap.add_task("interp(" + task_name + ", z=0)", scales=1, name=task_name + " top")
    #snap.add_task("integ(" + task_name + ", 'y', 'z')", scales=1, name=task_name + "2Davg")
    #snap.add_task("integ(" + task_name + ", 'z')", scales=1, name=task_name + "1Davg")

    scalar.add_task("vol_avg(" + task_name + "**2)", name=task_name + " squared")

# CFL
CFL = flow_tools.CFL(solver, initial_dt=init_dt, cadence=1, safety=0.2, max_dt=dt_max)
                     #max_change=1.5, max_dt=dt_max, threshold=0.1)
CFL.add_velocities(('u', 'v', 'w'))
CFL2 = flow_tools.CFL(solver, initial_dt=init_dt, cadence=1, safety=0.2, max_dt=dt_max)
                     #max_change=1.5, min_change=2e-1, max_dt=dt_max, threshold=0.1)#maybe need to add max dt and safety as
                                                                                #input variables if timestepping is an issue
CFL2.add_velocities(('Bx/MA', 'By/MA', 'Bz/MA'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("sqrt(u*u + v*v + w*w)", name='u_abs')

# Main loop
end_init_time = time.time()
logger.info('Initialization time: %f' %(end_init_time-start_init_time))
try:
    logger.info('Starting loop')
    start_run_time = time.time()
    while solver.ok:
        dt1 = CFL.compute_dt()
        dt2 = CFL2.compute_dt()

        dt = np.min([dt1, dt2])
        solver.step(dt)
        if (solver.iteration-1) % 10 == 0: #was 10
            logger.info('Iteration: %i, sim_time: %e, dt: %e, wall_time: %.2f sec' %(solver.iteration, solver.sim_time, dt, time.time()-start_run_time))
            logger.info('dt/dt2 = %f' %(dt/dt2))
            logger.info('Max u_abs = %f' %flow.max('u_abs'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' % solver.iteration)
    logger.info('Sim end time: %f' % solver.sim_time)
    logger.info('Run time: %.2f sec' % (end_run_time-start_run_time))
    logger.info('Run time: %f cpu-hr' % ((end_run_time-start_run_time)/60/60*domain.dist.comm_cart.size))

