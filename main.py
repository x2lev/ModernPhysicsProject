from multiprocessing import Pool, cpu_count, shared_memory
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import imageio.v3 as iio
import subprocess
import time

from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.backends.backend_agg import FigureCanvasAgg

subprocess.run('mkdir -p media/frames', shell=True)
subprocess.run('rm -r media/frames/*', shell=True, capture_output=True)

roygbiv = [
    '#E81416',
    '#FFA500', '#FFA500',
    '#FAEB36', '#FAEB36',
    '#79C314', '#79C314',
    '#487DE7', '#487DE7',
    '#4B369D', '#4B369D',
    '#70369D', '#70369D',
    '#E81416'
]
rb = ['#E81416', '#487DE7', '#E81416']
r = ['#E81416', '#E81416']
phase_cmap = LinearSegmentedColormap.from_list('phase_cmap', roygbiv)
norm = Normalize(vmin=-cp.pi, vmax=cp.pi)

runtime = 30
FPS = 60
rk4_steps = 10
lim = {'x': [-10, 10], 'y': [-10, 10], 'z': [0, 1]}
ZOOM = 2

x = cp.linspace(ZOOM * lim['x'][0], ZOOM * lim['x'][1], 750)
y = cp.linspace(ZOOM * lim['y'][0], ZOOM * lim['y'][1], 750)
x, y = cp.meshgrid(x, y, indexing="ij")
dx = x[1,0] - x[0,0]
dy = y[0,1] - y[0,0]

def T(p):
    dpdx, dpdy = cp.gradient(p, dx, dy, edge_order=2)
    d2pdx2 = cp.gradient(dpdx, dx, edge_order=2, axis=0)
    d2pdy2 = cp.gradient(dpdy, dy, edge_order=2, axis=1)
    return -1/2 * (d2pdx2 + d2pdy2)

V = lambda t, p: 0.5 * (x**2 + y**2) * p

H = lambda t, p: T(p) + V(t, p)
pdv = lambda t, p: -1j * H(t,p)

wavefunc = lambda y_, x_: cp.exp(-(x**2 + (y-2)**2)) * cp.exp(5j*x)

def area(f, m):
    x_ = cp.linspace(x[0], x[-1], m)
    y_ = cp.linspace(y[0], y[-1], m)
    dx_ = x_[1,0] - x_[0,0]
    dy_ = y_[0,1] - y_[0,0]
    return cp.trapz(cp.trapz(f(y, x) * cp.conj(f(y, x)), dx=dy_), dx=dx_)

psi_0 = wavefunc(y, x) / cp.sqrt(area(wavefunc, 1000))


def format_time(seconds):
    return f'{seconds // 3600:0>2.0f}:{(seconds % 3600) // 60:0>2.0f}:{seconds % 60:0>2.0f}'


def solve_ivp(f, p0, t):
    def rk4(t_, p, h_):
        k_1 = f(t_, p)
        k_2 = f(t_ + h_ / 2, p + h_ * k_1 / 2)
        k_3 = f(t_ + h_ / 2, p + h_ * k_2 / 2)
        k_4 = f(t_ + h_, p + h_ * k_3)
        return p + h_/6 * (k_1 + k_2 + k_3 + k_4)

    h = t[1] - t[0]
    sol = cp.zeros(shape=(*t.shape, *p0.shape), dtype=complex)
    sol[0] = p0
    n = 0
    m = rk4_steps*(len(t)-1)
    t_0 = time.time()
    for i in range(1, len(t)):
        sol[i] = sol[i-1]
        for j, _ in enumerate(cp.linspace(float(t[i-1]), float(t[i]), rk4_steps)):
            sol[i] = rk4(t[i-1], sol[i], h/rk4_steps)
            rem = (m / (n+1) - 1) * (time.time() - t_0)
            print(f'Evaluating RK4 step {(n:=(n+1))}/{m} ({n/m:.2%}) - {format_time(rem)}', end='\r')
    print(f'\33[2K\rFinished evaluating RK4 steps in {format_time(time.time()-t_0)}')
    return sol

zoom_in = lambda a: a[len(x)/2 * (1-1/ZOOM):len(x)/2 * (1+1/ZOOM):, len(y)/2 * (1-1/ZOOM):len(y)/2 * (1+1/ZOOM):]

psi = solve_ivp(pdv, psi_0, cp.linspace(0, runtime, runtime * FPS))
psi = cp.array([zoom_in(z) for z in psi])
mod = cp.abs(psi)
arg = cp.angle(psi)
rho = psi*cp.conj(psi)
x, y = zoom_in(x), zoom_in(y)

x, y, mod, arg, rho = map(lambda a: cp.real(a).get(), (x, y, mod, arg, rho))
fc = phase_cmap(norm(arg))

cp._default_memory_pool.free_all_blocks()

#lim['z'][1] = min(1, np.max(mod))
lim['z'][1] = min(1, np.max(rho))
def render_frame(i):
    fig, ax = plt.subplots(figsize=(10.8, 10.8), subplot_kw={'projection': '3d'})
    canvas = FigureCanvasAgg(fig)
    ax.clear()

    ax.set(xlim=lim['x'], ylim=lim['y'], zlim=lim['z'], xlabel='x', ylabel='y', zlabel='r')
    #ax.plot_surface(x, y, mod[i], facecolors=fc[i])
    ax.plot_surface(x, y, rho[i], cmap='viridis')
    ax.view_init(elev=30, azim=-45)
    ax.set_title(f'Frame {i + 1}/{len(mod)}')

    canvas.draw()
    iio.imwrite(f'media/frames/frame_{i:04d}.png', np.asarray(canvas.buffer_rgba()))
    plt.close(fig)


num_workers = cpu_count() - 1  # Leave one core free
print(f"Rendering with {num_workers} workers...")

render_args = range(len(psi))

time_0 = time.time()
with Pool(num_workers) as pool:
    for frame, _ in enumerate(pool.imap(render_frame, render_args), 1):
        progress = frame / len(mod)
        elapsed = time.time() - time_0
        remaining = (1 / progress - 1) * elapsed
        print(f'Rendered {frame}/{len(mod)} frames ({progress:.2%}) - {format_time(remaining)}', end='\r')
print(f'\33[2K\rFinished rendering in {format_time(time.time()-time_0)}')

subprocess.run('rm media/output.mp4', shell=True, capture_output=True)
subprocess.run(f'ffmpeg -r {FPS} -i media/frames/frame_%04d.png -pix_fmt yuv420p media/output.mp4', shell=True, capture_output=True)
subprocess.run('mpv media/output.mp4', shell=True)
