from multiprocessing import Pool, cpu_count, shared_memory
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import imageio.v3 as iio
import subprocess
import time

from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_agg import FigureCanvasAgg

subprocess.run('mkdir -p media/frames', shell=True)
subprocess.run('rm -r media/frames/*', shell=True, capture_output=True)

roygbiv = [
    (0.91, 0.078, 0.086),
    (1.0, 0.647, 0.0),
    (0.98, 0.922, 0.212),
    (0.475, 0.765, 0.078),
    (0.282, 0.49, 0.906),
    (0.294, 0.212, 0.616),
    (0.439, 0.212, 0.616),
    (0.91, 0.078, 0.086)
]

#lum = [0.2126*R + 0.7152*G + 0.0722*B for R, G, B in roygbiv]
lum = [R + G + B for R, G, B in roygbiv]
roygbiv = [(R/L*max(lum), G/L*max(lum), B/L*max(lum)) for (R, G, B), L in zip(roygbiv, lum)]

phase_cmap = LinearSegmentedColormap.from_list('phase_cmap', roygbiv)
norm = Normalize(vmin=-np.pi, vmax=np.pi)

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'Fira Sans',
    'font.size': 12
})

runtime = 60
fps = 60
rk4_steps = 1000
lim = {'x': [-5, 5], 'y': [-5, 5], 'z': [0, 1]}
zoom = 4

x = cp.linspace(zoom * lim['x'][0], zoom * lim['x'][1], 500)
y = cp.linspace(zoom * lim['y'][0], zoom * lim['y'][1], 500)
x, y = cp.meshgrid(x, y, indexing="ij")
dx = x[1,0] - x[0,0]
dy = y[0,1] - y[0,0]

def T(p):
    dpdx, dpdy = cp.gradient(p, dx, dy, edge_order=2)
    d2pdx2 = cp.gradient(dpdx, dx, edge_order=2, axis=0)
    d2pdy2 = cp.gradient(dpdy, dy, edge_order=2, axis=1)
    return -1/2 * (d2pdx2 + d2pdy2)

V = lambda t, p: 0.5 * 10 * (x**2 + y**2) * p

H = lambda t, p: T(p) + V(t, p)
pdv = lambda t, p: -10j * H(t,p)

wavefunc = lambda y_, x_: cp.exp(-((x+3)**2 + (y-2)**2)/2) * cp.exp(2j*x)

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
        for j in range(rk4_steps):
            sol[i] = rk4(t[i-1], sol[i-1], h/rk4_steps)
            rem = (m / (n+1) - 1) * (time.time() - t_0)
            print(f'Evaluating RK4 step {(n:=(n+1))}/{m} ({n/m:.2%}) - {format_time(rem)}', end='\r')
    print(f'\33[2K\rFinished evaluating {m} RK4 steps in {format_time(time.time()-t_0)}')
    return sol

zoom_in = lambda a: a[len(x)/2 * (1 - 1/zoom):len(x)/2 * (1 + 1/zoom):, len(y)/2*(1 - 1/zoom):len(y)/2 * (1 + 1/zoom):]

psi = solve_ivp(pdv, psi_0, cp.linspace(0, runtime, runtime * fps))
psi = cp.array([zoom_in(z) for z in psi])
mod = cp.abs(psi)
arg = cp.angle(psi)
rho = psi*cp.conj(psi)
x, y = zoom_in(x), zoom_in(y)

x, y, mod, arg, rho = map(lambda a: cp.real(a).get(), (x, y, mod, arg, rho))
fc = phase_cmap(norm(arg))
fc[..., :3] = fc[..., :3] * (((mod+.1)/(np.max(mod, axis=(1,2))[:, None, None]+.1))**(1/3))[:,..., None]

def find_limit(a, p):
    for e in np.linspace(np.max(a), 0, 100):
        if np.count_nonzero(a>e) > 1-p:
            return e
    return 1

lim['z'][1] = find_limit(mod, .95)
#lim['z'][1] = find_limit(rho, .95)
def render_frame(i):
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': '3d'})
    canvas = FigureCanvasAgg(fig)
    ax.clear()

    ax.set(xlim=lim['x'], ylim=lim['y'], zlim=lim['z'], xlabel='$x$', ylabel='$y$', zlabel='$r$')
    surf = ax.plot_surface(x, y, mod[i], cmap=phase_cmap, facecolors=fc[i], antialiased=True)#, edgecolor='black', linewidth=0.25)
    cbar = fig.colorbar(surf, shrink=0.5, label='$\\theta$')
    cbar.set_ticks(
        ticks=[0, 1/4, 1/2, 3/4, 1],
        labels=['$-\\pi$', '$-\\frac{\\pi}{2}$', '$0$', '$\\frac{\\pi}{2}$', '$\\pi$']
    )
    #ax.plot_surface(x, y, rho[i], cmap='viridis', antialiased=False)
    ax.view_init(elev=25, azim=25)
    ax.set_title(f'Gaussian Wave Packet in a Quantum Harmonic Oscillator (frame {i + 1}/{len(mod)})')

    canvas.draw()
    iio.imwrite(f'media/frames/frame_{i:04d}.png', np.asarray(canvas.buffer_rgba()))
    plt.close(fig)


num_workers = cpu_count() - 1  # Leave one core free
print(f"Rendering with {num_workers} workers...")

render_args = range(len(psi))

time_0 = time.time()
with Pool(num_workers) as pool:
    for frame, _ in enumerate(pool.imap(render_frame, render_args), 1):
        progress = frame / len(psi)
        elapsed = time.time() - time_0
        remaining = (1 / progress - 1) * elapsed
        print(f'Rendered {frame}/{len(psi)} frames ({progress:.2%}) - {format_time(remaining)}', end='\r')
print(f'\33[2K\rFinished rendering {len(psi)} frames in {format_time(time.time()-time_0)}')

subprocess.run('rm media/output.mp4', shell=True, capture_output=True)
subprocess.run(f'ffmpeg -r {fps} -i media/frames/frame_%04d.png -pix_fmt yuv420p media/output.mp4', shell=True, capture_output=True)
subprocess.run('mpv media/output.mp4', shell=True, capture_output=True)
