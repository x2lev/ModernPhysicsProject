from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import imageio.v3 as iio
import subprocess
import time

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_agg import FigureCanvasAgg

def format_time(seconds):
    seconds = int(seconds)
    return f'{seconds // 3600:0>2.0f}:{(seconds % 3600) // 60:0>2.0f}:{seconds % 60:0>2.0f}'


subprocess.run('mkdir -p media/frames', shell=True)
subprocess.run('rm -r media/frames/*', shell=True, capture_output=True)

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'Fira Sans',
    'font.size': 36
})

runtime = 40
fps = 60
split_steps = 250
surf_res = [100, 100]
render_res = [3840, 2160]
lim = {'x': [-10, 10], 'y': [-10, 10], 'z': [0, 1]}
zoom = 1

figsize = (render_res[0]/100, render_res[1]/100)

x = cp.linspace(zoom * lim['x'][0], zoom * lim['x'][1], surf_res[0])
y = cp.linspace(zoom * lim['y'][0], zoom * lim['y'][1], surf_res[1])
x, y = cp.meshgrid(x, y, indexing='ij')
dx = x[1, 0] - x[0, 0]
dy = y[0, 1] - y[0, 0]

assert 1/fps/split_steps < 1/2 *  dx**2

device = cp.cuda.Device()
device_name = cp.cuda.runtime.getDeviceProperties(device.id)['name'].decode()
print(f'Running computations on {device_name}...')

kx = cp.fft.fftfreq(x.shape[0], d=dx) * 2 * cp.pi
ky = cp.fft.fftfreq(y.shape[1], d=dy) * 2 * cp.pi
kx, ky = cp.meshgrid(kx, ky, indexing='ij')
k2 = kx ** 2 + ky ** 2

#mask = (x<=.1) & (x>=-.1) & ((y>=2.5) | ((y<=1.5) & (y>=-1.5)) | (y<=-2.5))
mask = (x>7.5) | (y>7.5) | (x<-7.5) | (y<-7.5)
#mask = x**2 + y**2 > 64
inf = cp.full_like(x, 1_000_000)
zero = cp.full_like(x, 0)
V_0 = cp.where(mask, inf, zero)
#V_0 = 0.5 * (x ** 2 + y ** 2)
potential = lambda t: V_0

#wavefunc = lambda y_, x_: (
#    cp.exp(-((x_ + 5) ** 2 + y_ ** 2)*2) * cp.exp(2j * y_)
#    + cp.exp(-((x_ - 5) ** 2 + y_ ** 2)*2) * cp.exp(2j * y_)
#    + cp.exp(-(x_ ** 2 + (y_ + 5) ** 2)*2) * cp.exp(2j * x_)
#    + cp.exp(-(x_ ** 2 + (y_ - 5) ** 2)*2) * cp.exp(-2j * x_)
#)

wavefunc = cp.exp(-2*(x ** 2 + y ** 2))

area = cp.sum(wavefunc * cp.conj(wavefunc)) * dx * dy
psi_0 = wavefunc / cp.sqrt(area)

def solve_schrodinger(p0, t):
    h = (t[1] - t[0]) / split_steps
    sol = cp.zeros(shape=(*t.shape, *p0.shape), dtype=complex)
    sol[0] = p0
    n = 0
    m = split_steps * (len(t) - 1)
    t_0 = time.time()
    for i in range(1, len(t)):
        sol[i] = sol[i - 1]
        for t_i in cp.linspace(t[i - 1], t[i], split_steps):
            sol[i] = cp.exp(-0.5j * h * potential(t_i)) * sol[i]
            sol[i] = cp.fft.ifft2(cp.exp(-0.5j * h * k2) * cp.fft.fft2(sol[i]))
            sol[i] = cp.exp(-0.5j * h * potential(t_i)) * sol[i]
            n += 1
        rem = (m / n - 1) * (time.time() - t_0)
        print(f'Evaluating split-step {n}/{m} ({n / m:.2%}) - {format_time(rem)}', end='\r')
    print(f'\33[2K\rFinished evaluating {m} split-steps in {format_time(time.time() - t_0)}')
    return sol

psi = solve_schrodinger(psi_0, cp.linspace(0, runtime, runtime * fps))

if zoom > 1:
    zoom_in = lambda a: a[
        int(len(x) / 2 * (1 - 1 / zoom)): int(len(x) / 2 * (1 + 1 / zoom)):,
        int(len(y) / 2 * (1 - 1 / zoom)): int(len(y) / 2 * (1 + 1 / zoom)):
    ]
    psi = cp.array([zoom_in(z) for z in psi])
    x, y = zoom_in(x), zoom_in(y)

mod = cp.abs(psi)
arg = cp.angle(psi)
#rho = psi*cp.conj(psi)

roygbiv = cp.array([
    (0.91, 0.078, 0.086),
    (1.0, 0.647, 0.0),
    (0.98, 0.922, 0.212),
    (0.475, 0.765, 0.078),
    (0.282, 0.49, 0.906),
    (0.294, 0.212, 0.616),
    (0.439, 0.212, 0.616),
    (0.91, 0.078, 0.086)
])

def colormap(colors: cp.ndarray, a: cp.ndarray):
    c = (len(colors)-1) * a
    index = c.astype(int)
    return (colors[(index + 1)%len(colors)] - colors[index]) * (c - index)[:,:,:,None] + colors[index]

phase_cmap = LinearSegmentedColormap.from_list('roygbiv', roygbiv.get())
fc = phase_cmap(((arg-cp.min(arg))/(cp.max(arg) - cp.min(arg))).get())

def find_limit(a: cp.ndarray, p):
    for e in cp.linspace(cp.max(a), 0, 100):
        if cp.count_nonzero(a > e) > 1 - p:
            return e
    return 1

lim['z'][1] = float(find_limit(mod, .95))
#lim['z'][1] = find_limit(rho, .95)

x, y, mod, arg = map(lambda a: cp.real(a).get(), (x, y, mod, arg))
#x, y, mod, arg, rho, fc = map(lambda a: cp.real(a).get(), (x, y, mod, arg, rho, fc))

def render_frame(i):
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': '3d'})
    canvas = FigureCanvasAgg(fig)
    ax.clear()

    ax.set(xlim=lim['x'], ylim=lim['y'], zlim=lim['z'], xlabel='$x$', ylabel='$y$', zlabel='$r$')
    surf = ax.plot_surface(x, y, mod[i], cmap=phase_cmap, facecolors=fc[i], antialiased=True)  #, edgecolor='black', linewidth=0.25)
    cbar = fig.colorbar(surf, shrink=0.5, label='$\\theta$')
    cbar.set_ticks(
        ticks=[0, 1 / 4, 1 / 2, 3 / 4, 1],
        labels=['$-\\pi$', '$-\\frac{\\pi}{2}$', '$0$', '$\\frac{\\pi}{2}$', '$\\pi$']
    )
    #ax.plot_surface(x, y, rho[i], cmap='viridis', antialiased=False)
    ax.view_init(elev=25, azim=25)
    ax.set_title(f'Frame {i + 1}/{len(mod)}')

    canvas.draw()
    iio.imwrite(f'media/frames/frame_{i:04d}.png', np.asarray(canvas.buffer_rgba()))
    plt.close(fig)


threads = cpu_count() - 1  # Leave one core free
print(f'Rendering on {threads} threads...')

render_args = range(len(psi))

time_0 = time.time()
with Pool(threads) as pool:
    for frame, _ in enumerate(pool.imap(render_frame, render_args), 1):
        progress = frame / len(psi)
        elapsed = time.time() - time_0
        remaining = (1 / progress - 1) * elapsed
        print(f'Rendered {frame}/{len(psi)} frames ({progress:.2%}) - {format_time(remaining)}', end='\r')
print(f'\33[2K\rFinished rendering {len(psi)} frames in {format_time(time.time() - time_0)}')

subprocess.run('rm media/output.mp4', shell=True, capture_output=True)
subprocess.run(f'ffmpeg -r {fps} -i media/frames/frame_%04d.png -pix_fmt yuv420p media/output.mp4', shell=True,
               capture_output=True)
subprocess.run('mpv media/output.mp4', shell=True, capture_output=True)
