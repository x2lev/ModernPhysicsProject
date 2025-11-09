import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import imageio.v3 as iio
import subprocess

from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.backends.backend_agg import FigureCanvasAgg

subprocess.run('mkdir -p media/frames', shell=True)
subprocess.run('rm -r media/frames/*', shell=True, capture_output=True)

fig, ax = plt.subplots(figsize=(10.8, 10.8), subplot_kw={'projection': '3d'})
canvas = FigureCanvasAgg(fig)

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
lim = {'x': [-25, 25], 'y': [-25, 25], 'z': [0, 1]}
ZOOM = 5

x = cp.linspace(ZOOM * lim['x'][0], ZOOM * lim['x'][1], 500)
y = cp.linspace(ZOOM * lim['y'][0], ZOOM * lim['y'][1], 500)
x, y = cp.meshgrid(x, y, indexing="ij")
dx = x[1,0] - x[0,0]
dy = y[0,1] - y[0,0]

#wavefunc = lambda y_, x_: cp.cos(x_**2 + y_**2) / (x_**2 + y_**2 + 1)**2

#wavefunc = lambda y_, x_: (
#    cp.exp(-((x_-5)**2 + y_**2)) * cp.exp(-5j * x_) +
#    cp.exp(-((x_+5)**2 + y_**2)) * cp.exp(5j * x_)
#)

r = lambda x_, y_: cp.sqrt(x_**2 + y_**2)
theta = lambda x_, y_: cp.arctan2(y_, x_)
wavefunc = lambda y_, x_: cp.exp(-r(x_, y_)**2 / 5) * cp.exp(1j * theta(x_, y_))

def area(f, m):
    x_ = cp.linspace(x[0], x[-1], m)
    y_ = cp.linspace(y[0], y[-1], m)
    dx_ = x_[1,0] - x_[0,0]
    dy_ = y_[0,1] - y_[0,0]
    return cp.trapz(cp.trapz(f(y, x) * cp.conj(f(y, x)), dx=dy_), dx=dx_)

psi_0 = wavefunc(y, x) / cp.sqrt(area(wavefunc, 100))

#mod = cp.abs(psi_0)
#arg = cp.angle(psi_0)
#ax.set(xlim=lim['x'], ylim=lim['y'], zlim=lim['z'], xlabel='x', ylabel='y', zlabel='r')
#ax.plot_surface(x.get(), y.get(), mod.get(), facecolors=phase_cmap(norm(arg.get())), edgecolor='none')
#ax.set_title(f'Initial Condition')
#plt.show()

def T(p):
    dpdx, dpdy = cp.gradient(p, dx, dy, edge_order=2)
    d2pdx2 = cp.gradient(dpdx, dx, edge_order=2, axis=0)
    d2pdy2 = cp.gradient(dpdy, dy, edge_order=2, axis=1)
    return 0.5j * (d2pdx2 + d2pdy2)

def V(t, p):
    return 0

def H(t, p):
    return T(p) + V(t, p)

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
    steps = 10
    for i in range(1, len(t)):
        sol[i] = sol[i-1]
        for j, _ in enumerate(cp.linspace(float(t[i-1]), float(t[i]), steps)):
            sol[i] = rk4(t[i-1], sol[i], h/steps)
            print(f'RK4 step number {(n:=int(steps*(i-1)+j+1))}/{(m:=steps*(len(t)-1))} ({n/m:.2%})', end='\r')
    print()
    return sol

zoom_in = lambda a: a[len(x)/2 * (1-1/ZOOM):len(x)/2 * (1+1/ZOOM):, len(y)/2 * (1-1/ZOOM):len(y)/2 * (1+1/ZOOM):]

psi = cp.array([zoom_in(z) for z in solve_ivp(H, psi_0, cp.linspace(0, runtime, runtime * FPS))])
mod = cp.abs(psi)
arg = cp.angle(psi)
x, y = zoom_in(x), zoom_in(y)

x, y, mod, arg = x.get(), y.get(), mod.get(), arg.get()
fc = phase_cmap(norm(arg))

lim['z'][1] = np.max(mod)*1
for frame in range(len(psi)):
    print(f'Rendering frame {(n:=frame + 1)}/{(m:=len(mod))} ({n/m:.2%})', end='\r')
    ax.clear()
    ax.set(xlim=lim['x'], ylim=lim['y'], zlim=lim['z'], xlabel='x', ylabel='y', zlabel='r')
    ax.plot_surface(x, y, mod[frame], facecolors=fc[frame], edgecolor='none')
    ax.set_title(f'Frame {frame + 1}/{len(mod)}')

    if frame % (FPS*10) == 0:
        pass#plt.show()

    canvas.draw()
    iio.imwrite(f'media/frames/frame_{frame:04d}.png', np.asarray(canvas.buffer_rgba()))

plt.close(fig)

subprocess.run('rm media/output.mp4', shell=True, capture_output=True)
subprocess.run(f'ffmpeg -r {FPS} -i media/frames/frame_%04d.png -pix_fmt yuv420p media/output.mp4', shell=True)
subprocess.run('mpv media/output.mp4', shell=True)
