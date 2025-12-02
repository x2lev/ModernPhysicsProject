from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import imageio.v3 as iio
import commentjson as json
import questionary
import subprocess
import shutil
import time
import os

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_agg import FigureCanvasAgg

def format_time(seconds):
    seconds = int(seconds)
    return f'{seconds // 3600:0>2.0f}:{(seconds % 3600) // 60:0>2.0f}:{seconds % 60:0>2.0f}'

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


os.makedirs('media/frames', exist_ok=True)
for file in os.listdir('media/frames'):
    os.remove(os.path.join('media/frames', file))

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'Fira Sans',
    'font.size': 20
})

config_name = questionary.select(
    'Choose file to read configuration from (you may edit any of these. see custom.jsonc for explanations):',
    sorted([c for c in os.listdir('configurations') if c != 'custom.jsonc']) + ['custom.jsonc']
).ask()

with open(f'configurations/{config_name}', 'r') as f:
    config = json.load(f)
    runtime = config['runtime']
    fps = config['fps']
    split_steps = config['split_steps']
    surf_res = config['surf_res']
    render_res = config['render_res']
    render_limits = config['render_limits']
    eval_limits = config['eval_limits']
    z_prop = config['z_prop']
    gauss_center = config['gauss_center']
    gauss_momentum = config['gauss_momentum']
    gauss_spread = config['gauss_spread']
    potential_field = config['potential_field']

render_wavefunction = questionary.select(
    'What to render?',
    ['complex wavefunction', 'probability density']
).ask() == 'complex wavefunction'

if render_limits['x'][0] < eval_limits['x'][0] and render_limits['x'][1] > eval_limits['x'][1] and render_limits['y'][0] < eval_limits['y'][0] and render_limits['y'][1] > eval_limits['y'][1]:
    render_range = {
        'x': render_limits['x'][1]-render_limits['x'][0],
        'y': render_limits['y'][1]-render_limits['y'][0]
    }
    eval_range = {
        'x': eval_limits['x'][1]-eval_limits['x'][0],
        'y': eval_limits['y'][1]-eval_limits['y'][0]
    }
    eval_center = {
        'x': (eval_limits['x'][0]+eval_limits['x'][1])/2,
        'y': (eval_limits['y'][0]+eval_limits['y'][1])/2
    }
    square_size = {
        'x': surf_res['x'] / render_range['x'],
        'y': surf_res['y'] / render_range['y']
    }
    new_eval_range = {
        'x': np.ceil(eval_range['x'] / square_size['x']) * square_size['x'],
        'y': np.ceil(eval_range['y'] / square_size['y']) * square_size['y']
    }
    eval_limits = {
        'x': [
            eval_center['x'] - new_eval_range['x']/2,
            eval_center['x'] + new_eval_range['x']/2
        ],
        'y': [
            eval_center['y'] - new_eval_range['y']/2,
            eval_center['y'] + new_eval_range['y']/2
        ]
    }
else:
    eval_limits = render_limits

dpi = 100
figsize = (
    (render_res[0]/dpi) // 2 * 2,
    (render_res[1]/dpi) // 2 * 2
)

x = cp.linspace(eval_limits['x'][0], eval_limits['x'][1], surf_res['x'])
y = cp.linspace(eval_limits['y'][0], eval_limits['y'][1], surf_res['y'])
x, y = cp.meshgrid(x, y, indexing='ij')
dx = x[1, 0] - x[0, 0]
dy = y[0, 1] - y[0, 0]

device = cp.cuda.Device()
device_name = cp.cuda.runtime.getDeviceProperties(device.id)['name'].decode()
print(f'Running CuPy on {device_name}...')

kx = cp.fft.fftfreq(x.shape[0], d=dx) * 2 * cp.pi
ky = cp.fft.fftfreq(y.shape[1], d=dy) * 2 * cp.pi
kx, ky = cp.meshgrid(kx, ky, indexing='ij')
k2 = kx ** 2 + ky ** 2

inf = cp.full_like(x, 1_000_000)
one = cp.full_like(x, 1)
zero = cp.full_like(x, 0)
match potential_field:
    case 'harmonic oscillator':
        potential = 0.5 * (x ** 2 + y ** 2)
    case 'square well':
        mask = (x<=.1) & (x>=-.1) & ((y>=2.5) | ((y<=1.5) & (y>=-1.5)) | (y<=-2.5))
        potential = cp.where(mask, inf, zero)
    case 'circle well':
        mask = x**2 + y**2 > 64
        potential = cp.where(mask, inf, zero)
    case 'free particle':
        potential = zero
    case 'coulomb':
        potential = 10 / (x**2+y**2)
    case 'custom':
        potential = None # implement your own!
        assert potential is not None, 'Implement your own potential field!'
    case _:
        raise ValueError('Choose a potential!')
            
# feel free to implement your own!
wavefunc = cp.exp(
    1j*(x*gauss_momentum['x'] + y*gauss_momentum['y'])
    -((x-gauss_center['x'])**2/(2*gauss_spread['x']**2)
      +(y-gauss_center['y'])**2/(2*gauss_spread['y']**2))
)

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
            sol[i] = cp.exp(-0.5j * h * potential) * sol[i]
            sol[i] = cp.fft.ifft2(cp.exp(-0.5j * h * k2) * cp.fft.fft2(sol[i]))
            sol[i] = cp.exp(-0.5j * h * potential) * sol[i]
            n += 1
        rem = (m / n - 1) * (time.time() - t_0)
        print(f'Evaluating split-step {n}/{m} ({n / m:.2%}) - {format_time(rem)}', end='\r')
    print(f'\33[2K\rFinished evaluating {m} split-steps in {format_time(time.time() - t_0)}')
    return sol

psi = solve_schrodinger(psi_0, cp.linspace(0, runtime, runtime * fps))

def zoom_in(a):
    mask = (
        (a[0] >= render_limits['x'][0]) &
        (a[0] <= render_limits['x'][1]) &
        (a[1] >= render_limits['y'][0]) &
        (a[1] <= render_limits['y'][1])
    )
    return a[:, mask]

psi = cp.array([zoom_in(z) for z in psi])
x, y = zoom_in(x), zoom_in(y)

x, y, psi = x.get(), y.get(), psi.get()

mod = np.abs(psi)
arg = np.angle(psi)
rho = np.abs(psi)**2

roygbiv = np.array([
    (0.91, 0.078, 0.086),
    (1.0, 0.647, 0.0),
    (0.98, 0.922, 0.212),
    (0.475, 0.765, 0.078),
    (0.282, 0.49, 0.906),
    (0.294, 0.212, 0.616),
    (0.439, 0.212, 0.616),
    (0.91, 0.078, 0.086)
])

def colormap(colors: np.ndarray, a: np.ndarray):
    c = (len(colors)-1) * a
    index = c.astype(int)
    return (colors[(index + 1)%len(colors)] - colors[index]) * (c - index)[:,:,:,None] + colors[index]

phase_cmap = LinearSegmentedColormap.from_list('roygbiv', roygbiv)
fc = phase_cmap(((arg-np.min(arg))/(np.max(arg) - np.min(arg))))

def find_limit(a: np.ndarray, p):
    for e in np.linspace(cp.max(a), 0, 100):
        if np.count_nonzero(a > e) > 1 - p:
            return e
    return 1

if render_wavefunction:
    render_limits['z'] = [0, float(find_limit(mod, .95))]
else:
    render_limits['z'] = [0, float(find_limit(rho, .95))]

def render_frame(i):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, subplot_kw={'projection': '3d'})
    canvas = FigureCanvasAgg(fig)
    ax.clear()

    ax.set(xlim=render_limits['x'], ylim=render_limits['y'], zlim=render_limits['z'], xlabel='$x$', ylabel='$y$', zlabel='$r$')
    if render_wavefunction:
        surf = ax.plot_surface(x, y, mod[i], cmap=phase_cmap, facecolors=fc[i], antialiased=True)  #, edgecolor='black', linewidth=0.25)
        cbar = fig.colorbar(surf, shrink=0.5, label='$\\theta$')
        cbar.set_ticks(
            ticks=[0, 1 / 4, 1 / 2, 3 / 4, 1],
            labels=['$-\\pi$', '$-\\frac{\\pi}{2}$', '$0$', '$\\frac{\\pi}{2}$', '$\\pi$']
        )
    else:
        ax.plot_surface(x, y, rho[i], cmap='viridis', antialiased=True)
    ax.view_init(elev=25, azim=25)
    ax.set_title(f'Frame {i + 1}/{len(mod)}')

    canvas.draw()
    iio.imwrite(f'media/frames/frame_{i:04d}.png', np.asarray(canvas.buffer_rgba()))
    plt.close(fig)


threads = cpu_count() - 1  # Leave one core free
print(f'Rendering with Matplotlib on {threads} threads...')
render_args = range(len(psi))

time_0 = time.time()
with Pool(threads) as pool:
    for frame, _ in enumerate(pool.imap(render_frame, render_args), 1):
        progress = frame / len(psi)
        elapsed = time.time() - time_0
        remaining = (1 / progress - 1) * elapsed
        print(f'Rendered {frame}/{len(psi)} frames ({progress:.2%}) - {format_time(remaining)}', end='\r')
print(f'\33[2K\rFinished rendering {len(psi)} frames in {format_time(time.time() - time_0)}')

if os.path.exists('media/output.mp4'):
    os.remove('media/output.mp4')
frames = [iio.imread(f'media/frames/frame_{i:04d}.png') for i in range(len(psi))]
iio.imwrite('media/output.mp4', frames, fps=fps)

if shutil.which('mpv'):
    subprocess.run('mpv media/output.mp4', shell=True)
