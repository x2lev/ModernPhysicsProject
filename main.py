import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as anim

from scipy.integrate import odeint
from scipy.integrate import solve_ivp

from matplotlib.colors import LinearSegmentedColormap, Normalize

fig,ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "3d"})

roygbiv = ['#E81416', '#FFA500', '#FAEB36', '#79C314', '#487DE7', '#4B369D', '#70369D', '#E81416']
phase_cmap = LinearSegmentedColormap.from_list('phase_cmap', roygbiv)
norm = Normalize(vmin=-np.pi, vmax=np.pi)

runtime = 10
FPS = 60

x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
x, y = np.meshgrid(x, y, indexing="ij")
dx = x[1] - x[0]
dt = 0.1*dx**2

wf = 1/np.sqrt(np.pi) * np.exp((-1+1j)*(x**2+y**2))

def T(w, dt):
    return 0.5j * np.sum(np.sum(g2 for g2 in np.gradient(g1)) for g1 in np.gradient(w)) * dt

def V(w, dt):
    return w

def H(w, dt):
    return T(w, dt) + V(w, dt)

surf = [ax.plot_surface(x, y, np.abs(wf), facecolors=phase_cmap(norm(np.angle(wf))), antialiased=False)]

elapsed_time = 0
def update(frame):
    global wf, elapsed_time
    while elapsed_time < frame/FPS:
        print(dt)
        k1 = H(wf, dt)
        k2 = H(wf + 0.5 * dt * k1, dt)
        k3 = H(wf + 0.5 * dt * k2, dt)
        k4 = H(wf + dt * k3, dt)

        wf += 0.5j * (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

        elapsed_time += dt

    surf[0].remove()
    surf[0] = ax.plot_surface(x, y, np.abs(wf), facecolors=phase_cmap(norm(np.angle(wf))), antialiased=False)
    ax.set_title(f"Frame {frame+1}/{FPS*runtime}")
    return surf

ax.set(xlim=[-10, 10], ylim=[-10, 10], zlim=[0,1], xlabel='x', ylabel='y', zlabel='r')
ani = anim.FuncAnimation(fig=fig, func=update, frames=FPS*runtime, interval=1/FPS)
ani.save(
    'output.mp4',
    fps=FPS,
    progress_callback=lambda i, n: print(f'Saving frame {i+1}/{n}'))
#plt.show()
