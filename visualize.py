from simulator import GridWorldSimulator

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


file_name = "world.pkl"


world = GridWorldSimulator.load(file_name)


fig, ax = plt.subplots(figsize=(12,12))
ln, = plt.plot([], [], 'co')
margin = 5
def init():
    ax.set_xlim(-margin, world.size+margin)
    ax.set_ylim(-margin, world.size+margin)
    ax.set_aspect('equal')
    return ln


def update(frame):
    world.step()
    locs = np.array([c.loc for c in world.creatures.values()])
    ln.set_data(locs[:,0], locs[:,1])
    ax.set_title(f"Step:{frame}")
    return ln

ani = FuncAnimation(fig, update, frames=200, repeat_delay = None, init_func=init,
                    blit=False)
plt.show()