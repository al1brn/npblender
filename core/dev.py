# ====================================================================================================
# Tests
# ====================================================================================================

import matplotlib.pyplot as plt
from numpy import pi
import numpy as np

from maths.distribs import *

# Ici, on suppose que tes fonctions sont importées :
# from mymodule import circle_dist, arc_dist, rect_dist, pie_dist, disk_dist

def plot_2d(points, title, ax=None, equal=True):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(points[:, 0], points[:, 1], '.', markersize=2)
    ax.set_title(title)
    ax.grid(True)
    if equal:
        ax.set_aspect('equal')

def test_distributions_2d(count=1000):
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    # ----- CIRCLE -----
    d = circle_dist(radius=1.0, count=count, seed=0)
    plot_2d(d['points'], "circle_dist", axs[0, 0])

    # ----- ARC -----
    d = arc_dist(radius=1.0, arc_angle=pi, arc_center=0.0, count=count, seed=1)
    plot_2d(d['points'], "arc_dist", axs[0, 1])

    # ----- RECT -----
    d = rect_dist(a=2.0, b=1.0, count=count, seed=2)
    plot_2d(d['points'], "rect_dist", axs[0, 2])

    # ----- PIE -----
    d = pie_dist(radius=0.5, outer_radius=1.0, pie_angle=pi, pie_center=pi/2, count=count, seed=3)
    plot_2d(d['points'], "pie_dist", axs[1, 0])

    # ----- DISK -----
    d = disk_dist(radius=1.0, count=count, seed=4)
    plot_2d(d['points'], "disk_dist", axs[1, 1])

    axs[1, 2].axis('off')
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# from mymodule import sphere_dist, dome_dist, cylinder_dist, cube_dist, ball_dist

def plot_3d(points, title, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2)
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True)

def test_distributions_3d(count=1000):
    fig = plt.figure(figsize=(14, 10))

    # 3x2 subplots
    axes = [fig.add_subplot(2, 3, i+1, projection='3d') for i in range(5)]

    # ----- SPHERE -----
    d = sphere_dist(radius=1.0, count=count, seed=0)
    plot_3d(d['points'], "sphere_dist", axes[0])

    # ----- DOME -----
    d = dome_dist(radius=1.0, angle=np.pi/2, axis=(0, 1, 0), count=count, seed=1)
    plot_3d(d['points'], "dome_dist", axes[1])

    # ----- CYLINDER -----
    d = cylinder_dist(radius=1.0, height=2.0, arc_angle=np.pi, count=count, seed=2)
    plot_3d(d['points'], "cylinder_dist", axes[2])

    # ----- CUBE -----
    d = cube_dist(size=(2, 2, 2), center=(0, 0, 0), count=count, seed=3)
    plot_3d(d['points'], "cube_dist", axes[3])

    # ----- BALL -----
    d = ball_dist(radius=1.0, angle=np.pi, axis=(0, 0, 1), count=count, seed=4)
    plot_3d(d['points'], "ball_dist", axes[4])

    plt.tight_layout()
    plt.show()



#test_distributions_2d()
test_distributions_3d()


