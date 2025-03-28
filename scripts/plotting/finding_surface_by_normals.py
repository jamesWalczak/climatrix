"""The script plots the method behind finding the surface based on SDF."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("darkgrid")

RANDOM_POINT = (9, 8)

x = np.linspace(-5, 15, 100)
y = np.linspace(-5, 15, 100)
X, Y = np.meshgrid(x, y)

radius = 10
SDF = np.sqrt(X**2 + Y**2) - radius

px, py = RANDOM_POINT

nx = px / np.sqrt(px**2 + py**2)
ny = py / np.sqrt(px**2 + py**2)

intersection_y = np.sqrt(radius**2 - px**2) if px <= radius else None

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(
    SDF,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin="lower",
    cmap="coolwarm",
    alpha=0.6,
)
ax.contour(X, Y, SDF, levels=10, colors="black", linewidths=0.5)

theta = np.linspace(0, 2 * np.pi, 100)
circle_x = radius * np.cos(theta)
circle_y = radius * np.sin(theta)
ax.plot(
    circle_x,
    circle_y,
    color="purple",
    linewidth=2,
    label="Shape to reconstrruct",
)

ax.scatter(px, py, color="black", label="Grid point", zorder=3)

ax.arrow(
    px,
    py,
    nx * 1.8,
    ny * 1.8,
    head_width=0.4,
    head_length=0.4,
    fc="green",
    ec="green",
    label="Normal Vector",
)
ax.arrow(
    px,
    py,
    nx * 2,
    0,
    head_width=0.4,
    head_length=0.2,
    fc="blue",
    ec="blue",
    linestyle="dashed",
    label="X-component",
)
ax.arrow(
    px,
    py,
    0,
    ny * 2,
    head_width=0.4,
    head_length=0.2,
    fc="orange",
    ec="orange",
    linestyle="dashed",
    label="Y-component",
)

ax.axvline(px, color="red", linestyle="dashed", linewidth=1.5)
if intersection_y is not None:
    ax.scatter(
        px,
        intersection_y,
        color="red",
        marker="x",
        s=50,
        label="Intersection point",
    )

ax.set_xlim(-5, 15)
ax.set_ylim(-5, 15)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_title("Surface finding with SDF gradient")
ax.legend()
ax.set_aspect("equal")

fig.savefig("docs/plots/finding_surface_by_normals.svg")
plt.show()
