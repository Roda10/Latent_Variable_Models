import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the convex function phi(x, y) = x^2 + y^2
def phi(x, y):
    return x**2 + y**2

# Define the random vector (X, Y): takes two pairs of values with equal probability (1/2)
point1 = (1, 1)  # (x1, y1)
point2 = (3, 2)  # (x2, y2)
p1, p2 = 0.5, 0.5

# Compute expectations
E_X = p1 * point1[0] + p2 * point2[0]  # E[X]
E_Y = p1 * point1[1] + p2 * point2[1]  # E[Y]
phi_E_XY = phi(E_X, E_Y)  # phi(E[X], E[Y])
E_phi_XY = p1 * phi(*point1) + p2 * phi(*point2)  # E[phi(X, Y)]

# Create mesh for plotting the surface
x = np.linspace(-4, 4, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = phi(X, Y)

# Points for the chord (line segment between (x1, y1, phi(x1, y1)) and (x2, y2, phi(x2, y2)))
chord_x = np.array([point1[0], point2[0]])
chord_y = np.array([point1[1], point2[1]])
chord_z = np.array([phi(*point1), phi(*point2)])

# Create the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, label=r'$\phi(x, y) = x^2 + y^2$')

# Plot the chord
ax.plot(chord_x, chord_y, chord_z, 'r-', linewidth=3, label='Chord')

# Plot the points on the surface
ax.scatter([point1[0], point2[0]], [point1[1], point2[1]], [phi(*point1), phi(*point2)], 
           color='red', s=100, label='Points on surface')

# Plot phi(E[X], E[Y]) on the surface
ax.scatter([E_X], [E_Y], [phi_E_XY], color='green', s=150, label=r'$\phi(E[X], E[Y])$')

# Plot E[phi(X, Y)] on the chord
# Find the point on the chord at (E_X, E_Y): linearly interpolate z
t = 0.5  # Since p1 = p2 = 0.5, the midpoint corresponds to E[X], E[Y]
chord_z_at_E = p1 * phi(*point1) + p2 * phi(*point2)  # This is E[phi(X, Y)]
ax.scatter([E_X], [E_Y], [E_phi_XY], color='purple', s=150, label=r'$E[\phi(X, Y)]$')

# Add annotations
ax.text(point1[0], point1[1], phi(*point1) + 1, f'({point1[0]}, {point1[1]}, {phi(*point1):.1f})', color='red')
ax.text(point2[0], point2[1], phi(*point2) + 1, f'({point2[0]}, {point2[1]}, {phi(*point2):.1f})', color='red')
ax.text(E_X + 0.2, E_Y, phi_E_XY, f'φ(E[X], E[Y]) = {phi_E_XY:.2f}', color='green')
ax.text(E_X + 0.2, E_Y, E_phi_XY + 1, f'E[φ(X, Y)] = {E_phi_XY:.2f}', color='purple')

# Customize the plot
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('φ(x, y)')
ax.set_title("3D Visualization of Jensen's Inequality")
ax.legend()

# Adjust view angle for better visibility
ax.view_init(elev=30, azim=135)

plt.show()

# Print the inequality
print(f"φ(E[X], E[Y]) = {phi_E_XY:.2f}")
print(f"E[φ(X, Y)] = {E_phi_XY:.2f}")
print(f"Jensen's Inequality: φ(E[X], E[Y]) <= E[φ(X, Y)] ({phi_E_XY:.2f} <= {E_phi_XY:.2f})")