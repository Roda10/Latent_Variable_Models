import numpy as np
import matplotlib.pyplot as plt

# Define the convex function phi(x) = x^2
def phi(x):
    return x**2

# Define the random variable X: takes values x1 and x2 with equal probability (1/2)
x1, x2 = 1, 3
p1, p2 = 0.5, 0.5

# Compute expectations
E_X = p1 * x1 + p2 * x2  # E[X]
phi_E_X = phi(E_X)  # phi(E[X])
E_phi_X = p1 * phi(x1) + p2 * phi(x2)  # E[phi(X)]

# Create points for plotting the function
x = np.linspace(0, 4, 400)
y = phi(x)

# Points for the chord (line segment between (x1, phi(x1)) and (x2, phi(x2)))
chord_x = np.array([x1, x2])
chord_y = np.array([phi(x1), phi(x2)])

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r'$\phi(x) = x^2$', color='blue')  # Plot the convex function
plt.plot(chord_x, chord_y, 'r--', label='Chord', linewidth=2)  # Plot the chord
plt.scatter([x1, x2], [phi(x1), phi(x2)], color='red', s=100, label='Points (x1, φ(x1)), (x2, φ(x2))')  # Points on curve
plt.scatter([E_X], [phi_E_X], color='green', s=100, label=r'$\phi(E[X])$', zorder=5)  # phi(E[X])
plt.scatter([E_X], [E_phi_X], color='purple', s=100, label=r'$E[\phi(X)]$', zorder=5)  # E[phi(X)] on chord
plt.axvline(E_X, color='gray', linestyle=':', label=r'$E[X]$')  # Vertical line at E[X]

# Add annotations
plt.text(x1, phi(x1) + 0.5, f'({x1}, {phi(x1)})', color='red')
plt.text(x2, phi(x2) + 0.5, f'({x2}, {phi(x2)})', color='red')
plt.text(E_X + 0.1, phi_E_X, f'φ(E[X]) = {phi_E_X}', color='green')
plt.text(E_X + 0.1, E_phi_X + 0.5, f'E[φ(X)] = {E_phi_X}', color='purple')

# Customize the plot
plt.title("Geometric Intuition of Jensen's Inequality")
plt.xlabel('x')
plt.ylabel('φ(x)')
plt.legend()
plt.grid(True)
plt.show()

# Print the inequality
print(f"φ(E[X]) = {phi_E_X}")
print(f"E[φ(X)] = {E_phi_X}")
print(f"Jensen's Inequality: φ(E[X]) <= E[φ(X)] ({phi_E_X} <= {E_phi_X})")