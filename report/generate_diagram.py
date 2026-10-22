import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# 1. TVAE
ax1 = axes[0]
theta = np.linspace(0, 2*np.pi, 200)
# Normal Core
r_norm = 1.8
x_norm = r_norm * np.cos(theta)
y_norm = r_norm * np.sin(theta)
ax1.fill(x_norm, y_norm, color='#3498db', alpha=0.25, label=r'Normal Latent Core ($r < 2.0\sigma$)')
ax1.plot(x_norm, y_norm, '--', color='#2980b9', lw=1.5)

# Normal points
np.random.seed(42)
z_norm = np.random.randn(120, 2) * 0.75
ax1.scatter(z_norm[:, 0], z_norm[:, 1], color='#2980b9', s=16, alpha=0.7)

# Anomaly Tail Ring (2.5 sigma to 4.0 sigma)
r_tail_in = 2.5
r_tail_out = 4.0
x_in = r_tail_in * np.cos(theta)
y_in = r_tail_in * np.sin(theta)
x_out = r_tail_out * np.cos(theta)
y_out = r_tail_out * np.sin(theta)
ax1.fill(np.append(x_out, x_in[::-1]), np.append(y_out, y_in[::-1]), color='#e74c3c', alpha=0.2, label=r'Tail Shell ($2.5\sigma \leq r \leq 4.2\sigma$)')
ax1.plot(x_in, y_in, ':', color='#c0392b', lw=1.2)
ax1.plot(x_out, y_out, '-', color='#c0392b', lw=1.5)

# Anomaly Samples
angles = np.random.uniform(0, 2*np.pi, 35)
radii = np.random.uniform(2.6, 3.8, 35)
z_anom = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
ax1.scatter(z_anom[:, 0], z_anom[:, 1], color='#c0392b', marker='^', s=35, label=r'Synthetic Tail Anomaly $\hat{x}_{tail}$')

# Directional shift arrow
ax1.annotate('', xy=(3.2, 0.4), xytext=(0.8, 0.1),
             arrowprops=dict(facecolor='#8e44ad', edgecolor='#8e44ad', width=2, headwidth=8))
ax1.text(1.2, 0.65, r'Latent Shift $\Delta z \sim \delta_{defect}$', color='#8e44ad', fontweight='bold', fontsize=9)

ax1.set_title('(a) TVAE: Latent Spherical Tail & Shift', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel(r'Latent Dimension $z_1$', fontsize=10)
ax1.set_ylabel(r'Latent Dimension $z_2$', fontsize=10)
ax1.set_xlim(-4.5, 4.5)
ax1.set_ylim(-4.5, 4.5)
ax1.legend(loc='lower left', fontsize=8, framealpha=0.9)
ax1.set_aspect('equal')

# 2. Fence GAN
ax2 = axes[1]
# Normal Data Region
x_core = 2.0 * np.cos(theta) + 0.3 * np.sin(2*theta)
y_core = 1.8 * np.sin(theta)
ax2.fill(x_core, y_core, color='#2ecc71', alpha=0.25, label=r'Normal Physical Regime ($D(x) \to 1$)')
ax2.plot(x_core, y_core, color='#27ae60', lw=1.5)

x_data = np.random.randn(100, 2) * 0.7
ax2.scatter(x_data[:, 0], x_data[:, 1], color='#27ae60', s=16, alpha=0.7)

# Fence Band (gamma = 0.5)
r_fence = 2.8
x_fence = (r_fence + 0.3 * np.sin(3*theta)) * np.cos(theta)
y_fence = (r_fence + 0.2 * np.cos(2*theta)) * np.sin(theta)
ax2.plot(x_fence, y_fence, '--', color='#d35400', lw=2.5, label=r'Discriminator Fence Ring $D(x) = \gamma$')

# Generated boundary points
gen_theta = np.linspace(0, 2*np.pi, 28, endpoint=False) + np.random.normal(0, 0.08, 28)
gen_r = r_fence + np.random.normal(0, 0.12, 28)
x_gen = gen_r * np.cos(gen_theta)
y_gen = gen_r * np.sin(gen_theta)
ax2.scatter(x_gen, y_gen, color='#e67e22', marker='o', s=40, edgecolors='k', linewidth=0.6, label=r'Fence Generator Samples $G(z)$')

# Repulsion arrows
ax2.annotate('', xy=(2.7, 1.8), xytext=(1.8, 1.2),
             arrowprops=dict(facecolor='#d35400', edgecolor='#d35400', arrowstyle='->', lw=1.8))
ax2.text(1.3, 2.1, r'Minimax Boundary Push: $|D(G(z)) - \gamma| \to 0$', color='#d35400', fontsize=8.5, fontweight='bold')

ax2.set_title('(b) Fence GAN: Boundary Band Enclosure', fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel(r'Sensor Feature $x_i$', fontsize=10)
ax2.set_ylabel(r'Sensor Feature $x_j$', fontsize=10)
ax2.set_xlim(-4.2, 4.2)
ax2.set_ylim(-4.2, 4.2)
ax2.legend(loc='lower left', fontsize=8, framealpha=0.9)
ax2.set_aspect('equal')

# 3. TabDDPM
ax3 = axes[2]
step_labels = [r'$t=T$' + '\n(Gaussian Noise)', r'$t=T/2$' + '\n(Denoising Transition)', r'$t=0$' + '\n(Synthesized Anomaly)']

# Column 1: Pure Noise
noise_x = np.random.randn(30) * 0.4
noise_y = np.random.randn(30) * 0.4 + 2.5
ax3.scatter(noise_x, noise_y, color='#7f8c8d', alpha=0.7, s=20)
ax3.text(0, 3.4, step_labels[0], ha='center', fontsize=8.5, fontweight='bold')

# Arrow 1
ax3.annotate('', xy=(1.5, 2.5), xytext=(0.4, 2.5),
             arrowprops=dict(facecolor='#2c3e50', edgecolor='#2c3e50', width=1.5, headwidth=6))
ax3.text(0.95, 2.7, r'Reverse Step' + '\n' + r'$p_\theta(x_{t-1}|x_t)$', ha='center', fontsize=7.5, color='#2c3e50')

# Column 2: Intermediate
mid_x = np.random.randn(30) * 0.3 + 1.8
mid_y = mid_x * 0.6 + np.random.randn(30) * 0.2 + 1.4
ax3.scatter(mid_x, mid_y, color='#f39c12', alpha=0.7, s=20)
ax3.text(1.8, 3.4, step_labels[1], ha='center', fontsize=8.5, fontweight='bold')

# Arrow 2 with Perturbed guidance
ax3.annotate('', xy=(3.3, 2.5), xytext=(2.2, 2.5),
             arrowprops=dict(facecolor='#c0392b', edgecolor='#c0392b', width=1.5, headwidth=6))
ax3.text(2.75, 2.7, r'Perturbed Guidance' + '\n' + r'$\nabla_x \log p(y|x)$', ha='center', fontsize=7.5, color='#c0392b', fontweight='bold')

# Column 3: Synthesized Anomaly vs Normal Real
real_nx = np.linspace(3.0, 4.2, 20)
real_ny = 0.9 * real_nx - 1.2
ax3.plot(real_nx, real_ny, color='#2980b9', lw=2, linestyle=':', label='Normal Physical Manifold')
anom_px = np.random.normal(3.8, 0.2, 25)
anom_py = 0.9 * anom_px - 0.5 + np.random.normal(0, 0.15, 25)
ax3.scatter(anom_px, anom_py, color='#e74c3c', marker='D', s=30, label=r'Guided Anomaly $\hat{x}_0$')
ax3.text(3.6, 3.4, step_labels[2], ha='center', fontsize=8.5, fontweight='bold')

ax3.set_title('(c) TabDDPM: Reverse Diffusion & Guidance', fontsize=12, fontweight='bold', pad=10)
ax3.set_xlabel(r'Diffusion Evolution Trajectory $\to$', fontsize=10)
ax3.set_ylabel(r'Sensor Feature Space', fontsize=10)
ax3.set_xlim(-0.8, 4.6)
ax3.set_ylim(0.5, 3.8)
ax3.legend(loc='lower right', fontsize=8, framealpha=0.9)

plt.tight_layout()
plt.savefig('report/figures/algorithm_schematic.png', bbox_inches='tight')
plt.close()
print('Algorithm schematic figure created successfully!')
