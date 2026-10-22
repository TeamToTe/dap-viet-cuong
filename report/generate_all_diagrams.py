import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# ==============================================================================
# 1. HÌNH 1: PIPELINE ARCHITECTURE (Clean, Compact, Height=4.2in)
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 4.0))
ax.axis('off')

# Color palette - modern academic
c_navy = '#1e3a8a'
c_blue = '#2563eb'
c_teal = '#0d9488'
c_amber = '#d97706'
c_purple = '#7c3aed'
c_slate = '#475569'

# Step 1: Data Ingestion & Preprocessing
p1 = patches.FancyBboxPatch((0.2, 0.4), 2.5, 3.2, boxstyle="round,pad=0.15", fc='#eff6ff', ec=c_blue, lw=1.8)
ax.add_patch(p1)
ax.text(1.45, 3.25, 'BƯỚC 1: DỮ LIỆU', ha='center', va='center', fontweight='bold', color=c_blue, fontsize=9.5)
ax.text(1.45, 2.75, 'DATASET_REGISTRY', ha='center', va='center', bbox=dict(boxstyle='square,pad=0.15', fc='white', ec=c_blue, lw=1), fontsize=8, fontweight='bold')
ax.text(1.45, 2.15, '• SKAB Valve1 (8 sensors)\n• AI4I 2020 CNC (5 sensors)', ha='center', va='center', fontsize=7.8, color='#1e293b')
ax.text(1.45, 1.45, 'Zero-Defect Split:\nTrain: 100% Bình thường\nTest: Chứa lỗi thật', ha='center', va='center', fontsize=7.2, color='#1e293b', bbox=dict(boxstyle='round,pad=0.15', fc='#dbeafe', ec='none'))
ax.text(1.45, 0.75, 'StandardScaler:\nCHỈ fit trên Normal Train\n(Chống rò rỉ dữ liệu)', ha='center', va='center', fontsize=7.2, fontweight='bold', color='#b91c1c')

# Arrow 1->2
ax.annotate('', xy=(3.0, 2.0), xytext=(2.7, 2.0), arrowprops=dict(facecolor=c_slate, edgecolor=c_slate, width=1.8, headwidth=6))

# Step 2: Generative Training
p2 = patches.FancyBboxPatch((3.1, 0.4), 2.5, 3.2, boxstyle="round,pad=0.15", fc='#f0fdfa', ec=c_teal, lw=1.8)
ax.add_patch(p2)
ax.text(4.35, 3.25, 'BƯỚC 2: HUẤN LUYỆN', ha='center', va='center', fontweight='bold', color=c_teal, fontsize=9.5)
ax.text(4.35, 2.75, 'MODEL_REGISTRY', ha='center', va='center', bbox=dict(boxstyle='square,pad=0.15', fc='white', ec=c_teal, lw=1), fontsize=8, fontweight='bold')
ax.text(4.35, 2.15, '1. TVAE (Mã hóa biến phân)\n2. Fence GAN (Minimax)\n3. TabDDPM (Khuếch tán)', ha='center', va='center', fontsize=7.8, color='#1e293b')
ax.text(4.35, 1.4, 'Học phân phối bình thường:\nL_TVAE = MSE + beta*KL\nL_GAN = |D(G)-gamma| + disp\nL_DDPM = ||eps - eps_theta||^2', ha='center', va='center', fontsize=6.8, color='#134e4a', bbox=dict(boxstyle='round,pad=0.15', fc='#ccfbf1', ec='none'))
ax.text(4.35, 0.75, 'Lưu Checkpoints:\ncheckpoints/<model>.pt\n(Độc lập theo từng Run)', ha='center', va='center', fontsize=7.2, color='#1e293b')

# Arrow 2->3
ax.annotate('', xy=(5.9, 2.0), xytext=(5.6, 2.0), arrowprops=dict(facecolor=c_slate, edgecolor=c_slate, width=1.8, headwidth=6))

# Step 3: Boundary Anomaly Synthesis
p3 = patches.FancyBboxPatch((6.0, 0.4), 2.5, 3.2, boxstyle="round,pad=0.15", fc='#fffbeb', ec=c_amber, lw=1.8)
ax.add_patch(p3)
ax.text(7.25, 3.25, 'BƯỚC 3: SINH MẪU BIÊN', ha='center', va='center', fontweight='bold', color=c_amber, fontsize=9.5)
ax.text(7.25, 2.75, 'BOUNDARY SYNTHESIS', ha='center', va='center', bbox=dict(boxstyle='square,pad=0.15', fc='white', ec=c_amber, lw=1), fontsize=8, fontweight='bold')
ax.text(7.25, 2.15, 'Sinh N = 2,000 mẫu lỗi:\n• TVAE: Vỏ cầu đuôi (Tail)\n• Fence GAN: Hàng rào viền\n• TabDDPM: Dẫn hướng lệch', ha='center', va='center', fontsize=7.5, color='#78350f')
ax.text(7.25, 1.35, 'Đặc tính mẫu sinh:\nNằm ngoài lõi an toàn\nnhưng bám sát biên cơ lý,\ntuân thủ tương quan vật lý', ha='center', va='center', fontsize=7.2, color='#78350f', bbox=dict(boxstyle='round,pad=0.15', fc='#fef3c7', ec='none'))
ax.text(7.25, 0.75, 'Tạo nhãn giả Anomaly\ny_synth = 1', ha='center', va='center', fontsize=7.2, fontweight='bold', color='#92400e')

# Arrow 3->4
ax.annotate('', xy=(8.8, 2.0), xytext=(8.5, 2.0), arrowprops=dict(facecolor=c_slate, edgecolor=c_slate, width=1.8, headwidth=6))

# Step 4 & 5: 3-Tier Evaluation & Reporting
p4 = patches.FancyBboxPatch((8.9, 0.4), 2.9, 3.2, boxstyle="round,pad=0.15", fc='#f5f3ff', ec=c_purple, lw=1.8)
ax.add_patch(p4)
ax.text(10.35, 3.25, 'BƯỚC 4 & 5: ĐÁNH GIÁ 3 TẦNG', ha='center', va='center', fontweight='bold', color=c_purple, fontsize=9.5)
ax.text(10.35, 2.75, 'METRIC_REGISTRY', ha='center', va='center', bbox=dict(boxstyle='square,pad=0.15', fc='white', ec=c_purple, lw=1), fontsize=8, fontweight='bold')
ax.text(10.35, 2.15, 'Tầng 1 (Fidelity): KS, Wasserstein\nTầng 2 (Correlation): Delta Frobenius\nTầng 3 (Utility TSTR): F1, ROC-AUC', ha='center', va='center', fontsize=7.4, color='#4c1d95')
ax.text(10.35, 1.45, 'So sánh Đối chứng:\n• Baseline: Isolation Forest\n• Oracle (Giám sát trần tập Train)', ha='center', va='center', fontsize=7.2, color='#4c1d95', bbox=dict(boxstyle='round,pad=0.15', fc='#ede9fe', ec='none'))
ax.text(10.35, 0.75, 'Xuất tự động:\nruns_leaderboard.csv & 5 biểu đồ', ha='center', va='center', fontsize=7.2, fontweight='bold', color='#5b21b6')

plt.xlim(0, 12.0)
plt.ylim(0.2, 3.8)
plt.tight_layout()
plt.savefig('report/figures/pipeline_architecture.png', bbox_inches='tight')
plt.close()
print('1. pipeline_architecture.png generated (compact)!')

# ==============================================================================
# 2. HÌNH 2: TVAE ARCHITECTURE & TAIL SAMPLING (Height=3.8in)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8), dpi=300)

# (a) Network Architecture
ax1.axis('off')
ax1.set_title('(a) Sơ đồ kiến trúc mạng Tabular VAE', fontsize=9.5, fontweight='bold', color='#1e3a8a')

# Input
ax1.add_patch(patches.FancyBboxPatch((0.2, 1.5), 1.4, 0.8, boxstyle="round,pad=0.08", fc='#f1f5f9', ec='#475569', lw=1.2))
ax1.text(0.9, 1.9, 'Cảm biến $x$\n$x \in \mathbb{R}^D$', ha='center', va='center', fontsize=7.5, fontweight='bold')

# Encoder
ax1.annotate('', xy=(2.0, 1.9), xytext=(1.6, 1.9), arrowprops=dict(facecolor='#475569', width=1.2, headwidth=5))
ax1.add_patch(patches.FancyBboxPatch((2.0, 0.8), 2.0, 2.2, boxstyle="round,pad=0.1", fc='#eff6ff', ec='#2563eb', lw=1.5))
ax1.text(3.0, 2.7, 'Encoder MLP', ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1e40af')
ax1.text(3.0, 2.1, 'Linear(D -> 64) + BN\nLeakyReLU(0.2)\nLinear(64 -> 32) + BN\nLeakyReLU(0.2)', ha='center', va='center', fontsize=6.8, color='#1e293b')
ax1.text(3.0, 1.15, 'fc_mu: $\mu(x)$\nfc_logvar: $\log \sigma^2(x)$', ha='center', va='center', fontsize=7.2, fontweight='bold', color='#1e3a8a')

# Reparameterization
ax1.annotate('', xy=(4.4, 2.1), xytext=(4.0, 2.1), arrowprops=dict(facecolor='#475569', width=1.2, headwidth=5))
ax1.annotate('', xy=(4.4, 1.6), xytext=(4.0, 1.6), arrowprops=dict(facecolor='#475569', width=1.2, headwidth=5))
ax1.add_patch(patches.FancyBboxPatch((4.4, 1.0), 2.0, 1.8, boxstyle="round,pad=0.1", fc='#fffbeb', ec='#d97706', lw=1.5))
ax1.text(5.4, 2.5, 'Reparameterization', ha='center', va='center', fontsize=7.8, fontweight='bold', color='#92400e')
ax1.text(5.4, 1.9, '$\epsilon \sim \mathcal{N}(0, I)$\n$z = \mu + \epsilon \odot \sigma$', ha='center', va='center', fontsize=7.5, color='#92400e', fontweight='bold')
ax1.text(5.4, 1.3, '$\mathcal{L}_{KL} = D_{KL}(q||p)$', ha='center', va='center', fontsize=7.0, color='#b45309')

# Decoder
ax1.annotate('', xy=(6.8, 1.9), xytext=(6.4, 1.9), arrowprops=dict(facecolor='#475569', width=1.2, headwidth=5))
ax1.add_patch(patches.FancyBboxPatch((6.8, 0.8), 2.0, 2.2, boxstyle="round,pad=0.1", fc='#f0fdfa', ec='#0d9488', lw=1.5))
ax1.text(7.8, 2.7, 'Decoder MLP', ha='center', va='center', fontsize=8.5, fontweight='bold', color='#0f766e')
ax1.text(7.8, 2.1, 'Linear(d -> 32) + BN\nLeakyReLU(0.2)\nLinear(32 -> 64) + BN\nLeakyReLU(0.2)', ha='center', va='center', fontsize=6.8, color='#1e293b')
ax1.text(7.8, 1.15, 'Linear(64 -> D)\n-> $\hat{x}$ (Tái tạo)', ha='center', va='center', fontsize=7.2, fontweight='bold', color='#0f766e')

# Output
ax1.annotate('', xy=(9.2, 1.9), xytext=(8.8, 1.9), arrowprops=dict(facecolor='#475569', width=1.2, headwidth=5))
ax1.add_patch(patches.FancyBboxPatch((9.2, 1.5), 1.5, 0.8, boxstyle="round,pad=0.08", fc='#f1f5f9', ec='#475569', lw=1.2))
ax1.text(9.95, 1.9, '$\hat{x} \in \mathbb{R}^D$\nMSE Loss: $\|x - \hat{x}\|^2$', ha='center', va='center', fontsize=7.2, fontweight='bold')

ax1.set_xlim(0, 11.0)
ax1.set_ylim(0.5, 3.2)

# (b) How it works: Tail hypersphere sampling
ax2.set_title('(b) Cơ chế lấy mẫu đuôi siêu cầu & Dịch chuyển định hướng', fontsize=9.5, fontweight='bold', color='#1e3a8a')
theta = np.linspace(0, 2*np.pi, 200)

r_norm = 1.6
ax2.fill(r_norm*np.cos(theta), r_norm*np.sin(theta), color='#3b82f6', alpha=0.22, label='Lõi tiềm ẩn bình thường ($r < 2.0\sigma$)')
ax2.plot(r_norm*np.cos(theta), r_norm*np.sin(theta), '--', color='#1d4ed8', lw=1.2)

np.random.seed(42)
zn = np.random.randn(60, 2) * 0.65
ax2.scatter(zn[:, 0], zn[:, 1], color='#1d4ed8', s=12, alpha=0.6)

# Tail Ring (2.5 to 4.0)
r_in, r_out = 2.4, 3.8
ax2.fill(np.append(r_out*np.cos(theta), (r_in*np.cos(theta))[::-1]),
         np.append(r_out*np.sin(theta), (r_in*np.sin(theta))[::-1]),
         color='#ef4444', alpha=0.18, label='Vỏ cầu lấy mẫu đuôi ($2.5\sigma \leq r \leq 4.2\sigma$)')
ax2.plot(r_in*np.cos(theta), r_in*np.sin(theta), ':', color='#b91c1c', lw=1.0)
ax2.plot(r_out*np.cos(theta), r_out*np.sin(theta), '-', color='#b91c1c', lw=1.2)

# Tail Anomaly points
angles = np.random.uniform(0, 2*np.pi, 20)
radii = np.random.uniform(2.5, 3.6, 20)
ax2.scatter(radii*np.cos(angles), radii*np.sin(angles), color='#b91c1c', marker='^', s=28, label='Mẫu dị biệt đuôi $\hat{z}_{tail} = r \cdot \hat{u}$')

# Directional shift
ax2.annotate('', xy=(2.9, 0.4), xytext=(0.7, 0.1),
             arrowprops=dict(facecolor='#7c3aed', edgecolor='#7c3aed', width=1.5, headwidth=6))
ax2.text(1.0, 0.6, 'Dịch chuyển định hướng:\n$z_{shift} = \mu(x) + \Delta z$', color='#7c3aed', fontweight='bold', fontsize=7.5)

ax2.set_xlabel('Không gian ẩn $z_1$', fontsize=8.5)
ax2.set_ylabel('Không gian ẩn $z_2$', fontsize=8.5)
ax2.set_xlim(-4.2, 4.2)
ax2.set_ylim(-4.2, 4.2)
ax2.set_aspect('equal')
ax2.legend(loc='lower left', fontsize=6.8, framealpha=0.85)
ax2.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('report/figures/tvae_architecture.png', bbox_inches='tight')
plt.close()
print('2. tvae_architecture.png generated (compact)!')

# ==============================================================================
# 3. HÌNH 3: FENCE GAN ARCHITECTURE & BOUNDARY (Height=3.8in)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8), dpi=300)

# (a) Network Architecture
ax1.axis('off')
ax1.set_title('(a) Sơ đồ mạng đối kháng Fence GAN', fontsize=9.5, fontweight='bold', color='#1e3a8a')

# Generator Input
ax1.add_patch(patches.FancyBboxPatch((0.2, 2.1), 1.2, 0.8, boxstyle="round,pad=0.08", fc='#fffbeb', ec='#d97706', lw=1.2))
ax1.text(0.8, 2.5, 'Nhiễu $z \sim \mathcal{N}(0, I)$\n$\text{dim}=16$', ha='center', va='center', fontsize=7.2, fontweight='bold', color='#92400e')

# Generator MLP
ax1.annotate('', xy=(1.8, 2.5), xytext=(1.4, 2.5), arrowprops=dict(facecolor='#475569', width=1.2, headwidth=5))
ax1.add_patch(patches.FancyBboxPatch((1.8, 1.5), 2.2, 2.0, boxstyle="round,pad=0.1", fc='#fff7ed', ec='#ea580c', lw=1.5))
ax1.text(2.9, 3.2, r'Generator $G(z)$', ha='center', va='center', fontsize=8.5, fontweight='bold', color='#c2410c')
ax1.text(2.9, 2.5, 'Linear(16 -> 64) + LeakyReLU\nBatchNorm1d(64)\nLinear(64 -> 32) + LeakyReLU\nLinear(32 -> D)', ha='center', va='center', fontsize=6.5, color='#1e293b')
ax1.text(2.9, 1.8, r'$\mathcal{L}_G = |D(G) - \gamma| + \alpha \mathcal{L}_{disp}$', ha='center', va='center', fontsize=6.8, fontweight='bold', color='#c2410c')

# Generated sample
ax1.annotate('', xy=(4.4, 2.5), xytext=(4.0, 2.5), arrowprops=dict(facecolor='#ea580c', edgecolor='#ea580c', width=1.2, headwidth=5))
ax1.add_patch(patches.FancyBboxPatch((4.4, 2.1), 1.4, 0.8, boxstyle="round,pad=0.08", fc='#fee2e2', ec='#ef4444', lw=1.2))
ax1.text(5.1, 2.5, 'Mẫu sinh $G(z)$\n(Biên dị biệt)', ha='center', va='center', fontsize=7.2, fontweight='bold', color='#991b1b')

# Real normal sample
ax1.add_patch(patches.FancyBboxPatch((4.4, 0.7), 1.4, 0.8, boxstyle="round,pad=0.08", fc='#dcfce7', ec='#22c55e', lw=1.2))
ax1.text(5.1, 1.1, 'Mẫu thật $x_{real}$\n(100% Normal)', ha='center', va='center', fontsize=7.2, fontweight='bold', color='#166534')

# Discriminator MLP
ax1.annotate('', xy=(6.2, 2.3), xytext=(5.8, 2.5), arrowprops=dict(facecolor='#475569', width=1.2, headwidth=5))
ax1.annotate('', xy=(6.2, 1.5), xytext=(5.8, 1.1), arrowprops=dict(facecolor='#475569', width=1.2, headwidth=5))
ax1.add_patch(patches.FancyBboxPatch((6.2, 0.7), 2.2, 2.4, boxstyle="round,pad=0.1", fc='#eff6ff', ec='#2563eb', lw=1.5))
ax1.text(7.3, 2.8, r'Discriminator $D(x)$', ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1e40af')
ax1.text(7.3, 2.1, 'Linear(D -> 64) + LeakyReLU\nDropout(0.2)\nLinear(64 -> 32) + LeakyReLU\nLinear(32 -> 1) + Sigmoid', ha='center', va='center', fontsize=6.5, color='#1e293b')
ax1.text(7.3, 1.1, r'$\mathcal{L}_D = \text{BCE}(D(x), 1)$' + '\n' + r'$+ \text{BCE}(D(G(z)), 0)$', ha='center', va='center', fontsize=6.8, fontweight='bold', color='#1e40af')

# Discriminator Output Score
ax1.annotate('', xy=(8.8, 1.9), xytext=(8.4, 1.9), arrowprops=dict(facecolor='#475569', width=1.2, headwidth=5))
ax1.add_patch(patches.FancyBboxPatch((8.8, 1.2), 1.8, 1.4, boxstyle="round,pad=0.08", fc='#f8fafc', ec='#475569', lw=1.2))
ax1.text(9.7, 2.3, r'Điểm số $D(x)$:', ha='center', va='center', fontsize=7.5, fontweight='bold')
ax1.text(9.7, 1.7, r'$D(x_{real}) \to 1$' + '\n' + r'$D(G(z)) \to \gamma = 0.5$' + '\n' + r'$D(\mathrm{outlier}) \to 0$', ha='center', va='center', fontsize=6.8, color='#1e293b')

ax1.set_xlim(0, 11.0)
ax1.set_ylim(0.4, 3.7)

# (b) How it works: Fence ring + dispersion
ax2.set_title(r'(b) Cơ chế Hàng rào biên $D(x) \approx \gamma$ & Lực đẩy phân tán', fontsize=9.5, fontweight='bold', color='#1e3a8a')

x_core = 1.8 * np.cos(theta) + 0.25 * np.sin(2*theta)
y_core = 1.6 * np.sin(theta)
ax2.fill(x_core, y_core, color='#22c55e', alpha=0.2, label=r'Vùng vận hành bình thường ($D \to 1$)')
ax2.plot(x_core, y_core, color='#16a34a', lw=1.2)
pts_norm = np.random.randn(50, 2) * 0.55
ax2.scatter(pts_norm[:, 0], pts_norm[:, 1], color='#16a34a', s=12, alpha=0.6)

# Fence Ring (gamma = 0.5)
r_fence = 2.6
xf = (r_fence + 0.2 * np.sin(3*theta)) * np.cos(theta)
yf = (r_fence + 0.15 * np.cos(2*theta)) * np.sin(theta)
ax2.plot(xf, yf, '--', color='#ea580c', lw=2.0, label=r'Hàng rào biên $D(x) = \gamma = 0.5$')

# Generator samples
gen_theta = np.linspace(0, 2*np.pi, 20, endpoint=False) + np.random.normal(0, 0.05, 20)
gen_r = r_fence + np.random.normal(0, 0.08, 20)
xg, yg = gen_r * np.cos(gen_theta), gen_r * np.sin(gen_theta)
ax2.scatter(xg, yg, color='#ea580c', marker='o', s=30, edgecolors='k', linewidth=0.6, label=r'Mẫu sinh $G(z)$ tại biên')

# Dispersion repulsion arrows
for i in [2, 7, 13]:
    ax2.annotate('', xy=(xg[(i+1)%len(xg)], yg[(i+1)%len(yg)]), xytext=(xg[i], yg[i]),
                 arrowprops=dict(facecolor='#dc2626', edgecolor='#dc2626', arrowstyle='<->', lw=1.2))
ax2.text(0.0, -3.4, r'$\mathcal{L}_{disp}$: Lực đẩy phân tán chống co cụm (Mode Collapse)',
         ha='center', fontsize=6.8, color='#dc2626', fontweight='bold', bbox=dict(boxstyle='round,pad=0.15', fc='#fef2f2', ec='#fecaca'))

ax2.set_xlabel('Cảm biến $x_i$', fontsize=8.5)
ax2.set_ylabel('Cảm biến $x_j$', fontsize=8.5)
ax2.set_xlim(-3.8, 3.8)
ax2.set_ylim(-3.8, 3.8)
ax2.set_aspect('equal')
ax2.legend(loc='upper right', fontsize=6.8, framealpha=0.85)
ax2.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('report/figures/fence_gan_architecture.png', bbox_inches='tight')
plt.close()
print('3. fence_gan_architecture.png generated (compact)!')

# ==============================================================================
# 4. HÌNH 4: TABDDPM ARCHITECTURE & DIFFUSION (Height=3.8in)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8), dpi=300)

# (a) Network Architecture
ax1.axis('off')
ax1.set_title('(a) Sơ đồ mạng Backbone TabDDPM (Residuals + Time Emb)', fontsize=9.5, fontweight='bold', color='#1e3a8a')

# Input x_t & t
ax1.add_patch(patches.FancyBboxPatch((0.2, 2.1), 1.3, 0.7, boxstyle="round,pad=0.08", fc='#f1f5f9', ec='#475569', lw=1.2))
ax1.text(0.85, 2.45, r'$x_t \in \mathbb{R}^D$' + '\n' + '(Mẫu có nhiễu)', ha='center', va='center', fontsize=7.2, fontweight='bold')

ax1.add_patch(patches.FancyBboxPatch((0.2, 0.8), 1.3, 0.7, boxstyle="round,pad=0.08", fc='#fffbeb', ec='#d97706', lw=1.2))
ax1.text(0.85, 1.15, r'Bước $t \in [1, T]$' + '\n' + 'Cosine Schedule', ha='center', va='center', fontsize=7.0, fontweight='bold', color='#92400e')

# Sinusoidal Time Embedding
ax1.annotate('', xy=(1.9, 1.15), xytext=(1.5, 1.15), arrowprops=dict(facecolor='#475569', width=1.2, headwidth=5))
ax1.add_patch(patches.FancyBboxPatch((1.9, 0.6), 1.8, 1.3, boxstyle="round,pad=0.08", fc='#fef3c7', ec='#d97706', lw=1.5))
ax1.text(2.8, 1.6, 'Sinusoidal Embedding', ha='center', va='center', fontsize=7.2, fontweight='bold', color='#92400e')
ax1.text(2.8, 1.05, 'Linear(dim -> 32) + SiLU\nLinear(32 -> 32)', ha='center', va='center', fontsize=6.5, color='#78350f')

# Fusion Block
ax1.annotate('', xy=(4.1, 2.45), xytext=(1.5, 2.45), arrowprops=dict(facecolor='#475569', width=1.2, headwidth=5))
ax1.annotate('', xy=(4.1, 1.6), xytext=(3.7, 1.15), arrowprops=dict(facecolor='#475569', width=1.2, headwidth=5))

ax1.add_patch(patches.FancyBboxPatch((4.1, 0.7), 2.0, 2.2, boxstyle="round,pad=0.1", fc='#eff6ff', ec='#2563eb', lw=1.5))
ax1.text(5.1, 2.6, 'Input Fusion & Block 1', ha='center', va='center', fontsize=8.0, fontweight='bold', color='#1e40af')
ax1.text(5.1, 2.0, r'$h = \mathrm{fc\_in}(x_t) + \mathrm{fc\_t}(t)$' + '\n' + r'$\mathrm{Block\ 1: Linear + BN + SiLU}$' + '\n' + r'$h_1 = h + b_1(h)\ \mathrm{(Skip)}$', ha='center', va='center', fontsize=6.5, color='#1e293b')

# Block 2 & Output
ax1.annotate('', xy=(6.5, 1.8), xytext=(6.1, 1.8), arrowprops=dict(facecolor='#475569', width=1.2, headwidth=5))
ax1.add_patch(patches.FancyBboxPatch((6.5, 0.7), 2.0, 2.2, boxstyle="round,pad=0.1", fc='#f0fdfa', ec='#0d9488', lw=1.5))
ax1.text(7.5, 2.6, 'Residual Block 2 & Out', ha='center', va='center', fontsize=8.0, fontweight='bold', color='#0f766e')
ax1.text(7.5, 2.0, r'$\mathrm{Block\ 2: Linear + BN + SiLU}$' + '\n' + r'$h_2 = h_1 + b_2(h_1)\ \mathrm{(Skip)}$' + '\n' + r'$\mathrm{fc\_out: Linear}(128 \to D)$', ha='center', va='center', fontsize=6.5, color='#1e293b')

# Output noise eps
ax1.annotate('', xy=(8.9, 1.8), xytext=(8.5, 1.8), arrowprops=dict(facecolor='#475569', width=1.2, headwidth=5))
ax1.add_patch(patches.FancyBboxPatch((8.9, 1.3), 1.5, 1.0, boxstyle="round,pad=0.08", fc='#faf5ff', ec='#7c3aed', lw=1.5))
ax1.text(9.65, 1.95, r'$\hat{\epsilon}_\theta(x_t, t)$', ha='center', va='center', fontsize=8.2, fontweight='bold', color='#6d28d9')
ax1.text(9.65, 1.55, r'$\mathcal{L}_{diff} = \|\epsilon - \hat{\epsilon}_\theta\|^2$', ha='center', va='center', fontsize=6.8, color='#5b21b6')

ax1.set_xlim(0, 10.8)
ax1.set_ylim(0.4, 3.2)

# (b) How it works: Diffusion & Perturbed Reverse Guidance
ax2.set_title(r'(b) Khuếch tán thuận & Khử nhiễu nghịch dẫn hướng', fontsize=9.5, fontweight='bold', color='#1e3a8a')

noise_x = np.random.randn(20) * 0.3 + 0.4
noise_y = np.random.randn(20) * 0.3 + 2.5
ax2.scatter(noise_x, noise_y, color='#64748b', alpha=0.7, s=15)
ax2.text(0.4, 3.1, r'$t = T$' + '\n' + '(Nhiễu Gauss)', ha='center', fontsize=7.2, fontweight='bold', color='#475569')

# Reverse Arrow 1
ax2.annotate('', xy=(1.6, 2.5), xytext=(0.9, 2.5), arrowprops=dict(facecolor='#2563eb', edgecolor='#2563eb', width=1.2, headwidth=5))
ax2.text(1.25, 2.7, 'Khử nhiễu\n' + r'$p_\theta(x_{t-1}|x_t)$', ha='center', fontsize=6.5, color='#1d4ed8')

# Step t = T/3
mid_x = np.random.randn(20) * 0.2 + 2.0
mid_y = mid_x * 0.45 + np.random.randn(20) * 0.12 + 1.6
ax2.scatter(mid_x, mid_y, color='#d97706', alpha=0.7, s=15)
ax2.text(2.0, 3.1, r'$t = T/3$' + '\n' + '(Chuyển tiếp)', ha='center', fontsize=7.2, fontweight='bold', color='#b45309')

# Reverse Arrow 2 with Guidance
ax2.annotate('', xy=(3.3, 2.5), xytext=(2.6, 2.5), arrowprops=dict(facecolor='#dc2626', edgecolor='#dc2626', width=1.5, headwidth=6))
ax2.text(2.95, 2.7, 'Perturbed Guidance\n' + r'$\nabla_x \log p_{guide}$', ha='center', fontsize=6.5, color='#b91c1c', fontweight='bold')

# Step t = 0
real_nx = np.linspace(3.1, 4.2, 20)
real_ny = 0.75 * real_nx - 1.1
ax2.plot(real_nx, real_ny, ':', color='#2563eb', lw=1.8, label='Đa tạp vận hành bình thường')

anom_px = np.random.normal(3.8, 0.18, 20)
anom_py = 0.75 * anom_px - 0.45 + np.random.normal(0, 0.1, 20)
ax2.scatter(anom_px, anom_py, color='#dc2626', marker='D', s=25, label=r'Mẫu lỗi sinh có dẫn hướng $\hat{x}_0$')
ax2.text(3.7, 3.1, r'$t = 0$' + '\n' + '(Hội tụ cảm biến)', ha='center', fontsize=7.2, fontweight='bold', color='#991b1b')

ax2.set_xlabel(r'Trục thời gian khuếch tán $\to$', fontsize=8.5)
ax2.set_ylabel('Không gian cảm biến vật lý', fontsize=8.5)
ax2.set_xlim(-0.2, 4.4)
ax2.set_ylim(0.8, 3.6)
ax2.legend(loc='lower right', fontsize=6.8, framealpha=0.85)
ax2.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('report/figures/tabddpm_architecture.png', bbox_inches='tight')
plt.close()
print('4. tabddpm_architecture.png generated (compact)!')
