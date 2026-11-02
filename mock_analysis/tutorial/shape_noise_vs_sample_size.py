import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
sigma_e = 0.26       # intrinsic ellipticity dispersion per component
gamma = 0.02         # assumed common shear
epsilon_obs = 0.20   # example observed ellipticity

# Normal probability density function, avoiding a SciPy dependency
def normal_pdf(x, mean, sigma):
    return (
        np.exp(-0.5 * ((x - mean) / sigma) ** 2)
        / (np.sqrt(2 * np.pi) * sigma)
    )

# ------------------------------------------------------------
# Create the three-panel figure
# ------------------------------------------------------------
fig, axes = plt.subplots(
    1, 3,
    figsize=(16, 4.8),
    constrained_layout=True
)

# ============================================================
# Panel 1: One galaxy
# ============================================================
ax = axes[0]

x1 = np.linspace(-0.8, 0.8, 1000)
p1 = normal_pdf(x1, gamma, sigma_e)

ax.plot(
    x1, p1,
    color="tab:blue",
    linewidth=2.5,
    label=r"$p(\epsilon_{\rm obs}\mid\gamma=0.02)$"
)
ax.fill_between(x1, p1, color="tab:blue", alpha=0.15)

# Mark one example observed galaxy
ax.axvline(
    epsilon_obs,
    color="black",
    linestyle="--",
    linewidth=2,
    label=fr"One galaxy: $\epsilon_{{\rm obs}}={epsilon_obs:.2f}$"
)

# Show several possible decompositions of the same observation
decomposition_text = (
    r"$\epsilon_{\rm obs}=\epsilon_{\rm int}+\gamma$" "\n\n"
    r"$0.20=0.18+0.02$" "\n"
    r"$0.20=0.20+0.00$" "\n"
    r"$0.20=0.23-0.03$"
)

ax.text(
    -0.74, 1.40,
    decomposition_text,
    fontsize=11,
    verticalalignment="top",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        edgecolor="0.6",
        alpha=0.9
    )
)

ax.set_title("A. One galaxy: degeneracy", fontsize=13)
ax.set_xlabel(r"Observed ellipticity component $\epsilon_{\rm obs}$")
ax.set_ylabel("Probability density")
ax.set_xlim(-0.8, 0.8)
ax.set_ylim(0, 1.75)
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.2)

# ============================================================
# Panel 2: Single-galaxy distributions
# ============================================================
ax = axes[1]

x2 = np.linspace(-0.8, 0.8, 1000)

# No-shear and sheared populations
p_no_shear = normal_pdf(x2, 0.0, sigma_e)
p_with_shear = normal_pdf(x2, gamma, sigma_e)

ax.plot(
    x2, p_no_shear,
    color="0.35",
    linewidth=2.5,
    label=r"No shear: $\mathcal{N}(0,\,0.26^2)$"
)
ax.plot(
    x2, p_with_shear,
    color="tab:blue",
    linewidth=2.5,
    label=r"With shear: $\mathcal{N}(0.02,\,0.26^2)$"
)

ax.fill_between(x2, p_no_shear, color="0.5", alpha=0.10)
ax.fill_between(x2, p_with_shear, color="tab:blue", alpha=0.10)

# Mark the two means
ax.axvline(0.0, color="0.35", linestyle=":", linewidth=1.5)
ax.axvline(gamma, color="tab:blue", linestyle=":", linewidth=1.5)

# Arrow showing the tiny shift in the population mean
arrow_y = 1.68
ax.annotate(
    "",
    xy=(gamma, arrow_y),
    xytext=(0, arrow_y),
    arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.5)
)
ax.text(
    gamma / 2,
    arrow_y + 0.03,
    r"$\gamma=0.02$",
    ha="center",
    va="bottom",
    fontsize=11
)

ax.text(
    0.33, 1.28,
    r"$\gamma/\sigma_\epsilon$"
    "\n"
    r"$=0.02/0.26$"
    "\n"
    r"$\approx 0.077$",
    ha="center",
    fontsize=11,
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        edgecolor="0.6",
        alpha=0.9
    )
)

ax.set_title("B. Single-galaxy populations", fontsize=13)
ax.set_xlabel(r"Observed ellipticity component $\epsilon_{\rm obs}$")
ax.set_ylabel("Probability density")
ax.set_xlim(-0.8, 0.8)
ax.set_ylim(0, 1.85)
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.2)

# ============================================================
# Panel 3: Distribution of the sample mean
# ============================================================
ax = axes[2]

x3 = np.linspace(-0.35, 0.35, 2000)
sample_sizes = [1, 100, 1000]
colors = ["tab:red", "tab:orange", "tab:green"]

for N, color in zip(sample_sizes, colors):
    sigma_mean = sigma_e / np.sqrt(N)
    p_mean = normal_pdf(x3, gamma, sigma_mean)

    ax.plot(
        x3, p_mean,
        color=color,
        linewidth=2.5,
        label=(
            fr"$N={N}$: "
            fr"$\sigma_{{\bar{{\epsilon}}}}={sigma_mean:.3f}$"
        )
    )

# Reference lines
ax.axvline(
    0,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label="No-shear expectation"
)
ax.axvline(
    gamma,
    color="tab:blue",
    linestyle=":",
    linewidth=2,
    label=fr"True shear $\gamma={gamma:.2f}$"
)

ax.annotate(
    r"All distributions remain centered"
    "\n"
    r"on the common shear $\gamma$",
    xy=(gamma, 44),
    xytext=(0.10, 42),
    arrowprops=dict(
        arrowstyle="->",
        color="tab:blue",
        linewidth=1.3
    ),
    fontsize=10,
    ha="left"
)

ax.set_title(r"C. Averaging $N$ galaxies", fontsize=13)
ax.set_xlabel(r"Sample mean $\bar{\epsilon}_{\rm obs}$")
ax.set_ylabel("Probability density")
ax.set_xlim(-0.35, 0.35)
ax.set_ylim(0, 52)
ax.legend(loc="upper left", fontsize=8.5)
ax.grid(alpha=0.2)

# ------------------------------------------------------------
# Add panel labels and save
# ------------------------------------------------------------
fig.suptitle(
    "How averaging suppresses shape noise",
    fontsize=16,
    fontweight="bold"
)

plt.savefig(
    "shape_noise_three_panels.pdf",
    bbox_inches="tight"
)
plt.savefig(
    "shape_noise_three_panels.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()