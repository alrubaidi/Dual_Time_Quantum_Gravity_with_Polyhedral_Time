"""
Generate Publication Figures for Dual-Time Quantum Gravity

This script generates all figures referenced in the paper.
Figures are saved to validation_suite/outputs/figures/

Figure List:
- Fig 1: Polyhedral structure (dodecahedron-icosahedron duality)
- Fig 2: Synchronization Landau-Ginzburg potential
- Fig 3: Coupled Kairos-Chronos evolution
- Fig 4: Decoherence rates: standard vs Φ-modified
- Fig 5: Cosmological epoch transitions
- Fig 6: 64-mode E8 correspondence
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from polyhedral_dynamics import (
    PolyhedralComplex, dodecahedron_adjacency, icosahedron_adjacency,
    N_DODECA_FACES, N_ICOSA_FACES
)
from synchronization import SynchronizationParameter
from decoherence import diosi_penrose_rate, modified_decoherence_rate

# Set style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs', 'figures')


def ensure_output_dir():
    """Ensure output directory exists."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def fig1_polyhedral_structure():
    """
    Figure 1: Polyhedral Time Complex Structure
    
    Shows dodecahedron (Chronos) and icosahedron (Kairos) with duality.
    """
    print("Generating Figure 1: Polyhedral Structure...")
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # (a) Adjacency matrices
    ax = axes[0]
    adj_d = dodecahedron_adjacency()
    adj_i = icosahedron_adjacency()
    
    # Combined view
    combined = np.zeros((32, 32))
    combined[:12, :12] = adj_d
    combined[12:, 12:] = adj_i
    
    im = ax.imshow(combined, cmap='Blues')
    ax.axhline(y=11.5, color='red', linewidth=2, linestyle='--')
    ax.axvline(x=11.5, color='red', linewidth=2, linestyle='--')
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Mode index')
    ax.set_title('(a) Combined adjacency structure')
    ax.text(5, 25, 'Dodeca\n(Chronos)', ha='center', color='white', fontsize=10)
    ax.text(21, 5, 'Icosa\n(Kairos)', ha='center', color='white', fontsize=10)
    
    # (b) Mode counting
    ax = axes[1]
    categories = ['Dodeca\nfaces', 'Dodeca\nvertices', 'Icosa\nfaces', 'Icosa\nvertices', 'Edges']
    values = [12, 20, 20, 12, 30]
    colors = ['#3498db', '#3498db', '#e74c3c', '#e74c3c', '#2ecc71']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title('(b) Polyhedral elements')
    
    # Add total
    ax.axhline(y=32, color='purple', linestyle='--', linewidth=2)
    ax.text(4.5, 33, 'Total: 32+32=64', fontsize=11, color='purple')
    
    # (c) Duality diagram
    ax = axes[2]
    ax.axis('off')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.5, 1.5)
    
    # Dodecahedron box (left)
    dodeca = plt.Rectangle((-1.8, -0.6), 1.4, 1.2, fill=True, 
                           facecolor='#3498db', alpha=0.3, edgecolor='#2980b9', linewidth=2)
    ax.add_patch(dodeca)
    ax.text(-1.1, 0, 'DODECAHEDRON\n12 faces\n20 vertices\nChronos', 
            ha='center', va='center', fontsize=10)
    
    # Icosahedron box (right)
    icosa = plt.Rectangle((0.4, -0.6), 1.4, 1.2, fill=True,
                          facecolor='#e74c3c', alpha=0.3, edgecolor='#c0392b', linewidth=2)
    ax.add_patch(icosa)
    ax.text(1.1, 0, 'ICOSAHEDRON\n20 faces\n12 vertices\nKairos',
            ha='center', va='center', fontsize=10)
    
    # Duality arrows
    ax.annotate('', xy=(0.3, 0.3), xytext=(-0.3, 0.3),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax.text(0, 0.5, 'Dual\n(V↔F)', ha='center', fontsize=10, color='purple')
    
    ax.annotate('', xy=(0.3, -0.3), xytext=(-0.3, -0.3),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax.text(0, -0.5, 'Shared\n30 edges', ha='center', fontsize=10, color='green')
    
    ax.set_title('(c) Duality structure')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_polyhedral_structure.pdf'), 
                bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_polyhedral_structure.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    print("  ✓ Saved fig1_polyhedral_structure.pdf/png")


def fig2_landau_ginzburg():
    """
    Figure 2: Landau-Ginzburg Effective Potential
    
    Shows V_eff(Φ) for different values of the control parameter μ².
    """
    print("Generating Figure 2: Landau-Ginzburg Potential...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # (a) Effective potential for different μ²
    ax = axes[0]
    Phi = np.linspace(-0.2, 1.5, 500)
    
    mu2_values = [-0.5, 0, 0.5, 1.0]
    lambda4 = 1.0
    colors = plt.cm.RdYlBu(np.linspace(0.8, 0.2, len(mu2_values)))
    
    for mu2, color in zip(mu2_values, colors):
        V = -0.5 * mu2 * Phi**2 + 0.25 * lambda4 * Phi**4
        ax.plot(Phi, V, color=color, linewidth=2, label=f'$\\mu^2 = {mu2}$')
    
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_xlabel('$\\Phi$')
    ax.set_ylabel('$V_{\\mathrm{eff}}(\\Phi)$')
    ax.set_title('(a) Effective potential')
    ax.legend()
    ax.set_xlim(-0.2, 1.5)
    ax.set_ylim(-0.5, 0.5)
    
    # (b) Phase diagram
    ax = axes[1]
    
    mu2_range = np.linspace(-1, 2, 100)
    Phi_star = np.zeros_like(mu2_range)
    Phi_star[mu2_range > 0] = np.sqrt(mu2_range[mu2_range > 0])
    
    ax.fill_between(mu2_range[mu2_range <= 0], 0, 1.5, alpha=0.3, color='blue',
                    label='Desynchronized (quantum)')
    ax.fill_between(mu2_range[mu2_range > 0], Phi_star[mu2_range > 0], 1.5, 
                    alpha=0.3, color='red', label='Synchronized (classical)')
    
    ax.plot(mu2_range, Phi_star, 'k-', linewidth=3, label='$\\Phi_* = \\sqrt{\\mu^2/\\lambda}$')
    ax.axvline(x=0, color='purple', linewidth=2, linestyle='--', label='Critical point')
    
    ax.set_xlabel('Control parameter $\\mu^2$')
    ax.set_ylabel('Order parameter $\\Phi_*$')
    ax.set_title('(b) Phase diagram')
    ax.legend(loc='upper left')
    ax.set_xlim(-1, 2)
    ax.set_ylim(0, 1.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_landau_ginzburg.pdf'),
                bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_landau_ginzburg.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    print("  ✓ Saved fig2_landau_ginzburg.pdf/png")


def fig3_coupled_evolution():
    """
    Figure 3: Coupled Kairos-Chronos Evolution
    
    Shows purity and synchronization during polyhedral dynamics.
    """
    print("Generating Figure 3: Coupled Evolution...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Run simulation with measurement at step 250
    complex = PolyhedralComplex(initial_purity=1.0)
    
    n_steps = 500
    dt = 0.1
    measure_step = 250
    
    purity_hist = []
    Phi_hist = []
    kairos_hist = []
    chronos_diag_hist = []
    
    for n in range(n_steps):
        measure = (n == measure_step)
        purity, Phi = complex.step(dt, measure=measure, measure_strength=0.5)
        purity_hist.append(purity)
        Phi_hist.append(Phi)
        kairos_hist.append(complex.g_kairos.copy())
        chronos_diag_hist.append(np.diag(complex.rho_chronos).real.copy())
    
    purity_hist = np.array(purity_hist)
    Phi_hist = np.array(Phi_hist)
    kairos_hist = np.array(kairos_hist)
    chronos_diag_hist = np.array(chronos_diag_hist)
    t = np.arange(n_steps) * dt
    
    # (a) Purity evolution
    ax = axes[0, 0]
    ax.plot(t, purity_hist, 'b-', linewidth=2)
    ax.axvline(x=measure_step * dt, color='red', linestyle='--', 
               label=f'Measurement at $t = {measure_step * dt}$')
    ax.set_xlabel('Kairos time $t_r$')
    ax.set_ylabel('Purity $\\mathrm{Tr}(\\rho^2)$')
    ax.set_title('(a) Chronos state purity')
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # (b) Synchronization parameter
    ax = axes[0, 1]
    ax.plot(t, Phi_hist, 'purple', linewidth=2)
    ax.axvline(x=measure_step * dt, color='red', linestyle='--')
    ax.axhline(y=1.0, color='green', linestyle=':', label='$\\Phi = 1$ (fully synchronized)')
    ax.set_xlabel('Kairos time $t_r$')
    ax.set_ylabel('$\\Phi$')
    ax.set_title('(b) Synchronization parameter')
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # (c) Kairos modes (selected)
    ax = axes[1, 0]
    for a in range(5):
        ax.plot(t, np.abs(kairos_hist[:, a])**2, alpha=0.7, 
                linewidth=1.5, label=f'Mode {a}')
    ax.axvline(x=measure_step * dt, color='red', linestyle='--')
    ax.set_xlabel('Kairos time $t_r$')
    ax.set_ylabel('$|g_a|^2$')
    ax.set_title('(c) Kairos mode amplitudes')
    ax.legend(ncol=2, fontsize=9)
    
    # (d) Chronos face populations
    ax = axes[1, 1]
    im = ax.imshow(chronos_diag_hist.T, aspect='auto', cmap='viridis',
                   extent=[0, t[-1], 0, 12])
    ax.axvline(x=measure_step * dt, color='red', linestyle='--')
    ax.set_xlabel('Kairos time $t_r$')
    ax.set_ylabel('Chronos face index')
    ax.set_title('(d) Chronos face populations')
    plt.colorbar(im, ax=ax, label='Population')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_coupled_evolution.pdf'),
                bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_coupled_evolution.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    print("  ✓ Saved fig3_coupled_evolution.pdf/png")


def fig4_decoherence_rates():
    """
    Figure 4: Gravitational Decoherence Rates
    
    Compares Diósi-Penrose rates with Φ-modified rates.
    """
    print("Generating Figure 4: Decoherence Rates...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Physical parameters
    M_range = np.logspace(-15, -9, 100)  # Mass range (kg)
    R = 1e-6  # 1 μm radius
    Delta_x = 1e-7  # 100 nm superposition
    
    # (a) Decoherence rate vs mass for different Φ
    ax = axes[0]
    
    Gamma_DP = np.array([diosi_penrose_rate(M, R, Delta_x) for M in M_range])
    
    Phi_values = [1.0, 0.5, 0.1, 0.01]
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(Phi_values)))
    
    for Phi, color in zip(Phi_values, colors):
        Gamma_mod = np.array([modified_decoherence_rate(M, R, Delta_x, Phi) 
                              for M in M_range])
        ax.loglog(M_range, Gamma_mod, color=color, linewidth=2,
                  label=f'$\\Phi = {Phi}$')
    
    ax.set_xlabel('Mass $M$ (kg)')
    ax.set_ylabel('Decoherence rate $\\Gamma$ (s$^{-1}$)')
    ax.set_title('(a) $\\Phi$-modified decoherence rates')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (b) Φ dependence for fixed mass
    ax = axes[1]
    
    Phi_range = np.linspace(0.01, 1.0, 100)
    M_fixed = 1e-12  # 1 pg
    
    Gamma_standard = diosi_penrose_rate(M_fixed, R, Delta_x)
    
    for p in [0.5, 1.0, 2.0]:
        Gamma = np.array([modified_decoherence_rate(M_fixed, R, Delta_x, Phi, p=p)
                          for Phi in Phi_range])
        ax.semilogy(Phi_range, Gamma, linewidth=2, label=f'$p = {p}$')
    
    ax.axhline(y=Gamma_standard, color='gray', linestyle='--', 
               label='Diósi-Penrose')
    ax.axvline(x=1.0, color='green', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Synchronization $\\Phi$')
    ax.set_ylabel('Decoherence rate $\\Gamma$ (s$^{-1}$)')
    ax.set_title(f'(b) $\\Phi$ dependence ($M = 10^{{-12}}$ kg)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_decoherence_rates.pdf'),
                bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_decoherence_rates.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    print("  ✓ Saved fig4_decoherence_rates.pdf/png")


def fig5_cosmology():
    """
    Figure 5: Cosmological Epoch Transitions
    
    Shows evolution of scale factor and synchronization parameter.
    """
    print("Generating Figure 5: Cosmological Epochs...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Simplified cosmological model
    # Use log scale factor: N = ln(a)
    N = np.linspace(-40, 0, 1000)  # From inflation to today
    a = np.exp(N)
    
    # Model Φ evolution through epochs
    # Φ starts low (quantum), increases through reheating, stabilizes
    def Phi_cosmo(N):
        """Model synchronization through cosmological history."""
        # Inflation: Φ small
        # Reheating: Φ increases rapidly
        # Radiation/matter: Φ ≈ 1
        
        N_reheat = -35
        width = 2
        Phi = 0.5 * (1 + np.tanh((N - N_reheat) / width))
        return 0.1 + 0.9 * Phi
    
    Phi_hist = Phi_cosmo(N)
    
    # (a) Scale factor and Φ
    ax = axes[0]
    
    ax.semilogy(N, a, 'b-', linewidth=2, label='Scale factor $a$')
    ax.set_xlabel('$\\ln(a)$')
    ax.set_ylabel('$a$ (normalized)', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax.twinx()
    ax2.plot(N, Phi_hist, 'r-', linewidth=2, label='$\\Phi$')
    ax2.set_ylabel('$\\Phi$', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 1.1)
    
    # Mark epochs
    epochs = [
        (-38, 'Inflation'),
        (-35, 'Reheating'),
        (-25, 'Radiation'),
        (-8, 'Matter'),
        (0, 'Today'),
    ]
    for N_epoch, label in epochs:
        ax.axvline(x=N_epoch, color='gray', linestyle=':', alpha=0.5)
        ax.text(N_epoch, 1e-40, label, rotation=90, va='bottom', fontsize=9)
    
    ax.set_title('(a) Cosmological evolution of $\\Phi$')
    
    # (b) Hubble rate modification
    ax = axes[1]
    
    H0 = 67.4  # km/s/Mpc
    Omega_m = 0.315
    Omega_Lambda = 0.685
    
    # Standard Friedmann
    H_std = H0 * np.sqrt(Omega_m * a**(-3) + Omega_Lambda)
    
    # Modified with Φ-dependent dark energy
    def Lambda_sync(Phi, Lambda_0=1.0, Delta_Lambda=0.1):
        return Lambda_0 + Delta_Lambda * (1 - Phi)**2
    
    Lambda_ratio = Lambda_sync(Phi_hist) / Lambda_sync(1.0)
    H_mod = H0 * np.sqrt(Omega_m * a**(-3) + Omega_Lambda * Lambda_ratio)
    
    N_late = N > -10
    ax.plot(N[N_late], H_std[N_late], 'b--', linewidth=2, label='Standard $\\Lambda$CDM')
    ax.plot(N[N_late], H_mod[N_late], 'r-', linewidth=2, label='$\\Phi$-modified')
    
    ax.set_xlabel('$\\ln(a)$')
    ax.set_ylabel('$H(a)$ (km/s/Mpc)')
    ax.set_title('(b) Late-time Hubble rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_cosmology.pdf'),
                bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_cosmology.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    print("  ✓ Saved fig5_cosmology.pdf/png")


def fig6_e8_correspondence():
    """
    Figure 6: 64-Mode E8 Correspondence
    
    Shows connection between polyhedral modes and E8 structure.
    """
    print("Generating Figure 6: E8 Correspondence...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # (a) Mode decomposition
    ax = axes[0]
    
    # Polyhedral mode structure
    modes = {
        'Dodeca vertices': (20, '#3498db'),
        'Dodeca faces': (12, '#2980b9'),
        'Icosa vertices': (12, '#e74c3c'),
        'Icosa faces': (20, '#c0392b'),
    }
    
    labels = list(modes.keys())
    sizes = [m[0] for m in modes.values()]
    colors = [m[1] for m in modes.values()]
    
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%d',
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.5, edgecolor='white')
    )
    
    ax.text(0, 0, '64', fontsize=24, fontweight='bold', ha='center', va='center')
    ax.set_title('(a) Polyhedral mode count')
    
    # (b) E8 structure diagram
    ax = axes[1]
    ax.axis('off')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.5, 1.5)
    
    # E8 box
    e8_box = plt.Rectangle((-1.8, -1.2), 3.6, 2.4, fill=True,
                            facecolor='#9b59b6', alpha=0.2, 
                            edgecolor='#8e44ad', linewidth=2)
    ax.add_patch(e8_box)
    ax.text(0, 1.0, 'E₈ Root System', fontsize=14, fontweight='bold',
            ha='center', color='#8e44ad')
    
    # Structure
    ax.text(0, 0.5, '240 roots', fontsize=12, ha='center')
    ax.text(0, 0.1, '120 + 120 = positive + negative', fontsize=10, ha='center')
    ax.text(0, -0.3, '64 stabilizers ↔ 64 polyhedral modes', fontsize=11, 
            ha='center', color='#27ae60', fontweight='bold')
    
    # Connection arrow
    ax.annotate('', xy=(0, -0.7), xytext=(0, -0.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Lower box
    poly_box = plt.Rectangle((-1.2, -1.1), 2.4, 0.4, fill=True,
                              facecolor='#2ecc71', alpha=0.3,
                              edgecolor='#27ae60', linewidth=2)
    ax.add_patch(poly_box)
    ax.text(0, -0.9, '32 (dodeca) + 32 (icosa) = 64', fontsize=10, ha='center')
    
    ax.set_title('(b) E8 quantum error correction connection')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_e8_correspondence.pdf'),
                bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_e8_correspondence.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    print("  ✓ Saved fig6_e8_correspondence.pdf/png")


def main():
    """Generate all publication figures."""
    print("=" * 60)
    print("Generating Publication Figures")
    print("Dual-Time Quantum Gravity with Polyhedral Time")
    print("=" * 60)
    
    ensure_output_dir()
    
    fig1_polyhedral_structure()
    fig2_landau_ginzburg()
    fig3_coupled_evolution()
    fig4_decoherence_rates()
    fig5_cosmology()
    fig6_e8_correspondence()
    
    print("=" * 60)
    print("All figures generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
