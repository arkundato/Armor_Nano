import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
import os
import numpy as np
import glob
from datetime import datetime

# ============================================================================
# @rtoto@rkundato
# ============================================================================
rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'legend.title_fontsize': 13,
    'axes.linewidth': 1.5,
    'axes.grid': True,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'grid.color': '#cccccc',
    'grid.linestyle': '--',
    'grid.linewidth': 0.8,
    'grid.alpha': 0.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.top': True,
    'ytick.right': True,
    'lines.linewidth': 2.0,
    'lines.markersize': 7,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})

# ============================================================================
# LOGGER
# ============================================================================
class Logger:
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'w', encoding='utf-8')
        self.write_header()

    def write_header(self):
        header = f"""{'='*80}
MULTI-SPEED PENETRATION ANALYSIS - DATA ENTRY AND RESULTS
{'='*80}
Tanggal/Time Analisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
DOKUMENTASI VARIABEL FISIS:
{'='*80}
1. PENETRATION DEPTH (P.D.)
   Definisi: Jarak antara titik masuk (onset interaksi: |Δv/v0| >= 1%) dan titik berhenti.
   Unit: Ångström (Å)

2. PROJECTILE MASS
   Rumus: m = N_atoms × m_atomic × 1.66054×10⁻²⁷ kg/amu
   Unit: kg atau amu
{'='*80}
"""
        self.file.write(header)

    def log(self, message, console=True):
        self.file.write(message + '\n')
        self.file.flush()
        if console:
            print(message)

    def close(self):
        self.file.write(f"\n{'='*80}\n")
        self.file.write(f"Analisis selesai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.file.write(f"{'='*80}\n")
        self.file.close()

# ============================================================================
# INISIALISASI
# ============================================================================
logger = Logger('data_entry.dat')
logger.log("=" * 60)
logger.log("MULTI-SPEED PENETRATION ANALYSIS")
logger.log("=" * 60)

data_files = glob.glob('*.txt')
files_config = []

colors = [
    '#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e',
    '#17becf', '#e377c2', '#8c564b', '#7f7f7f', '#bcbd22',
]
linestyles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

for idx, file in enumerate(data_files):
    file_name = os.path.splitext(os.path.basename(file))[0].replace('penetration_analysis_', '')
    files_config.append({
        'file': file,
        'name': file_name,
        'color': colors[idx % len(colors)],
        'linestyle': linestyles[idx % len(linestyles)],
        'marker': markers[idx % len(markers)]
    })

logger.log(f"\nFile data terdeteksi: {len(data_files)}")
for conf in files_config:
    logger.log(f"  - {conf['name']}: {conf['file']}")

# ============================================================================
# INPUT PARAMETER PROJECTILE
# ============================================================================
logger.log("\n" + "=" * 60)
logger.log("INPUT PARAMETER Projectile")
logger.log("=" * 60)

allowed_projectiles = ['Fe', 'W', 'Ti']
atomic_masses = {'Fe': 55.845, 'W': 183.84, 'Ti': 47.867}

projectile_name = input("Masukkan elemen Projectile (Fe, W, Ti) [default: Fe]: ") or "Fe"
projectile_name = projectile_name.capitalize()
if projectile_name not in allowed_projectiles:
    logger.log("Elemen tidak valid. Menggunakan default 'Fe'.")
    projectile_name = "Fe"
logger.log(f"1. Elemen Projectile: {projectile_name}")

num_atoms_input = input("Masukkan jumlah atom dalam Projectile [default: 1]: ") or "1"
try:
    num_atoms = int(num_atoms_input)
    if num_atoms <= 0:
        raise ValueError
except ValueError:
    logger.log("Input tidak valid. Menggunakan default 1 atom.")
    num_atoms = 1

atomic_mass_amu = atomic_masses[projectile_name]
projectile_mass_amu = num_atoms * atomic_mass_amu
projectile_mass_kg = projectile_mass_amu * 1.66054e-27
mass_unit = "amu (total)"

logger.log(f"2. Jumlah Atom: {num_atoms}")
logger.log(f"3. Massa Total: {projectile_mass_amu:.4e} amu = {projectile_mass_kg:.4e} kg")

logger.log("\nPilihan bentuk Projectile:")
logger.log("1. Spherical")
logger.log("2. Pointed Cylinder")
shape_choice = input("Pilih bentuk [1=Spherical, 2=Pointed Cylinder, default=1]: ").strip() or "1"
projectile_shape = "Pointed Cylinder" if shape_choice == "2" else "Spherical"
logger.log(f"4. Bentuk: {projectile_shape}")

# ============================================================================
# INPUT PARAMETER TARGET
# ============================================================================
logger.log("\n" + "=" * 60)
logger.log("INPUT PARAMETER TARGET")
logger.log("=" * 60)

armor_thickness_input = input("Masukkan ketebalan armor (Å) untuk garis back armor [skip jika kosong]: ")
try:
    armor_thickness = float(armor_thickness_input)
    has_armor_thickness = True
    logger.log(f"1. Ketebalan Armor: {armor_thickness} Å")
except ValueError:
    has_armor_thickness = False
    armor_thickness = None

# ============================================================================
# INPUT OPSI TITIK BERHENTI
# ============================================================================
logger.log("\n" + "=" * 60)
logger.log("INPUT OPSI TITIK BERHENTI")
logger.log("=" * 60)

stop_mode = input("Pilih opsi [1=Otomatis, 2=Manual, default=1]: ").strip() or "1"
manual_stop_time = None
if stop_mode == "2":
    manual_time_input = input("Masukkan Time titik berhenti (ps) dari titik masuk: ")
    try:
        manual_stop_time = float(manual_time_input)
        if manual_stop_time <= 0:
            stop_mode = "1"
            manual_stop_time = None
    except ValueError:
        stop_mode = "1"
        manual_stop_time = None

# ============================================================================
# INPUT MODE ANALISIS
# ============================================================================
logger.log("\n" + "=" * 60)
logger.log("PILIHAN MODE ANALISIS")
logger.log("=" * 60)
logger.log("1. Multi-input plotting (tampilkan semua file)")
logger.log("2. Rata-rata Penetration Depth (hitung mean + error)")

analysis_mode = input("Pilih mode [1/2, default=1]: ").strip() or "1"
if analysis_mode not in ["1", "2"]:
    analysis_mode = "1"

logger.log(f"Mode terpilih: {'Multi-input plotting' if analysis_mode == '1' else 'Rata-rata Penetration Depth'}")

# ============================================================================
# INPUT PARAMETER PLOT
# ============================================================================
logger.log("\n" + "=" * 60)
logger.log("INPUT PARAMETER PLOT")
logger.log("=" * 60)

time_range_input = input("Masukkan Time maksimum untuk plot Speed vs Time (ps) [default: full range]: ")
max_time_plot = float(time_range_input) if time_range_input.strip() else None

time_range_z_input = input("Masukkan Time maksimum untuk plot Z Position vs Time (ps) [default: full range]: ")
max_time_z_plot = float(time_range_z_input) if time_range_z_input.strip() else None

user_input_label = input("Masukkan label kustom (max 40 karakter) [default: kosong]: ")
if len(user_input_label) > 40:
    user_input_label = user_input_label[:40]

output_dir = "comparison_plots"
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# FUNGSI PEMROSESAN DATA
# ============================================================================
def process_data_file(filepath, manual_stop_time=None):
    try:
        df = pd.read_csv(filepath, skiprows=1, sep=r'\s+',
                         names=['step', 'time', 'com_z', 'vz_proj', 'speed_proj', 'penetration_depth'],
                         engine='python')
        logger.log(f"  Jumlah step dibaca: {len(df)}, Step terakhir: {df['step'].iloc[-1]}")

        if df.empty:
            return None

        v0 = df['speed_proj'].iloc[0]
        delta = np.abs((df['speed_proj'] - v0) / v0)
        entry_cond = delta >= 0.01
        entry_idx = entry_cond.idxmax() if entry_cond.any() else 0
        z_entry = df['com_z'].iloc[entry_idx]
        time_entry = df['time'].iloc[entry_idx]

        df_post = df.iloc[entry_idx:].copy().reset_index(drop=True)
        df_post['time_normalized'] = df_post['time'] - time_entry
        df_post['z_normalized'] = z_entry - df_post['com_z']

        if manual_stop_time is not None:
            time_diff = np.abs(df_post['time_normalized'] - manual_stop_time)
            stop_idx = time_diff.idxmin()
            stop_found = True
        else:
            end_percent = df_post['speed_proj'] / v0
            stop_cond = end_percent < 0.01
            if stop_cond.any():
                stop_idx = stop_cond.idxmax()
                tail_end = min(stop_idx + 5, len(df_post))
                std_tail = df_post['speed_proj'].iloc[stop_idx:tail_end].std()
                if std_tail > 0.001 * v0:
                    stop_idx = len(df_post) - 1
                stop_found = df_post['speed_proj'].iloc[stop_idx] < 0.01 * v0
            else:
                stop_idx = len(df_post) - 1
                stop_found = False

        z_stop = df_post['com_z'].iloc[stop_idx]
        time_stop = df_post['time'].iloc[stop_idx]
        penetration_depth = np.abs(z_stop - z_entry)
        residual_velocity = df_post['speed_proj'].iloc[stop_idx]

        return {
            'df': df_post,
            'z_entry': z_entry,
            'time_entry': time_entry,
            'z_stop': z_stop,
            'time_stop': time_stop,
            'penetration_depth': penetration_depth,
            'stop_found': stop_found,
            'initial_speed': v0,
            'residual_velocity': residual_velocity
        }
    except Exception as e:
        logger.log(f"ERROR memproses {filepath}: {str(e)}")
        return None

# ============================================================================
# PROSES DATA FILE
# ============================================================================
logger.log("\n" + "=" * 60)
logger.log("MEMPROSES DATA FILE")
logger.log("=" * 60)

data_processed = {}
for conf in files_config:
    if os.path.exists(conf['file']):
        logger.log(f"Memproses {conf['name']}...")
        data = process_data_file(conf['file'], manual_stop_time=manual_stop_time)
        if data is not None:
            data_processed[conf['name']] = {**data, **conf}

if not data_processed:
    logger.log("ERROR: Tidak ada file data valid!")
    logger.close()
    exit(1)

logger.log(f"\nTotal file berhasil diproses: {len(data_processed)}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_sorted_items(data_dict, key='penetration_depth', ascending=False):
    items = list(data_dict.items())
    items.sort(key=lambda x: x[1].get(key, 0), reverse=not ascending)
    return items

def add_info_box(ax, text, loc='upper left'):
    x = 0.03 if 'left' in loc else 0.97
    ha = 'left' if 'left' in loc else 'right'
    y = 0.97 if 'upper' in loc else 0.03
    va = 'top' if 'upper' in loc else 'bottom'
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=12, va=va, ha=ha,
            bbox=dict(boxstyle='square,pad=0.4', facecolor='white',
                      edgecolor='black', linewidth=1.2, alpha=0.92))

fig_size = (8, 6)

# ============================================================================
# MODE 1: MULTI-INPUT PLOTTING
# ============================================================================
if analysis_mode == "1":
    logger.log("\n" + "=" * 60)
    logger.log("MEMBUAT PLOT MULTI-INPUT (KUALITAS PUBLIKASI)")
    logger.log("=" * 60)

    sorted_items = get_sorted_items(data_processed, key='penetration_depth', ascending=False)

    # --- PLOT 1: Speed vs Time ---
    fig, ax = plt.subplots(figsize=fig_size)
    for name, data in sorted_items:
        df_plot = data['df'].copy()
        if max_time_plot is not None:
            df_plot = df_plot[df_plot['time_normalized'] <= max_time_plot]
        if not df_plot.empty:
            pd_val = data['penetration_depth']
            label = f"{name} (P.D. = {pd_val:.1f} Å)"
            n = len(df_plot)
            ax.plot(df_plot['time_normalized'], df_plot['speed_proj'],
                    label=label, linewidth=2.0,
                    color=data['color'], linestyle=data['linestyle'],
                    marker=data['marker'], markevery=max(1, n // 15),
                    markersize=7, alpha=0.9)

    ax.set_xlabel('Time from Contact (ps)', fontsize=16)
    ax.set_ylabel('Speed (Å ps$^{-1}$)', fontsize=16)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which='minor', direction='in', length=3, width=1.0, top=True, right=True)

    info_text = f'{projectile_name} – {projectile_shape}'
    if user_input_label:
        info_text += f'\n{user_input_label}'
    add_info_box(ax, info_text, loc='upper right')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.75),
              frameon=True, edgecolor='black', fancybox=False, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_speed_vs_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.log("✓ Plot Speed vs Time saved")

    # --- PLOT 2: Penetration Depth vs Time ---
    fig, ax = plt.subplots(figsize=fig_size)
    for name, data in sorted_items:
        df_plot = data['df'].copy()
        if max_time_z_plot is not None:
            df_plot = df_plot[df_plot['time_normalized'] <= max_time_z_plot]
        if not df_plot.empty:
            pd_val = data['penetration_depth']
            label = f"{name} (P.D. = {pd_val:.1f} Å)"
            n = len(df_plot)
            ax.plot(df_plot['time_normalized'], df_plot['z_normalized'],
                    label=label, linewidth=2.0,
                    color=data['color'], linestyle=data['linestyle'],
                    marker=data['marker'], markevery=max(1, n // 15),
                    markersize=7, alpha=0.9)

    ax.set_xlabel('Time from Contact (ps)', fontsize=16)
    ax.set_ylabel('Penetration Depth (Å)', fontsize=16)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which='minor', direction='in', length=3, width=1.0, top=True, right=True)

    info_text = f'{projectile_name} – {projectile_shape}'
    if user_input_label:
        info_text += f'\n{user_input_label}'
    add_info_box(ax, info_text, loc='upper left')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower right',
              frameon=True, edgecolor='black', fancybox=False, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_penetration_vs_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.log("✓ Plot Penetration Depth vs Time saved")

    # --- PLOT 3: Speed vs Penetration Depth ---
    fig, ax = plt.subplots(figsize=fig_size)
    time_limit = None
    if max_time_plot is not None and max_time_z_plot is not None:
        time_limit = min(max_time_plot, max_time_z_plot)
    elif max_time_plot is not None:
        time_limit = max_time_plot
    elif max_time_z_plot is not None:
        time_limit = max_time_z_plot

    for name, data in sorted_items:
        df_plot = data['df'].copy()
        if time_limit is not None:
            df_plot = df_plot[df_plot['time_normalized'] <= time_limit]
        if not df_plot.empty:
            pd_val = data['penetration_depth']
            label = f"{name} (P.D. = {pd_val:.1f} Å)"
            n = len(df_plot)
            ax.plot(df_plot['z_normalized'], df_plot['speed_proj'],
                    label=label, linewidth=2.0,
                    color=data['color'], linestyle=data['linestyle'],
                    marker=data['marker'], markevery=max(1, n // 15),
                    markersize=7, alpha=0.9)

    ax.axvline(x=0, color='navy', linestyle='--', linewidth=1.5, alpha=0.8, label='Entry Point')
    if has_armor_thickness:
        ax.axvline(x=armor_thickness, color='crimson', linestyle='--', linewidth=1.5, alpha=0.8,
                   label=f'Back Armor ({armor_thickness:.1f} Å)')

    ax.set_xlabel('Penetration Depth (Å)', fontsize=16)
    ax.set_ylabel('Speed (Å ps$^{-1}$)', fontsize=16)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which='minor', direction='in', length=3, width=1.0, top=True, right=True)

    info_text = f'{projectile_name} – {projectile_shape}'
    if user_input_label:
        info_text += f'\n{user_input_label}'
    add_info_box(ax, info_text, loc='upper right')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.80),
              frameon=True, edgecolor='black', fancybox=False, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_speed_vs_penetration.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.log("✓ Plot Speed vs Penetration Depth saved")

    # --- PLOT 4: Bar Chart ---
    fig, ax = plt.subplots(figsize=fig_size)
    sorted_bar = sorted(data_processed.items(), key=lambda x: x[1]['penetration_depth'], reverse=True)
    names_bar = [item[0] for item in sorted_bar]
    pds_bar = [item[1]['penetration_depth'] for item in sorted_bar]
    colors_bar = [item[1]['color'] for item in sorted_bar]

    bars = ax.bar(names_bar, pds_bar, color=colors_bar, alpha=0.80,
                  edgecolor='black', linewidth=1.5, width=0.6)
    for bar, pd_val in zip(bars, pds_bar):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + max(pds_bar) * 0.01,
                f'{pd_val:.1f} Å', ha='center', va='bottom', fontsize=12, fontweight='bold')

    info_text = f'{projectile_name} – {projectile_shape}'
    if user_input_label:
        info_text += f'\n{user_input_label}'
    add_info_box(ax, info_text, loc='upper right')

    ax.set_xlabel('Speed / File', fontsize=16)
    ax.set_ylabel('Penetration Depth (Å)', fontsize=16)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which='minor', direction='in', length=3, width=1.0)
    ax.set_ylim(0, max(pds_bar) * 1.15)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_penetration_depth_bar.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.log("✓ Plot bar Penetration Depth saved")

# ============================================================================
# MODE 2: RATA-RATA PENETRATION DEPTH
# ============================================================================
elif analysis_mode == "2":
    logger.log("\n" + "=" * 60)
    logger.log("PERHITUNGAN RATA-RATA PENETRATION DEPTH")
    logger.log("=" * 60)

    n_files = len(data_processed)
    pd_values = np.array([d['penetration_depth'] for d in data_processed.values()])
    res_vel_values = np.array([d['residual_velocity'] for d in data_processed.values()])
    init_speed_values = np.array([d['initial_speed'] for d in data_processed.values()])
    stop_time_values = np.array([
        d['time_stop'] - d['time_entry'] if d['stop_found'] else np.nan
        for d in data_processed.values()
    ])

    pd_mean = np.mean(pd_values)
    pd_std = np.std(pd_values, ddof=1) if n_files > 1 else 0.0
    pd_sem = pd_std / np.sqrt(n_files) if n_files > 1 else 0.0
    pd_pct = (pd_std / pd_mean * 100) if pd_mean != 0 else 0.0
    pd_sem_pct = (pd_sem / pd_mean * 100) if pd_mean != 0 else 0.0

    res_vel_mean = np.mean(res_vel_values)
    res_vel_std = np.std(res_vel_values, ddof=1) if n_files > 1 else 0.0
    res_vel_pct = (res_vel_std / res_vel_mean * 100) if res_vel_mean != 0 else 0.0

    init_speed_mean = np.mean(init_speed_values)
    init_speed_std = np.std(init_speed_values, ddof=1) if n_files > 1 else 0.0
    init_speed_pct = (init_speed_std / init_speed_mean * 100) if init_speed_mean != 0 else 0.0

    valid_stop = stop_time_values[~np.isnan(stop_time_values)]
    stop_mean = np.mean(valid_stop) if len(valid_stop) > 0 else np.nan
    stop_std = np.std(valid_stop, ddof=1) if len(valid_stop) > 1 else 0.0
    stop_pct = (stop_std / stop_mean * 100) if (not np.isnan(stop_mean) and stop_mean != 0) else 0.0

    logger.log(f"\nJumlah file (N): {n_files}")
    logger.log("-" * 105)
    logger.log(f"{'File':<20} {'Init Speed (Å/ps)':<20} {'P.D. (Å)':<15} {'Res. Vel.':<15} {'Stop Time (ps)':<15}")
    logger.log("-" * 105)
    for name, data in data_processed.items():
        st = f"{data['time_stop'] - data['time_entry']:.2f}" if data['stop_found'] else "N/A"
        logger.log(f"{name:<20} {data['initial_speed']:<20.4f} {data['penetration_depth']:<15.2f} "
                   f"{data['residual_velocity']:<15.4f} {st:<15}")
    logger.log("-" * 105)
    logger.log(f"{'MEAN':<20} {init_speed_mean:<20.4f} {pd_mean:<15.2f} {res_vel_mean:<15.4f} "
               f"{stop_mean:<15.2f}" if not np.isnan(stop_mean) else
               f"{'MEAN':<20} {init_speed_mean:<20.4f} {pd_mean:<15.2f} {res_vel_mean:<15.4f} {'N/A':<15}")
    logger.log(f"{'STD (σ)':<20} {init_speed_std:<20.4f} {pd_std:<15.2f} {res_vel_std:<15.4f} "
               f"{stop_std:<15.2f}" if not np.isnan(stop_mean) else
               f"{'STD (σ)':<20} {init_speed_std:<20.4f} {pd_std:<15.2f} {res_vel_std:<15.4f} {'N/A':<15}")
    logger.log(f"{'Error % (σ/mean)':<20} {init_speed_pct:<20.2f} {pd_pct:<15.2f} {res_vel_pct:<15.2f} "
               f"{stop_pct:<15.2f}" if not np.isnan(stop_mean) else
               f"{'Error % (σ/mean)':<20} {init_speed_pct:<20.2f} {pd_pct:<15.2f} {res_vel_pct:<15.2f} {'N/A':<15}")
    logger.log(f"{'SEM (σ/√N)':<20} {'—':<20} {pd_sem:<15.2f} {'—':<15} {'—':<15}")
    logger.log(f"{'SEM % (SEM/mean)':<20} {'—':<20} {f'{pd_sem_pct:.2f}':<15} {'—':<15} {'—':<15}")
    logger.log("-" * 105)

    logger.log(f"\n>>> Penetration Depth = {pd_mean:.2f} ± {pd_std:.2f} Å (mean ± σ) [{pd_pct:.2f}%]")
    logger.log(f">>> Penetration Depth = {pd_mean:.2f} ± {pd_sem:.2f} Å (mean ± SEM) [{pd_sem_pct:.2f}%]")

    # --- PLOT: Combined curves with errorbar (mean ± std) ---
    logger.log("\nMEMBUAT PLOT GABUNGAN DENGAN ERRORBAR...")

    # Interpolate all data onto common time grid for averaging
    all_dfs = []
    max_common_time = np.inf
    for name, data in data_processed.items():
        df_tmp = data['df'].copy()
        t_max = df_tmp['time_normalized'].max()
        if t_max < max_common_time:
            max_common_time = t_max
        all_dfs.append(df_tmp)

    if max_time_plot is not None:
        max_common_time = min(max_common_time, max_time_plot)
    if max_time_z_plot is not None:
        max_common_time_z = min(max_common_time, max_time_z_plot)
    else:
        max_common_time_z = max_common_time

    n_points = 200
    common_time = np.linspace(0, max_common_time, n_points)
    common_time_z = np.linspace(0, max_common_time_z, n_points)

    # Interpolate speed
    interp_speeds = np.zeros((n_files, n_points))
    for i, (name, data) in enumerate(data_processed.items()):
        df_tmp = data['df']
        interp_speeds[i] = np.interp(common_time, df_tmp['time_normalized'], df_tmp['speed_proj'])

    speed_mean = np.mean(interp_speeds, axis=0)
    speed_std = np.std(interp_speeds, axis=0, ddof=1) if n_files > 1 else np.zeros(n_points)

    # Interpolate penetration depth (z_normalized)
    interp_z = np.zeros((n_files, n_points))
    for i, (name, data) in enumerate(data_processed.items()):
        df_tmp = data['df']
        interp_z[i] = np.interp(common_time_z, df_tmp['time_normalized'], df_tmp['z_normalized'])

    z_mean = np.mean(interp_z, axis=0)
    z_std = np.std(interp_z, axis=0, ddof=1) if n_files > 1 else np.zeros(n_points)

    avg_color = '#1f77b4'

    # --- PLOT A: Average Speed vs Time with errorbar ---
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(common_time, speed_mean, color=avg_color, linewidth=2.5, label='Mean', zorder=3)
    ax.fill_between(common_time, speed_mean - speed_std, speed_mean + speed_std,
                    color=avg_color, alpha=0.25, label=f'± 1σ (N={n_files})', zorder=2)

    # Errorbar at selected points
    n_err = 12
    err_idx = np.linspace(0, n_points - 1, n_err, dtype=int)
    ax.errorbar(common_time[err_idx], speed_mean[err_idx], yerr=speed_std[err_idx],
                fmt='none', ecolor='black', elinewidth=1.5, capsize=4, capthick=1.5, zorder=4)

    ax.set_xlabel('Time from Contact (ps)', fontsize=16)
    ax.set_ylabel('Speed (Å ps$^{-1}$)', fontsize=16)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which='minor', direction='in', length=3, width=1.0, top=True, right=True)

    info_text = f'{projectile_name} – {projectile_shape}\nN = {n_files}'
    if user_input_label:
        info_text += f'\n{user_input_label}'
    add_info_box(ax, info_text, loc='upper right')

    pd_box = f'P.D. = {pd_mean:.2f} ± {pd_std:.2f} Å ({pd_pct:.1f}%)'
    add_info_box(ax, pd_box, loc='lower left')

    ax.legend(loc='center right', frameon=True, edgecolor='black', fancybox=False, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/average_speed_vs_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.log("✓ Plot Average Speed vs Time saved")

    # --- PLOT B: Average Penetration Depth vs Time with errorbar ---
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(common_time_z, z_mean, color=avg_color, linewidth=2.5, label='Mean', zorder=3)
    ax.fill_between(common_time_z, z_mean - z_std, z_mean + z_std,
                    color=avg_color, alpha=0.25, label=f'± 1σ (N={n_files})', zorder=2)

    err_idx_z = np.linspace(0, n_points - 1, n_err, dtype=int)
    ax.errorbar(common_time_z[err_idx_z], z_mean[err_idx_z], yerr=z_std[err_idx_z],
                fmt='none', ecolor='black', elinewidth=1.5, capsize=4, capthick=1.5, zorder=4)

    ax.set_xlabel('Time from Contact (ps)', fontsize=16)
    ax.set_ylabel('Penetration Depth (Å)', fontsize=16)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which='minor', direction='in', length=3, width=1.0, top=True, right=True)

    info_text = f'{projectile_name} – {projectile_shape}\nN = {n_files}'
    if user_input_label:
        info_text += f'\n{user_input_label}'
    add_info_box(ax, info_text, loc='upper left')

    # Add final PD result box
    pd_box = f'P.D. = {pd_mean:.2f} ± {pd_std:.2f} Å ({pd_pct:.1f}%)'
    add_info_box(ax, pd_box, loc='lower right')

    ax.legend(loc='center left', bbox_to_anchor=(0.03, 0.55),
              frameon=True, edgecolor='black', fancybox=False, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/average_penetration_vs_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.log("✓ Plot Average Penetration Depth vs Time saved")

    # --- PLOT C: Average Speed vs Penetration Depth with errorbar ---
    # Interpolate speed as function of penetration depth
    max_common_z = np.inf
    for name, data in data_processed.items():
        df_tmp = data['df']
        z_max = df_tmp['z_normalized'].max()
        if z_max < max_common_z:
            max_common_z = z_max

    common_z = np.linspace(0, max_common_z, n_points)
    interp_speed_vs_z = np.zeros((n_files, n_points))
    for i, (name, data) in enumerate(data_processed.items()):
        df_tmp = data['df'].sort_values('z_normalized')
        interp_speed_vs_z[i] = np.interp(common_z, df_tmp['z_normalized'], df_tmp['speed_proj'])

    svz_mean = np.mean(interp_speed_vs_z, axis=0)
    svz_std = np.std(interp_speed_vs_z, axis=0, ddof=1) if n_files > 1 else np.zeros(n_points)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(common_z, svz_mean, color=avg_color, linewidth=2.5, label='Mean', zorder=3)
    ax.fill_between(common_z, svz_mean - svz_std, svz_mean + svz_std,
                    color=avg_color, alpha=0.25, label=f'± 1σ (N={n_files})', zorder=2)

    err_idx_svz = np.linspace(0, n_points - 1, n_err, dtype=int)
    ax.errorbar(common_z[err_idx_svz], svz_mean[err_idx_svz], yerr=svz_std[err_idx_svz],
                fmt='none', ecolor='black', elinewidth=1.5, capsize=4, capthick=1.5, zorder=4)

    ax.axvline(x=0, color='navy', linestyle='--', linewidth=1.5, alpha=0.8, label='Entry Point')
    if has_armor_thickness:
        ax.axvline(x=armor_thickness, color='crimson', linestyle='--', linewidth=1.5, alpha=0.8,
                   label=f'Back Armor ({armor_thickness:.1f} Å)')

    ax.set_xlabel('Penetration Depth (Å)', fontsize=16)
    ax.set_ylabel('Speed (Å ps$^{-1}$)', fontsize=16)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which='minor', direction='in', length=3, width=1.0, top=True, right=True)

    info_text = f'{projectile_name} – {projectile_shape}\nN = {n_files}'
    if user_input_label:
        info_text += f'\n{user_input_label}'
    add_info_box(ax, info_text, loc='upper right')

    pd_box_c = f'P.D. = {pd_mean:.2f} ± {pd_std:.2f} Å ({pd_pct:.1f}%)'
    add_info_box(ax, pd_box_c, loc='lower left')

    ax.legend(loc='upper right', bbox_to_anchor=(0.99, 0.75),
              frameon=True, edgecolor='black', fancybox=False, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/average_speed_vs_penetration.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.log("✓ Plot Average Speed vs Penetration Depth saved")

    # --- PLOT D: Bar chart of individual PD values + mean with errorbar ---
    fig, ax = plt.subplots(figsize=fig_size)
    sorted_bar = sorted(data_processed.items(), key=lambda x: x[1]['penetration_depth'], reverse=True)
    names_bar = [item[0] for item in sorted_bar]
    pds_bar = [item[1]['penetration_depth'] for item in sorted_bar]
    colors_bar = [item[1]['color'] for item in sorted_bar]

    # Individual bars
    x_pos = np.arange(len(names_bar))
    bars = ax.bar(x_pos, pds_bar, color=colors_bar, alpha=0.80,
                  edgecolor='black', linewidth=1.5, width=0.6)
    for bar, pd_val in zip(bars, pds_bar):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + max(pds_bar) * 0.01,
                f'{pd_val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Mean bar with errorbar
    mean_x = len(names_bar)
    ax.bar(mean_x, pd_mean, color='#555555', alpha=0.85,
           edgecolor='black', linewidth=1.5, width=0.6,
           yerr=pd_std, capsize=6, ecolor='black', error_kw={'linewidth': 2, 'capthick': 2})
    ax.text(mean_x, pd_mean + pd_std + max(pds_bar) * 0.01,
            f'{pd_mean:.1f}±{pd_std:.1f} ({pd_pct:.1f}%)', ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='#333333')

    all_names = names_bar + ['Mean']
    ax.set_xticks(list(x_pos) + [mean_x])
    ax.set_xticklabels(all_names, rotation=30, ha='right')

    ax.set_xlabel('Run', fontsize=16)
    ax.set_ylabel('Penetration Depth (Å)', fontsize=16)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which='minor', direction='in', length=3, width=1.0)
    ax.set_ylim(0, max(max(pds_bar), pd_mean + pd_std) * 1.20)

    info_text = f'{projectile_name} – {projectile_shape}'
    if user_input_label:
        info_text += f'\n{user_input_label}'
    add_info_box(ax, info_text, loc='upper right')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/average_penetration_depth_bar.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.log("✓ Plot bar Penetration Depth (with mean) saved")

# ============================================================================
# SUMMARY TABLE
# ============================================================================
logger.log("\n" + "=" * 120)
logger.log("RINGKASAN KOMPREHENSIF")
logger.log("=" * 120)
logger.log(f"Projectile: {projectile_name} ({projectile_shape})")
logger.log(f"Massa: {projectile_mass_amu:.4e} {mass_unit} ({projectile_mass_kg:.4e} kg)")
logger.log(f"Mode: {'Multi-input plotting' if analysis_mode == '1' else 'Rata-rata Penetration Depth'}")
logger.log("-" * 120)
logger.log(f"{'File/Speed':<20} {'P.D. (Å)':<12} {'Res. Vel. (Å/ps)':<18} {'Stop Time (ps)':<15}")
logger.log("-" * 120)

for name, data in sorted(data_processed.items(), key=lambda x: x[1]['penetration_depth'], reverse=True):
    pd_str = f"{data['penetration_depth']:.2f}"
    res_vel_str = f"{data['residual_velocity']:.3f}"
    stop_time_str = f"{data['time_stop'] - data['time_entry']:.2f}" if data['stop_found'] else "Tidak tercapai"
    logger.log(f"{name:<20} {pd_str:<12} {res_vel_str:<18} {stop_time_str:<15}")

if analysis_mode == "2":
    logger.log("-" * 120)
    logger.log(f"{'MEAN ± STD':<20} {pd_mean:.2f} ± {pd_std:.2f} Å ({pd_pct:.2f}%)")
    logger.log(f"{'MEAN ± SEM':<20} {pd_mean:.2f} ± {pd_sem:.2f} Å ({pd_sem_pct:.2f}%)")

logger.log("=" * 120)
logger.log(f"\nSemua plot disimpan di direktori '{output_dir}'")
logger.close()

print(f"\n{'='*60}")
print(f"✓ ANALISIS SELESAI")
print(f"{'='*60}")
print(f"Plot: {output_dir}/")
print(f"Log : data_entry.dat")
print(f"{'='*60}")
