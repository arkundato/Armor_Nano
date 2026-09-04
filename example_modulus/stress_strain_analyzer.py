import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
import os
import sys
import glob
import atexit
from datetime import datetime

# =============@rtoto@rkundato

DEFAULT_ELASTIC_LIMIT = 0.02   # 2% — batas default region elastis
OUTLIER_R2_THRESH     = 0.90   # R² minimum supaya data dianggap layak
OUTLIER_ZSCORE_THRESH = 2.0    # Z-score |E - E_mean| / E_std untuk flag outlier
OUTLIER_IQR_FENCE     = 1.5    # IQR fence multiplier (Tukey's rule: 1.5=mild, 3.0=extreme)
OUTLIER_CORR_THRESH   = 0.80   # Korelasi Pearson minimum vs median kurva

# ============================================================
# PARAMETER ERROR BAR
# ============================================================
N_ERRORBAR_POINTS = 15   # jumlah titik strain tempat error bar ditampilkan
ERRORBAR_CAPSIZE  = 4    # ukuran cap error bar (pt)
ERRORBAR_CAPTHICK = 1.5  # ketebalan cap error bar
ERRORBAR_ELINEWIDTH = 1.8  # ketebalan garis error bar


# ============================================================
# SISTEM LOGGING LAYAR KE FILE screen.txt
# ============================================================
class ScreenLogger:
    """

    """

    def __init__(self, filepath="screen.txt"):
        self.filepath      = filepath
        self._orig_stdout  = None
        self._orig_stderr  = None
        self._log_file     = None
        self._active       = False

    # ── Objek yang menggantikan sys.stdout / sys.stderr ──────────────
    class _Tee:
        """Meneruskan write() ke dua stream sekaligus."""
        def __init__(self, original_stream, log_file):
            self._orig = original_stream
            self._log  = log_file

        def write(self, text):
            self._orig.write(text)
            self._orig.flush()
            try:
                self._log.write(text)
                self._log.flush()
            except Exception:
                pass

        def flush(self):
            self._orig.flush()
            try:
                self._log.flush()
            except Exception:
                pass

        # Teruskan atribut lain (misal .encoding) ke stream asli
        def __getattr__(self, name):
            return getattr(self._orig, name)

    # ── start / stop ─────────────────────────────────────────────────
    def start(self):
        if self._active:
            return
        self._log_file    = open(self.filepath, "w", encoding="utf-8",
                                 buffering=1)
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = self._Tee(self._orig_stdout, self._log_file)
        sys.stderr = self._Tee(self._orig_stderr, self._log_file)
        self._active = True

        # Tulis header di awal file
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        header = (
            "=" * 65 + "\n"
            f"  LOG LAYAR — Analisis Young's Modulus\n"
            f"  Waktu mulai : {ts}\n"
            "=" * 65 + "\n\n"
        )
        self._log_file.write(header)
        self._log_file.flush()

    def stop(self):
        if not self._active:
            return

        # Tulis footer sebelum menutup
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        footer = (
            "\n" + "=" * 65 + "\n"
            f"  LOG SELESAI — {ts}\n"
            "=" * 65 + "\n"
        )
        try:
            self._log_file.write(footer)
            self._log_file.flush()
        except Exception:
            pass

        # Kembalikan stdout / stderr ke semula
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr

        try:
            self._log_file.close()
        except Exception:
            pass

        self._active = False
        print(f"\n  ✓ Semua output layar telah disimpan ke: {self.filepath}")


# Instance global — diinisiasi di main(), diregister ke atexit
_screen_logger = ScreenLogger("screen.txt")


# ============================================================
# STYLE GRAFIK STANDAR PAPER (Origin-style)
# ============================================================
def set_paper_style():
    plt.rcParams.update({
        'figure.facecolor':    'white',
        'figure.edgecolor':    'white',
        'axes.facecolor':      'white',
        'axes.edgecolor':      'black',
        'axes.linewidth':      1.5,
        'axes.labelcolor':     'black',
        'axes.labelweight':    'bold',
        'axes.labelsize':      14,
        'axes.grid':           True,
        'axes.axisbelow':      True,
        'grid.color':          '#CCCCCC',
        'grid.linestyle':      '--',
        'grid.linewidth':      0.6,
        'grid.alpha':          0.7,
        'xtick.color':         'black',
        'ytick.color':         'black',
        'xtick.labelsize':     13,
        'ytick.labelsize':     13,
        'xtick.direction':     'in',
        'ytick.direction':     'in',
        'xtick.major.width':   1.5,
        'ytick.major.width':   1.5,
        'xtick.minor.width':   1.0,
        'ytick.minor.width':   1.0,
        'xtick.major.size':    6,
        'ytick.major.size':    6,
        'xtick.minor.size':    3,
        'ytick.minor.size':    3,
        'xtick.top':           True,
        'ytick.right':         True,
        'legend.frameon':      True,
        'legend.framealpha':   0.0,
        'legend.facecolor':    'white',
        'legend.edgecolor':    'black',
        'legend.fontsize':     11,
        'lines.linewidth':     1.8,
        'font.family':         'serif',
        'font.weight':         'bold',
        'font.size':           13,
        'text.color':          'black',
        'savefig.facecolor':   'white',
        'savefig.edgecolor':   'white',
        'savefig.dpi':         300,
        'savefig.bbox':        'tight',
    })


# ============================================================
# DETEKSI DAN PEMILIHAN FILE
# ============================================================
def detect_data_files():
    files = glob.glob("*.txt") + glob.glob("*.dat") + glob.glob("*.csv")
    # Jangan masukkan screen.txt ke dalam daftar file data
    files = [f for f in files if f.lower() != "screen.txt"]
    return sorted(set(files))


def select_input_files():
    print("\n" + "=" * 65)
    print("  PILIH FILE DATA STRESS-STRAIN (MULTI-KONFIGURASI)")
    print("=" * 65)

    available = detect_data_files()

    if not available:
        print("\n  ⚠  Tidak ada file .txt/.dat/.csv terdeteksi di folder ini.")
        raw = input("  Masukkan nama file (pisahkan koma jika >1): ").strip()
        return [f.strip() for f in raw.split(",") if f.strip()]

    print("\n  File yang terdeteksi:")
    for i, f in enumerate(available, 1):
        try:
            size = os.path.getsize(f) / 1024
            print(f"    [{i:2d}] {f}  ({size:.2f} KB)")
        except Exception:
            print(f"    [{i:2d}] {f}")

    print(f"\n  [A] Gunakan SEMUA file di atas")
    print(f"  [M] Pilih manual (masukkan nomor, pisahkan koma)\n")

    choice = input("  Pilih [A/M] (Enter = semua): ").strip().upper()

    if choice == "" or choice == "A":
        print(f"\n  ✓ Menggunakan semua {len(available)} file.")
        return available

    # Pilih manual berdasarkan nomor
    raw = input("  Masukkan nomor file (contoh: 1,3,4): ").strip()
    selected = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(available):
                selected.append(available[idx])
    if not selected:
        print("  ⚠  Tidak ada file valid dipilih. Menggunakan semua.")
        return available
    print(f"\n  ✓ File dipilih: {selected}")
    return selected


# ============================================================
# LOAD SATU FILE
# ============================================================
def load_single_file(filename):
    try:
        data = np.loadtxt(filename, skiprows=1)
    except FileNotFoundError:
        print(f"  ✗ File '{filename}' tidak ditemukan.")
        return None
    except Exception as e:
        print(f"  ✗ Error membaca '{filename}': {e}")
        return None

    if data.ndim != 2 or data.shape[1] != 7:
        print(f"  ✗ '{filename}' tidak memiliki 7 kolom — dilewati.")
        return None

    return data


# ============================================================
# CARI ELASTIC LIMIT OTOMATIS (sama seperti script asli)
# ============================================================
def find_best_elastic_limit(strain_x, stress_xx):
    candidates = np.arange(0.003, 0.051, 0.001)
    results = []

    for lim in candidates:
        mask = (strain_x > 0.0005) & (strain_x <= lim) & (stress_xx > 0)
        s_el = strain_x[mask]
        sig_el = stress_xx[mask]
        if len(s_el) < 5:
            continue
        slope, intercept, r_val, *_ = stats.linregress(s_el, sig_el)
        r2 = r_val ** 2
        results.append((lim, slope, intercept, r2, len(s_el)))

    if not results:
        return DEFAULT_ELASTIC_LIMIT

    results_by_r2 = sorted(results, key=lambda x: x[3], reverse=True)
    best = None
    for r in results_by_r2:
        if r[3] >= 0.99:
            if best is None or r[4] > best[4]:
                best = r
    if best is None:
        for r in results_by_r2:
            if r[3] >= 0.95:
                best = r
                break
    if best is None:
        best = results_by_r2[0]

    return best[0]


# ============================================================
# PILIH ELASTIC LIMIT (menu ringkas untuk mode multi-file)
# ============================================================
def select_elastic_limit_multi(strain_x_ref, stress_xx_ref):
    """
    Elastic limit dipilih sekali dari file pertama,
    lalu diterapkan ke semua file supaya perbandingan konsisten.
    """
    print(f"\n{'=' * 65}")
    print(f"  PENENTUAN BATAS REGION ELASTIS")
    print(f"  (diterapkan seragam ke semua konfigurasi)")
    print(f"{'=' * 65}")
    print(f"\n  [1] Otomatis — maksimalkan R² (dari file pertama)")
    print(f"  [2] Default  — {DEFAULT_ELASTIC_LIMIT*100:.1f}%")
    print(f"  [3] Manual   — masukkan nilai sendiri (%)\n")

    choice = input("  Pilih [1/2/3] (Enter = default [2]): ").strip()

    if choice == "1":
        lim = find_best_elastic_limit(strain_x_ref, stress_xx_ref)
        print(f"  ✓ Elastic limit otomatis: {lim*100:.2f}%")
        return lim, f"Otomatis (R² optimal)"
    elif choice == "3":
        while True:
            val = input("  Masukkan nilai (dalam %): ").strip()
            try:
                lim = float(val) / 100.0
                if 0.001 <= lim <= 0.20:
                    print(f"  ✓ Elastic limit manual: {lim*100:.2f}%")
                    return lim, f"Manual ({lim*100:.2f}%)"
                else:
                    print("  ⚠  Harus antara 0.1%–20%.")
            except ValueError:
                print("  ⚠  Input tidak valid.")
    else:
        print(f"  ✓ Elastic limit default: {DEFAULT_ELASTIC_LIMIT*100:.1f}%")
        return DEFAULT_ELASTIC_LIMIT, f"Default ({DEFAULT_ELASTIC_LIMIT*100:.1f}%)"


# ============================================================
# HITUNG MODULUS SATU KONFIGURASI
# ============================================================
def calc_modulus(strain_x, stress_xx, elastic_limit):
    mask = (strain_x > 0.0005) & (strain_x <= elastic_limit) & (stress_xx > 0)
    s_el   = strain_x[mask]
    sig_el = stress_xx[mask]

    if len(s_el) < 5:
        return None, None, None, None, None

    slope, intercept, r_val, *_ = stats.linregress(s_el, sig_el)
    r2 = r_val ** 2
    return slope, intercept, r2, s_el, sig_el


# ============================================================
# INTERPOLASI KE GRID STRAIN SERAGAM
# ============================================================
def interpolate_to_grid(strain_x, stress_xx, grid):
    """Interpolasi linier supaya semua kurva punya titik strain yang sama."""
    valid = (strain_x >= grid[0]) & (strain_x <= grid[-1])
    if np.sum(valid) < 2:
        return np.full_like(grid, np.nan)
    return np.interp(grid, strain_x[valid], stress_xx[valid],
                     left=np.nan, right=np.nan)


# ============================================================
# DETEKSI OUTLIER — 4 KRITERIA
# ============================================================
def detect_outliers(config_list, strain_grid, stress_matrix):
    """
    Kriteria flag sebagai SUSPECT (salah satu terpenuhi):
      1. R² fit elastis < OUTLIER_R2_THRESH
      2. |Z-score(E)| > OUTLIER_ZSCORE_THRESH
      3. E di luar Tukey IQR fence  (lebih robust untuk n kecil)
      4. Korelasi Pearson kurva vs median < OUTLIER_CORR_THRESH
    """
    n = len(config_list)
    flags = [False] * n

    median_curve = np.nanmedian(stress_matrix, axis=0)

    E_vals = np.array([c['E'] for c in config_list], dtype=float)
    E_mean = np.mean(E_vals)
    E_std  = np.std(E_vals, ddof=1) if n > 1 else 0.0

    Q1, Q3   = np.percentile(E_vals, [25, 75])
    IQR      = Q3 - Q1
    iqr_lo   = Q1 - OUTLIER_IQR_FENCE * IQR
    iqr_hi   = Q3 + OUTLIER_IQR_FENCE * IQR

    print(f"\n{'=' * 65}")
    print(f"  DETEKSI OUTLIER")
    print(f"{'=' * 65}")
    print(f"\n  Statistik E ensemble:")
    print(f"    Mean  : {E_mean:.2f} GPa")
    print(f"    Std   : {E_std:.2f} GPa")
    print(f"    Q1/Q3 : {Q1:.2f} / {Q3:.2f} GPa  (IQR = {IQR:.2f})")
    print(f"    Fence : [{iqr_lo:.2f}, {iqr_hi:.2f}] GPa  "
          f"(Tukey ×{OUTLIER_IQR_FENCE})")
    print(f"\n  Thresholds aktif:")
    print(f"    R²    < {OUTLIER_R2_THRESH}  → data noisy")
    print(f"    Z     > {OUTLIER_ZSCORE_THRESH}  → jauh dari mean (std-based)")
    print(f"    IQR   luar fence    → jauh dari median (robust)")
    print(f"    Corr  < {OUTLIER_CORR_THRESH}  → bentuk kurva berbeda")
    print(f"\n  {'No':>3}  {'File':<28}  {'E':>7}  {'Z':>6}  "
          f"{'IQR':>5}  {'R²':>8}  {'Corr':>7}  Status")
    print(f"  {'-'*3}  {'-'*28}  {'-'*7}  {'-'*6}  "
          f"{'-'*5}  {'-'*8}  {'-'*7}  {'-'*10}")

    reasons = []

    for i, cfg in enumerate(config_list):
        E  = cfg['E']
        r2 = cfg['r2']
        z  = abs(E - E_mean) / E_std if E_std > 1e-6 else 0.0
        outside_iqr = (E < iqr_lo) or (E > iqr_hi)

        row = stress_matrix[i]
        valid_idx = ~np.isnan(row) & ~np.isnan(median_curve)
        if np.sum(valid_idx) > 5:
            corr, _ = stats.pearsonr(row[valid_idx], median_curve[valid_idx])
        else:
            corr = 1.0

        flag_reasons = []
        if r2 < OUTLIER_R2_THRESH:
            flag_reasons.append(f"R²={r2:.4f}<{OUTLIER_R2_THRESH}")
        if z > OUTLIER_ZSCORE_THRESH:
            flag_reasons.append(f"Z={z:.2f}>{OUTLIER_ZSCORE_THRESH}")
        if outside_iqr:
            direction = "tinggi" if E > iqr_hi else "rendah"
            flag_reasons.append(f"IQR-fence ({direction})")
        if corr < OUTLIER_CORR_THRESH:
            flag_reasons.append(f"corr={corr:.3f}<{OUTLIER_CORR_THRESH}")

        is_outlier = len(flag_reasons) > 0
        flags[i]   = is_outlier
        reasons.append(flag_reasons)

        iqr_mark = "OUT" if outside_iqr else "  ok"
        status   = "SUSPECT" if is_outlier else "OK"
        name     = cfg['name'][:28]
        print(f"  {i+1:>3}  {name:<28}  {E:>7.2f}  {z:>6.2f}  "
              f"{iqr_mark:>5}  {r2:>8.5f}  {corr:>7.4f}  {status}")
        if flag_reasons:
            print(f"       Alasan: {', '.join(flag_reasons)}")

    n_flagged = sum(flags)
    print(f"\n  Ringkasan: {n - n_flagged} OK, {n_flagged} SUSPECT")
    return flags, reasons


# ============================================================
# TANYA USER: BUANG ATAU PERTAHANKAN OUTLIER
# ============================================================
def handle_outliers(config_list, flags, reasons):
    include = [True] * len(config_list)
    n_flagged = sum(flags)

    if n_flagged == 0:
        print(f"\n  ✓ Tidak ada outlier terdeteksi. Semua data digunakan.")
        return include

    print(f"\n{'=' * 65}")
    print(f"  PENANGANAN DATA SUSPECT ({n_flagged} file)")
    print(f"{'=' * 65}")
    print(f"\n  Opsi tersedia untuk setiap file suspect:")
    print(f"    [K] Pertahankan — ikut dirata-rata (default)")
    print(f"    [B] Buang       — exclude dari rata-rata dan plot ensemble\n")

    for i, cfg in enumerate(config_list):
        if not flags[i]:
            continue

        print(f"  File  : {cfg['name']}")
        print(f"  E     : {cfg['E']:.2f} GPa")
        print(f"  Alasan: {', '.join(reasons[i])}")
        ans = input("  Keputusan [K/B] (Enter = K): ").strip().upper()
        if ans == "B":
            include[i] = False
            print(f"  → '{cfg['name']}' DIBUANG dari ensemble.\n")
        else:
            include[i] = True
            print(f"  → '{cfg['name']}' DIPERTAHANKAN.\n")

    n_kept   = sum(include)
    n_dropped = len(include) - n_kept
    print(f"  Ringkasan: {n_kept} file digunakan, {n_dropped} dibuang.")
    return include


# ============================================================
# BOOTSTRAP ENSEMBLE
# ============================================================
def bootstrap_ensemble_modulus(E_list, n_bootstrap=5000, ci=95):
    E_arr = np.array(E_list)
    n     = len(E_arr)
    if n < 2:
        return np.std(E_arr, ddof=1), E_arr[0], E_arr[0]

    rng = np.random.default_rng(seed=42)
    E_samples = np.array([
        np.mean(rng.choice(E_arr, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (100 - ci) / 2
    return (np.std(E_samples, ddof=1),
            np.percentile(E_samples, alpha),
            np.percentile(E_samples, 100 - alpha))


# ============================================================
# MENU PILIHAN JENIS GRAFIK
# ============================================================
def select_plot_mode():
    """
    Tanya user ingin mencetak grafik versi mana:
      [1] Lengkap tanpa error bar  — semua kurva berwarna-warni
      [2] Tunggal dengan error bar — hanya kurva mean, hitam-putih
      [3] Keduanya                 — simpan dua file sekaligus
    """
    print(f"\n{'=' * 65}")
    print(f"  PILIH JENIS GRAFIK YANG DICETAK")
    print(f"{'=' * 65}")
    print(f"\n  [1] Grafik LENGKAP tanpa error bar")
    print(f"      → Semua kurva individual (warna-warni) + mean + ±1σ band")
    print(f"  [2] Grafik TUNGGAL dengan error bar")
    print(f"      → Hanya kurva mean (hitam) + ±1σ band + error bar")
    print(f"  [3] KEDUANYA  (simpan dua file)")
    print(f"\n  (Enter = [3] keduanya)\n")

    choice = input("  Pilih [1/2/3]: ").strip()
    if choice == "1":
        print("  ✓ Mode: Grafik lengkap warna-warni tanpa error bar.")
        return "color"
    elif choice == "2":
        print("  ✓ Mode: Grafik tunggal hitam-putih dengan error bar.")
        return "bw"
    else:
        print("  ✓ Mode: Keduanya.")
        return "both"


# ============================================================
# HITUNG POSISI ERROR BAR YANG RELEVAN
# ============================================================
def compute_errorbar_positions(strain_grid, mean_curve, std_curve,
                                n_points=N_ERRORBAR_POINTS):
    """
    """
    n = len(strain_grid)
    if n < n_points:
        return np.arange(n)

    dx = np.diff(strain_grid * 100)
    dy = np.diff(mean_curve)
    arc = np.concatenate([[0], np.cumsum(np.sqrt(dx**2 + dy**2))])
    total_arc = arc[-1]

    target_arcs = np.linspace(0, total_arc, n_points)
    indices = []
    for ta in target_arcs:
        idx = np.argmin(np.abs(arc - ta))
        indices.append(idx)

    seen = set()
    unique_indices = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)

    return np.array(unique_indices)


# ============================================================
# HELPER: TEXT BOX & AXIS FORMATTING
# ============================================================
def _add_textbox(ax, E_mean, E_std_ens, E_ci_lo, E_ci_hi, r2_mean, n_used):
    lines = [
        f'$E$ = {E_mean:.2f} $\\pm$ {E_std_ens:.2f} GPa',
        f'95% CI  [{E_ci_lo:.2f}, {E_ci_hi:.2f}] GPa',
        f'$R^2$ = {r2_mean:.5f}',
        f'$n$ = {n_used} config(s)',
    ]
    props = dict(boxstyle='round,pad=0.55', facecolor='white',
                 edgecolor='black', linewidth=1.2, alpha=1.0)
    ax.text(0.97, 0.05, '\n'.join(lines),
            transform=ax.transAxes,
            fontsize=11.5, fontweight='bold', color='black',
            verticalalignment='bottom', horizontalalignment='right',
            bbox=props)


def _format_axis(ax):
    ax.set_xlabel('Strain (%)', fontsize=14, fontweight='bold', labelpad=8)
    ax.set_ylabel('Stress (GPa)', fontsize=14, fontweight='bold', labelpad=8)
    ax.tick_params(axis='both', which='major',
                   labelsize=13, width=1.5, length=6, direction='in')
    ax.tick_params(axis='both', which='minor',
                   width=1.0, length=3, direction='in')
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    leg = ax.legend(loc='upper left', fontsize=10,
                    frameon=True, framealpha=0.0, edgecolor='black')
    for text in leg.get_texts():
        text.set_fontweight('bold')


# ============================================================
# GRAFIK 1: LENGKAP WARNA-WARNI TANPA ERROR BAR
# ============================================================
def plot_ensemble_color(config_list, include, strain_grid, stress_matrix,
                        mean_curve, std_curve,
                        elastic_limit,
                        E_mean, E_std_ens, E_ci_lo, E_ci_hi,
                        r2_mean,
                        output_file='ensemble_color.png'):
    set_paper_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    strain_pct = strain_grid * 100
    n_used = sum(include)

    colors_used = plt.cm.tab10(np.linspace(0, 0.9, max(n_used, 1)))
    color_idx = 0
    for i, cfg in enumerate(config_list):
        row  = stress_matrix[i]
        name = os.path.splitext(cfg['name'])[0]
        if include[i]:
            ax.plot(strain_pct, row,
                    color=colors_used[color_idx], linewidth=1.4,
                    alpha=0.60, zorder=2,
                    label=f"{name}  (E={cfg['E']:.1f} GPa)")
            color_idx += 1
        else:
            ax.plot(strain_pct, row,
                    color='#BBBBBB', linewidth=1.0,
                    alpha=0.45, linestyle=':', zorder=1)

    ax.fill_between(strain_pct,
                    mean_curve - std_curve,
                    mean_curve + std_curve,
                    color='#888888', alpha=0.18, zorder=3,
                    label='±1σ band')

    ax.plot(strain_pct, mean_curve,
            color='black', linewidth=2.8, zorder=4,
            label='Ensemble mean')

    ax.axvline(elastic_limit * 100, color='#C62828',
               linewidth=1.2, linestyle='--', alpha=0.6,
               label=f'Elastic limit ({elastic_limit*100:.1f}%)')

    _add_textbox(ax, E_mean, E_std_ens, E_ci_lo, E_ci_hi, r2_mean, n_used)
    _format_axis(ax)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='white')
    print(f"  ✓ [Grafik 1] Warna-warni tanpa error bar → {output_file}")
    return fig, ax


# ============================================================
# GRAFIK 2: TUNGGAL HITAM-PUTIH DENGAN ERROR BAR
# ============================================================
def plot_ensemble_bw_errorbar(config_list, include, strain_grid, stress_matrix,
                               mean_curve, std_curve,
                               elastic_limit,
                               E_mean, E_std_ens, E_ci_lo, E_ci_hi,
                               r2_mean,
                               output_file='ensemble_errorbar.png'):
    set_paper_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    strain_pct = strain_grid * 100
    n_used = sum(include)

    ax.fill_between(strain_pct,
                    mean_curve - std_curve,
                    mean_curve + std_curve,
                    color='#AAAAAA', alpha=0.30, zorder=2,
                    label='±1σ band')

    ax.plot(strain_pct, mean_curve,
            color='black', linewidth=2.8, zorder=3,
            label='Ensemble mean')

    eb_idx    = compute_errorbar_positions(strain_grid, mean_curve,
                                           std_curve, n_points=N_ERRORBAR_POINTS)
    eb_strain = strain_pct[eb_idx]
    eb_mean   = mean_curve[eb_idx]
    eb_std    = std_curve[eb_idx]

    ax.errorbar(eb_strain, eb_mean,
                yerr=eb_std,
                fmt='none',
                ecolor='#B71C1C',
                elinewidth=ERRORBAR_ELINEWIDTH,
                capsize=ERRORBAR_CAPSIZE,
                capthick=ERRORBAR_CAPTHICK,
                alpha=0.90,
                zorder=4,
                label=f'Error bar ±1σ  (n={N_ERRORBAR_POINTS} pts)')

    ax.plot(eb_strain, eb_mean,
            'o', color='#B71C1C', markersize=4.0,
            alpha=0.90, zorder=5)

    ax.axvline(elastic_limit * 100, color='#333333',
               linewidth=1.2, linestyle='--', alpha=0.65,
               label=f'Elastic limit ({elastic_limit*100:.1f}%)')

    _add_textbox(ax, E_mean, E_std_ens, E_ci_lo, E_ci_hi, r2_mean, n_used)
    _format_axis(ax)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='white')
    print(f"  ✓ [Grafik 2] Tunggal hitam + error bar → {output_file}")
    return fig, ax


# ============================================================
# DISPATCHER: PANGGIL PLOT SESUAI MODE
# ============================================================
def plot_ensemble(config_list, include, strain_grid, stress_matrix,
                  mean_curve, std_curve,
                  elastic_limit,
                  E_mean, E_std_ens, E_ci_lo, E_ci_hi,
                  r2_mean,
                  plot_mode='both'):
    print(f"\n{'=' * 65}")
    print(f"  MENCETAK GRAFIK")
    print(f"{'=' * 65}\n")

    figs = []
    if plot_mode in ('color', 'both'):
        fig1, ax1 = plot_ensemble_color(
            config_list, include, strain_grid, stress_matrix,
            mean_curve, std_curve, elastic_limit,
            E_mean, E_std_ens, E_ci_lo, E_ci_hi, r2_mean,
            output_file='ensemble_color.png'
        )
        figs.append((fig1, ax1))

    if plot_mode in ('bw', 'both'):
        fig2, ax2 = plot_ensemble_bw_errorbar(
            config_list, include, strain_grid, stress_matrix,
            mean_curve, std_curve, elastic_limit,
            E_mean, E_std_ens, E_ci_lo, E_ci_hi, r2_mean,
            output_file='ensemble_errorbar.png'
        )
        figs.append((fig2, ax2))

    return figs


# ============================================================
# ANALISIS AKURASI
# ============================================================
def analyse_accuracy(config_list, include, strain_grid,
                     stress_matrix_plot, mean_curve):
    used_idx = [i for i, inc in enumerate(include) if inc]
    results  = []

    for i in used_idx:
        curve = stress_matrix_plot[i]
        valid = ~np.isnan(curve) & ~np.isnan(mean_curve)
        if np.sum(valid) < 5:
            continue

        diff    = curve[valid] - mean_curve[valid]
        rmse    = np.sqrt(np.mean(diff ** 2))
        mae     = np.mean(np.abs(diff))
        max_ae  = np.max(np.abs(diff))

        ss_res  = np.sum(diff ** 2)
        ss_tot  = np.sum((mean_curve[valid] - np.mean(mean_curve[valid])) ** 2)
        r2_fit  = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        results.append({
            'idx'    : i,
            'name'   : config_list[i]['name'],
            'E'      : config_list[i]['E'],
            'rmse'   : rmse,
            'mae'    : mae,
            'max_ae' : max_ae,
            'r2_fit' : r2_fit,
        })

    if not results:
        return results

    rmse_vals = np.array([r['rmse'] for r in results])
    mae_vals  = np.array([r['mae']  for r in results])

    rmse_min, rmse_max = rmse_vals.min(), rmse_vals.max()
    mae_min,  mae_max  = mae_vals.min(),  mae_vals.max()

    for r in results:
        norm_rmse = ((r['rmse'] - rmse_min) / (rmse_max - rmse_min)
                     if rmse_max > rmse_min else 0.0)
        norm_mae  = ((r['mae']  - mae_min)  / (mae_max  - mae_min)
                     if mae_max  > mae_min  else 0.0)
        r['score'] = 0.5 * norm_rmse + 0.5 * norm_mae

    results.sort(key=lambda x: x['score'])
    return results


# ============================================================
# SIMPAN HASIL
# ============================================================
def save_results(config_list, include, E_mean, E_std_ens,
                 E_ci_lo, E_ci_hi, elastic_limit, method_label,
                 strain_grid, stress_matrix_plot, mean_curve,
                 filename='ensemble_results.txt'):

    accuracy = analyse_accuracy(config_list, include, strain_grid,
                                stress_matrix_plot, mean_curve)

    with open(filename, 'w') as f:
        f.write("=" * 65 + "\n")
        f.write("  HASIL ANALISIS ENSEMBLE YOUNG'S MODULUS\n")
        f.write("  Simulasi LAMMPS MD — Uniaxial Tension\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"  Elastic limit  : {elastic_limit*100:.2f}%  ({method_label})\n\n")

        f.write("  HASIL PER KONFIGURASI:\n")
        f.write(f"  {'File':<35}  {'E (GPa)':>8}  {'R²':>8}  {'Status'}\n")
        f.write(f"  {'-'*35}  {'-'*8}  {'-'*8}  {'-'*10}\n")
        for i, cfg in enumerate(config_list):
            status = "digunakan" if include[i] else "DIBUANG"
            f.write(f"  {cfg['name']:<35}  {cfg['E']:>8.2f}  "
                    f"{cfg['r2']:>8.5f}  {status}\n")

        n_used = sum(include)
        f.write(f"\n  HASIL ENSEMBLE ({n_used} konfigurasi):\n")
        f.write(f"  {'Young Modulus mean':<30}: {E_mean:.2f} GPa\n")
        f.write(f"  {'Std deviation':<30}: {E_std_ens:.2f} GPa\n")
        f.write(f"  {'95% CI':<30}: [{E_ci_lo:.2f}, {E_ci_hi:.2f}] GPa\n")

        if accuracy:
            f.write("\n" + "=" * 65 + "\n")
            f.write("  ANALISIS AKURASI TERHADAP KURVA ENSEMBLE GABUNGAN\n")
            f.write("  (hanya konfigurasi yang DIGUNAKAN)\n")
            f.write("=" * 65 + "\n\n")
            f.write("  Metrik yang digunakan:\n")
            f.write("    RMSE   = Root Mean Square Error stress (GPa)\n")
            f.write("    MAE    = Mean Absolute Error stress (GPa)\n")
            f.write("    MaxAE  = Deviasi maksimum dari kurva mean (GPa)\n")
            f.write("    R²_fit = Koefisien determinasi kurva vs mean\n")
            f.write("    Skor   = Rata-rata RMSE & MAE ternormalisasi\n")
            f.write("             (0.000 = sempurna, 1.000 = terjauh)\n\n")

            f.write(f"  {'Rank':>4}  {'File':<35}  {'E(GPa)':>7}  "
                    f"{'RMSE':>8}  {'MAE':>8}  {'MaxAE':>8}  "
                    f"{'R²_fit':>8}  {'Skor':>6}\n")
            f.write(f"  {'-'*4}  {'-'*35}  {'-'*7}  "
                    f"{'-'*8}  {'-'*8}  {'-'*8}  "
                    f"{'-'*8}  {'-'*6}\n")

            for rank, r in enumerate(accuracy, 1):
                marker = "  ← TERBAIK" if rank == 1 else ""
                f.write(f"  {rank:>4}  {r['name']:<35}  {r['E']:>7.2f}  "
                        f"{r['rmse']:>8.5f}  {r['mae']:>8.5f}  "
                        f"{r['max_ae']:>8.5f}  {r['r2_fit']:>8.5f}  "
                        f"{r['score']:>6.3f}{marker}\n")

            best  = accuracy[0]
            worst = accuracy[-1]

            f.write("\n" + "-" * 65 + "\n")
            f.write("  KESIMPULAN\n")
            f.write("-" * 65 + "\n\n")
            f.write(f"  Konfigurasi PALING AKURAT (paling mendekati kurva gabungan):\n")
            f.write(f"    → {best['name']}\n")
            f.write(f"       E      = {best['E']:.2f} GPa\n")
            f.write(f"       RMSE   = {best['rmse']:.5f} GPa\n")
            f.write(f"       MAE    = {best['mae']:.5f} GPa\n")
            f.write(f"       MaxAE  = {best['max_ae']:.5f} GPa\n")
            f.write(f"       R²_fit = {best['r2_fit']:.5f}\n")
            f.write(f"       Skor   = {best['score']:.3f}  (terkecil = terbaik)\n\n")

            if len(accuracy) > 1:
                f.write(f"  Konfigurasi PALING JAUH dari kurva gabungan:\n")
                f.write(f"    → {worst['name']}\n")
                f.write(f"       E      = {worst['E']:.2f} GPa\n")
                f.write(f"       RMSE   = {worst['rmse']:.5f} GPa\n")
                f.write(f"       MAE    = {worst['mae']:.5f} GPa\n")
                f.write(f"       MaxAE  = {worst['max_ae']:.5f} GPa\n")
                f.write(f"       R²_fit = {worst['r2_fit']:.5f}\n")
                f.write(f"       Skor   = {worst['score']:.3f}\n\n")

            f.write(f"  Urutan akurasi (terbaik → terburuk):\n")
            for k, r in enumerate(accuracy):
                f.write(f"    #{k+1:>2}  {r['name']:<35}  "
                        f"RMSE={r['rmse']:.5f}  Skor={r['score']:.3f}\n")

        f.write("\n" + "=" * 65 + "\n")

    print(f"  ✓ Hasil disimpan: {filename}")

    if accuracy:
        print(f"\n{'=' * 65}")
        print(f"  AKURASI TERHADAP KURVA ENSEMBLE GABUNGAN")
        print(f"{'=' * 65}")
        print(f"  {'Rank':>4}  {'File':<30}  {'RMSE':>8}  {'MAE':>8}  "
              f"{'R²_fit':>8}  {'Skor':>6}")
        print(f"  {'-'*4}  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")
        for rank, r in enumerate(accuracy, 1):
            marker = " ← TERBAIK" if rank == 1 else ""
            print(f"  {rank:>4}  {r['name']:<30}  {r['rmse']:>8.5f}  "
                  f"{r['mae']:>8.5f}  {r['r2_fit']:>8.5f}  "
                  f"{r['score']:>6.3f}{marker}")
        print(f"\n  ✓ Konfigurasi paling akurat: {accuracy[0]['name']}")


# ============================================================
# SIMPAN DATA KURVA YANG DIFINALISASI
# ============================================================
def save_finalized_data(config_list, include, strain_grid,
                        stress_matrix, mean_curve, std_curve,
                        filename='finalized_curves.csv'):
    used_idx   = [i for i, inc in enumerate(include) if inc]
    used_names = [os.path.splitext(config_list[i]['name'])[0] for i in used_idx]
    used_E     = [config_list[i]['E'] for i in used_idx]

    header_parts = ['Strain']
    for name, E in zip(used_names, used_E):
        header_parts.append(f"{name}_Stress_GPa (E={E:.2f}GPa)")
    header_parts += ['Mean_Stress_GPa', 'Std_Stress_GPa',
                     'Band_Lower_GPa', 'Band_Upper_GPa']

    rows = np.column_stack(
        [strain_grid] +
        [stress_matrix[i] for i in used_idx] +
        [mean_curve, std_curve,
         mean_curve - std_curve,
         mean_curve + std_curve]
    )

    with open(filename, 'w') as f:
        f.write(','.join(header_parts) + '\n')
        for row in rows:
            f.write(','.join(f"{v:.8f}" for v in row) + '\n')

    print(f"  ✓ Data kurva finalisasi disimpan: {filename}")
    print(f"    Konfigurasi tersimpan : {len(used_idx)}  "
          f"({', '.join(used_names)})")
    print(f"    Jumlah titik strain   : {len(strain_grid)}")


# ============================================================
# MAIN
# ============================================================
def main():
    # ── Mulai logging layar — HARUS dipanggil pertama kali ──────────────
    _screen_logger.start()
    # Daftarkan stop() agar dipanggil otomatis saat program keluar
    # (termasuk saat user menutup jendela grafik / Ctrl+C / exception)
    atexit.register(_screen_logger.stop)

    print("\n" + "=" * 65)
    print("  ANALISIS ENSEMBLE YOUNG'S MODULUS")
    print("  LAMMPS MD — Multi-Konfigurasi")
    print("=" * 65)

    # ── 1. Pilih file ────────────────────────────────────────────────────
    filenames = select_input_files()
    if not filenames:
        print("  ✗ Tidak ada file dipilih. Keluar.")
        return

    # ── 2. Load semua data ───────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  MEMUAT DATA")
    print(f"{'=' * 65}\n")

    datasets = []
    for fname in filenames:
        data = load_single_file(fname)
        if data is not None:
            datasets.append({'name': fname, 'data': data})
            print(f"  ✓ {fname}  ({len(data)} baris)")

    if not datasets:
        print("  ✗ Tidak ada data valid. Keluar.")
        return

    if len(datasets) < 2:
        print("\n  ⚠  Hanya 1 file valid. Untuk ensemble minimal 2 file.")
        print("  Lanjutkan analisis tunggal? [Y/N]")
        ans = input("  ").strip().upper()
        if ans != "Y":
            return

    # ── 3. Pilih elastic limit ───────────────────────────────────────────
    ref_data    = datasets[0]['data']
    strain_ref  = ref_data[:, 0]
    stress_ref  = ref_data[:, 3]
    elastic_limit, method_label = select_elastic_limit_multi(strain_ref, stress_ref)

    # ── 4. Hitung E tiap konfigurasi ─────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  PERHITUNGAN MODULUS PER KONFIGURASI")
    print(f"{'=' * 65}\n")
    print(f"  {'No':>3}  {'File':<35}  {'E (GPa)':>8}  {'R²':>8}  {'N pts':>6}")
    print(f"  {'-'*3}  {'-'*35}  {'-'*8}  {'-'*8}  {'-'*6}")

    config_list = []
    for i, ds in enumerate(datasets):
        sx = ds['data'][:, 0]
        sy = ds['data'][:, 3]
        E, intercept, r2, s_el, sig_el = calc_modulus(sx, sy, elastic_limit)
        if E is None:
            print(f"  {i+1:>3}  {ds['name']:<35}  {'—':>8}  {'—':>8}  {'<5':>6}  ⚠ skip")
            continue
        config_list.append({
            'name': ds['name'], 'data': ds['data'],
            'E': E, 'intercept': intercept, 'r2': r2,
            's_el': s_el, 'sig_el': sig_el
        })
        name = ds['name'][:35]
        print(f"  {i+1:>3}  {name:<35}  {E:>8.2f}  {r2:>8.5f}  {len(s_el):>6}")

    if not config_list:
        print("  ✗ Tidak ada konfigurasi valid. Keluar.")
        return

    # ── 5. Buat grid strain seragam ──────────────────────────────────────
    all_strain_min = max(cfg['data'][:, 0].min() for cfg in config_list)
    all_strain_max = min(cfg['data'][:, 0].max() for cfg in config_list)
    if all_strain_min >= all_strain_max:
        print("  ✗ Rentang strain antar konfigurasi tidak overlap. Keluar.")
        return
    n_grid = 500
    strain_grid = np.linspace(all_strain_min, all_strain_max, n_grid)
    print(f"\n  Grid strain : {all_strain_min*100:.3f}% - {all_strain_max*100:.3f}%"
          f"  ({n_grid} titik)")

    stress_matrix = np.zeros((len(config_list), n_grid))
    for i, cfg in enumerate(config_list):
        sx = cfg['data'][:, 0]
        sy = cfg['data'][:, 3]
        stress_matrix[i] = interpolate_to_grid(sx, sy, strain_grid)

    # ── 6. Deteksi outlier ───────────────────────────────────────────────
    flags, reasons = detect_outliers(config_list, strain_grid, stress_matrix)

    # ── 7. User memutuskan buang/pertahankan ─────────────────────────────
    include = handle_outliers(config_list, flags, reasons)

    # ── 8. Hitung ensemble dari yang digunakan ───────────────────────────
    used_idx = [i for i, inc in enumerate(include) if inc]
    used_E   = [config_list[i]['E']  for i in used_idx]
    used_r2  = [config_list[i]['r2'] for i in used_idx]

    used_strain_min = max(config_list[i]['data'][:, 0].min() for i in used_idx)
    used_strain_max = min(config_list[i]['data'][:, 0].max() for i in used_idx)
    strain_grid_used = np.linspace(used_strain_min, used_strain_max, n_grid)

    used_stress = np.zeros((len(used_idx), n_grid))
    for k, i in enumerate(used_idx):
        sx = config_list[i]['data'][:, 0]
        sy = config_list[i]['data'][:, 3]
        used_stress[k] = interpolate_to_grid(sx, sy, strain_grid_used)

    all_valid = ~np.any(np.isnan(used_stress), axis=0)
    strain_grid = strain_grid_used[all_valid]
    used_stress = used_stress[:, all_valid]

    stress_matrix_plot = np.zeros((len(config_list), len(strain_grid)))
    for i, cfg in enumerate(config_list):
        sx = cfg['data'][:, 0]
        sy = cfg['data'][:, 3]
        stress_matrix_plot[i] = interpolate_to_grid(sx, sy, strain_grid)

    mean_curve = np.mean(used_stress, axis=0)
    std_curve  = np.std(used_stress,  axis=0, ddof=1)

    E_mean   = np.mean(used_E)
    r2_mean  = np.mean(used_r2)
    E_std_ens, E_ci_lo, E_ci_hi = bootstrap_ensemble_modulus(used_E)

    print(f"\n{'=' * 65}")
    print(f"  HASIL ENSEMBLE  ({len(used_idx)} konfigurasi digunakan)")
    print(f"{'=' * 65}")
    print(f"  E mean         : {E_mean:.2f} GPa")
    print(f"  Std deviation  : {E_std_ens:.2f} GPa")
    print(f"  95% CI         : [{E_ci_lo:.2f}, {E_ci_hi:.2f}] GPa")
    print(f"  R² mean        : {r2_mean:.5f}")

    # ── 9. Pilih jenis grafik lalu plot ──────────────────────────────────
    plot_mode = select_plot_mode()
    plot_ensemble(
        config_list, include, strain_grid, stress_matrix_plot,
        mean_curve, std_curve,
        elastic_limit,
        E_mean, E_std_ens, E_ci_lo, E_ci_hi,
        r2_mean,
        plot_mode=plot_mode
    )

    # ── 10. Simpan hasil + analisis akurasi ──────────────────────────────
    save_results(config_list, include, E_mean, E_std_ens,
                 E_ci_lo, E_ci_hi, elastic_limit, method_label,
                 strain_grid, stress_matrix_plot, mean_curve,
                 filename='ensemble_results.txt')

    # ── 11. Simpan data kurva yang digunakan di grafik ───────────────────
    save_finalized_data(config_list, include, strain_grid,
                        stress_matrix_plot, mean_curve, std_curve)

    print("\n" + "=" * 65)
    print(f"  SELESAI")
    print(f"  E  =  {E_mean:.2f} ± {E_std_ens:.2f} GPa")
    print(f"  95% CI  [{E_ci_lo:.2f}, {E_ci_hi:.2f}] GPa")
    print(f"  n  =  {len(used_idx)} konfigurasi")
    print("=" * 65 + "\n")

    # plt.show() dipanggil TERAKHIR — setelah ini program akan menunggu
    # user menutup jendela grafik; atexit akan memanggil _screen_logger.stop()
    plt.show()


if __name__ == "__main__":
    main()
