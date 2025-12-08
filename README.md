

# FUS BBB Simulations: Real, End-to-End Guides

**With CT input, k‑space solver, Marmottant bubbles, bioheat coupling, sweeps, parameter tables, and recaps.**

1.  **Acoustic propagation through the skull** (k‑space pseudospectral, single focused transducer, CT → acoustic properties)
2.  **Microbubble dynamics** (Marmottant model)
3.  **Thermal rise** (bioheat equation fed by intensity from the acoustic simulation)
4.  **Parameter sweeping and comparison**
5.  **Clear parameter tables** with literature sources
6.  **Recaps** for each simulation (inputs, outputs, equations)

-----

## Project Setup and Folders

**Step-by-step (one-time)**

1.  **Create folder:** `fus_project`
2.  **Create subfolders:**
      * `data/` (put your CT files here: NIfTI like `head_ct.nii.gz` or a DICOM folder)
      * `scripts/`
      * `results/`
3.  **Create environment (Windows):**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
4.  **Create `requirements.txt`:**
    ```text
    numpy
    scipy
    matplotlib
    nibabel
    pydicom
    scikit-image
    h5py
    ```
5.  **Install:**
    ```bash
    pip install -r requirements.txt
    ```

-----

## 1\) Acoustic Simulation

**k‑space pseudospectral with real CT skull and single focused transducer**

### What you’ll do (plain words)

  * Load a CT slice (NIfTI or DICOM).
  * Convert HU values to acoustic speed, density, and attenuation (bone vs soft tissue).
  * Place a single focused transducer at the top and aim at a focal point.
  * Run a k‑space pseudospectral solver to get the pressure map.
  * Measure peak negative pressure (PNP) and intensity at the focus.
  * Save plots and metrics.

### Full Guide (Step-by-Step)

**Put CT file in `data/`:**

  * NIfTI: `data/head_ct.nii.gz`
  * DICOM: `data/ct_dicom/` (folder with `.dcm` files)

**Create script:** `scripts/acoustic_kspace_2d.py` (see code section below).

**Run (NIfTI):**

```bash
python scripts/acoustic_kspace_2d.py --ct_nifti data/head_ct.nii.gz --slice_index 80 --f0_khz 500 --drive_pnp_kpa 300
```

**Run (DICOM):**

```bash
python scripts/acoustic_kspace_2d.py --ct_dicom data/ct_dicom --slice_index 80 --f0_khz 500 --drive_pnp_kpa 300
```

**Check outputs in `results/`:**

  * `acoustic_slice80_final.png`
  * `acoustic_slice80_tXXXX.png` (snapshots)
  * `acoustic_slice80_metrics.json` (peak PNP, p\_rms, intensity)

### Equations (Exact, Simple)

 **HU → Density (clamped linear):**

$$\rho(HU) = \rho_{\text{water}} + k_\rho \cdot HU, \quad \rho_{\text{water}} \approx 1000 \text{ kg/m}^3, k_\rho \approx 0.5 \frac{\text{kg}}{\text{m}^3 \cdot HU}$$

*Clamp to realistic range:* $\rho \in [950, 2000] \text{ kg/m}^3$.

**HU → Speed (clamped linear):**

$$c(HU) = c_{\text{water}} + k_c \cdot HU, \quad c_{\text{water}} \approx 1480 \text{ m/s}, k_c \approx 1.0 \frac{\text{m}}{\text{s} \cdot HU}$$

*Clamp to* $c \in [1400, 3600] \text{ m/s}$.

**Attenuation (power law, convert to Nepers/m):**

$$\alpha(f) = \alpha_0 \left(\frac{f}{1 \text{ MHz}}\right)^n, \quad \alpha_0^{\text{bone}} \approx 30 \frac{\text{dB}}{\text{cm} \cdot \text{MHz}}, \alpha_0^{\text{brain}} \approx 0.7 \frac{\text{dB}}{\text{cm} \cdot \text{MHz}}, n \approx 1$$

$$\alpha_{\text{Np/m}} = \alpha_{\text{dB/cm}} \cdot \frac{0.115}{0.01}$$

**Single Focused Transducer Phase (geometric time-of-flight):**

$$\phi(x) = -2\pi f_0 \frac{r(x)}{c_{\text{ref}}}, \quad r(x) = |\mathbf{x} - \mathbf{x}_{\text{focus}}|$$

**Linear Damped Wave (k-space pseudospectral):**

$$\frac{\partial^2 p}{\partial t^2} = c(\mathbf{x})^2 \nabla^2 p - 2\alpha(\mathbf{x})\frac{\partial p}{\partial t} + s(\mathbf{x}, t)$$

Spatial Laplacian via FFT: $\widehat{\nabla^2} p = -k^2 \hat{p}$, *where* $k^2 = k_x^2 + k_y^2$

**Time Stepping (Leapfrog with damping):**

1.  Update a velocity-like field $v \approx \partial p/\partial t$.
2.  Then update $p$ with $v$.

**Intensity at Focus:**
$$I = \frac{p_{\text{rms}}^2}{\rho\ c}$$

### Code: `scripts/acoustic_kspace_2d.py`

```python
import numpy as np, matplotlib.pyplot as plt, argparse, os, json
import nibabel as nib, pydicom

def load_ct_slice(ct_nifti=None, ct_dicom=None, slice_index=80):
    if ct_nifti:
        img = nib.load(ct_nifti)
        vol = img.get_fdata()
        ct_slice = vol[:, :, slice_index].astype(np.float32)
        return ct_slice
    elif ct_dicom:
        files = sorted([os.path.join(ct_dicom, f) for f in os.listdir(ct_dicom) if f.lower().endswith('.dcm')])
        slices = []
        for fp in files:
            d = pydicom.dcmread(fp)
            arr = d.pixel_array.astype(np.float32)
            slope = getattr(d, 'RescaleSlope', 1.0)
            intercept = getattr(d, 'RescaleIntercept', 0.0)
            arr = arr * slope + intercept  # to HU
            slices.append(arr)
        vol = np.stack(slices, axis=-1)
        return vol[:, :, slice_index].astype(np.float32)
    else:
        raise ValueError("Provide --ct_nifti or --ct_dicom")

def hu_to_props(hu, f0_hz):
    hu = np.clip(hu, -1000, 3000)
    rho = 1000.0 + 0.5 * hu
    rho = np.clip(rho, 950.0, 2000.0)
    c = 1480.0 + 1.0 * hu
    c = np.clip(c, 1400.0, 3600.0)
    bone_mask = hu > 300
    alpha0_db_cm_mhz = np.where(bone_mask, 30.0, 0.7)
    f_mhz = f0_hz / 1e6
    alpha_db_cm = alpha0_db_cm_mhz * (f_mhz ** 1.0)
    alpha_np_m = alpha_db_cm * 0.115 / 0.01
    return c, rho, alpha_np_m

def phase_row_for_focus(nx, dx, dy, focus_xy, f0_hz, c_ref=1540.0, src_y=5):
    xf, yf = focus_xy
    row = np.zeros(nx, dtype=np.float32)
    for i in range(nx):
        r = np.sqrt(((i - xf)*dx)**2 + ((src_y - yf)*dy)**2)
        row[i] = -2*np.pi*f0_hz * r / c_ref
    return row

def acoustic_sim(ct_slice, f0_hz, drive_pnp_pa, dx, dy, focus_xy, nt, dt, src_y, out_prefix):
    nx, ny = ct_slice.shape
    c, rho, alpha_np_m = hu_to_props(ct_slice, f0_hz)

    p = np.zeros((nx, ny), dtype=np.float32)
    v = np.zeros_like(p)

    kx = np.fft.fftfreq(nx, dx) * 2*np.pi
    ky = np.fft.fftfreq(ny, dy) * 2*np.pi
    KX, KY = np.meshgrid(ky, kx)
    k2 = KX**2 + KY**2

    phase_row = phase_row_for_focus(nx, dx, dy, focus_xy, f0_hz, src_y=src_y)
    omega = 2*np.pi*f0_hz

    xf, yf = focus_xy
    peak_pnp = 0.0
    store = []

    for n in range(nt):
        t = n*dt
        p[:, src_y] += drive_pnp_pa * np.sin(omega*t + phase_row)

        Pk = np.fft.fft2(p)
        lap_p = np.real(np.fft.ifft2(-k2 * Pk))

        v = v + (c**2 * lap_p - 2*alpha_np_m * v) * dt
        p = p + v * dt

        if n % (nt//10) == 0 or n == nt-1:
            store.append(p.copy())
            plt.figure(figsize=(6,5))
            plt.imshow(p.T, cmap='RdBu', origin='lower')
            plt.scatter([xf],[yf], c='yellow', s=20)
            plt.colorbar(label='Pressure (Pa)')
            plt.title(f't={t*1e6:.1f} µs')
            plt.tight_layout()
            plt.savefig(f'{out_prefix}_t{n}.png', dpi=180)
            plt.close()

        roi = p[xf-3:xf+3, yf-3:yf+3]
        peak_pnp = max(peak_pnp, float(np.abs(roi.min())))

    stack = np.stack(store, axis=0)
    roi_stack = stack[:, xf-3:xf+3, yf-3:yf+3]
    p_rms = float(np.sqrt(np.mean(roi_stack**2)))
    rf = float(rho[xf, yf]); cf = float(c[xf, yf])
    intensity = float((p_rms**2)/(rf*cf))

    metrics = {"peak_pnp_pa": peak_pnp, "p_rms_pa": p_rms, "intensity_w_m2": intensity}
    with open(f'{out_prefix}_metrics.json','w') as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(6,5))
    plt.imshow(p.T, cmap='RdBu', origin='lower')
    plt.scatter([xf],[yf], c='yellow', s=20)
    plt.colorbar(label='Pressure (Pa)')
    plt.title('Final pressure field')
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_final.png', dpi=180)
    plt.close()

    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ct_nifti', type=str, default=None)
    ap.add_argument('--ct_dicom', type=str, default=None)
    ap.add_argument('--slice_index', type=int, default=80)
    ap.add_argument('--f0_khz', type=float, default=500.0)
    ap.add_argument('--drive_pnp_kpa', type=float, default=300.0)
    ap.add_argument('--dx_mm', type=float, default=0.5)
    ap.add_argument('--dy_mm', type=float, default=0.5)
    ap.add_argument('--focus_x', type=int, default=256)
    ap.add_argument('--focus_y', type=int, default=380)
    ap.add_argument('--src_y', type=int, default=8)
    ap.add_argument('--nt', type=int, default=4000)
    ap.add_argument('--dt_us', type=float, default=0.2)
    ap.add_argument('--out_prefix', type=str, default='results/acoustic_slice')
    args = ap.parse_args()

    os.makedirs('results', exist_ok=True)
    ct_slice = load_ct_slice(args.ct_nifti, args.ct_dicom, args.slice_index)
    f0_hz = args.f0_khz * 1e3
    dx = args.dx_mm * 1e-3
    dy = args.dy_mm * 1e-3
    drive_pnp_pa = args.drive_pnp_kpa * 1e3

    metrics = acoustic_sim(
        ct_slice=ct_slice, f0_hz=f0_hz, drive_pnp_pa=drive_pnp_pa,
        dx=dx, dy=dy, focus_xy=(args.focus_x, args.focus_y),
        nt=args.nt, dt=args.dt_us*1e-6, src_y=args.src_y,
        out_prefix=f'{args.out_prefix}{args.slice_index}'
    )
    print("Acoustic metrics:", metrics)

if __name__ == "__main__":
    main()
```


## 2\) Bubble Simulation

**Marmottant (coated microbubble)**

### What you’ll do

  * Take the PNP and frequency from the acoustic simulation at the focus.
  * Simulate how a coated bubble oscillates with that drive.
  * Look at radius-over-time and max excursion to judge stable vs risky behavior.

### Full Guide

**Create script:** `scripts/bubble_marmottant.py` (code below).

**Run with acoustic outputs (example):**

```bash
python scripts/bubble_marmottant.py --pnp_pa 250000 --f0_hz 500000 --r0_um 2.0
```

**Outputs in `results/`:** `bubble_radius.png`, printed max excursion.

### Equations (Exact)

**Gas Pressure (Polytropic):**
$$P_g(R) = P_0\left(\frac{R_0}{R}\right)^{3\gamma}$$

 **Shell Surface Tension (Marmottant):**

$$\sigma(R) = \begin{cases} 
0 & R < 0.9R_0 \\
\chi\left(\frac{R}{R_0} - 1\right) & 0.9R_0 \leq R \leq 1.2R_0 \\
\sigma_{\text{water}} & R > 1.2R_0
\end{cases}$$

 **Acoustic Drive:**
$$P_{\text{ac}}(t) = -P_{\text{PNP}} \sin(2\pi f_0 t)$$


**Modified Rayleigh–Plesset ODE:**
$$\ddot{R} = \frac{1}{\rho R}\left(P_g - P_0 + P_{\text{ac}} - \frac{2\sigma(R)}{R} - \frac{4\mu \dot{R}}{R}\right) - \frac{3}{2}\frac{\dot{R}^2}{R}$$

### Code: `scripts/bubble_marmottant.py`

```python
import numpy as np, matplotlib.pyplot as plt, argparse, os
from scipy.integrate import solve_ivp

def bubble_sim(pnp_pa, f0_hz, R0_um, out_path):
    R0 = R0_um * 1e-6
    rho = 1000.0; mu = 0.001; P0 = 101325.0; gamma = 1.4
    chi = 0.5; sigma_water = 0.072
    omega = 2*np.pi*f0_hz
    Rbuck = 0.9*R0; Rrupt = 1.2*R0

    def sigma(R):
        if R < Rbuck: return 0.0
        elif R <= Rrupt: return chi * (R/R0 - 1.0)
        else: return sigma_water

    def rhs(t, y):
        R, dR = y
        Pac = -pnp_pa * np.sin(omega*t)
        Pg = P0 * (R0/R)**(3*gamma)
        S = sigma(R)
        ddR = (1.0/(rho*R))*(Pg - P0 + Pac - 2.0*S/R - 4.0*mu*dR/R) - 1.5*(dR**2)/R
        return [dR, ddR]

    sol = solve_ivp(rhs, [0, 2e-3], [R0, 0.0], max_step=1e-7, rtol=1e-7, atol=1e-9)
    t = sol.t; R = sol.y[0]
    max_exc_um = (R.max()-R0)*1e6

    plt.figure(figsize=(6,4))
    plt.plot(t*1e3, R*1e6)
    plt.xlabel('Time (ms)'); plt.ylabel('Radius (µm)')
    plt.title('Marmottant bubble oscillation')
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

    return {"R0_um": R0_um, "max_excursion_um": float(max_exc_um)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pnp_pa', type=float, required=True)
    ap.add_argument('--f0_hz', type=float, required=True)
    ap.add_argument('--r0_um', type=float, default=2.0)
    ap.add_argument('--out_path', type=str, default='results/bubble_radius.png')
    args = ap.parse_args()
    os.makedirs('results', exist_ok=True)
    metrics = bubble_sim(args.pnp_pa, args.f0_hz, args.r0_um, args.out_path)
    print("Bubble metrics:", metrics)

if __name__ == "__main__":
    main()
```


## 3\) Thermal Simulation

**Bioheat fed by intensity from acoustic simulation**

### What you’ll do

  * Use intensity at the focus from the acoustic simulation.
  * Compute temperature rise using the bioheat equation.
  * Check peak temperature rise (keep small, e.g., \< 1 °C).

### Full Guide

**Create script:** `scripts/bioheat_2d.py` (code below).

**Run with intensity from acoustic JSON (example):**

```bash
python scripts/bioheat_2d.py --intensity_w_m2 6.5 --alpha_np_m 0.5 --exposure_s 120
```

**Outputs in `results/`:** `temperature_final.png`, printed peak ΔT.

### Equations (Exact)

**Bioheat PDE:**
$$\rho c_p \frac{\partial T}{\partial t} = \nabla\cdot(k\nabla T) + Q_{\text{ac}} - \omega_b c_b (T - T_b)$$

**Acoustic Heating Source:**
$$Q_{\text{ac}} = 2\alpha I$$

**Intensity from Acoustic Sim:**
$$I = \frac{p_{\text{rms}}^2}{\rho c}$$

### Code: `scripts/bioheat_2d.py`

```python
import numpy as np, matplotlib.pyplot as plt, argparse, os

def bioheat(intensity, alpha_np_m, exposure_s, nx=256, ny=256, dx=1e-3, dy=1e-3, dt=0.02):
    rho = 1030.0; cp = 3600.0; k = 0.5
    wb = 0.01; cb = 4180.0; Tb = 37.0

    x = np.arange(nx)*dx; y = np.arange(ny)*dy
    X, Y = np.meshgrid(x, y, indexing='ij')
    xf, yf = nx*dx/2, ny*dy*0.7
    sigma = 5e-3

    Q = 2.0 * alpha_np_m * intensity * np.exp(-((X-xf)**2+(Y-yf)**2)/(2*sigma**2))
    T = np.ones((nx, ny))*Tb

    Nt = int(exposure_s/dt)
    for n in range(Nt):
        lap = (np.roll(T,1,0)+np.roll(T,-1,0)+np.roll(T,1,1)+np.roll(T,-1,1)-4*T)/(dx*dy)
        dT = (k*lap + Q - wb*cb*(T - Tb)) / (rho*cp)
        T += dt*dT
        if n % max(1, Nt//5) == 0 or n == Nt-1:
            plt.figure(figsize=(6,5))
            plt.imshow(T.T, origin='lower', cmap='inferno')
            plt.colorbar(label='°C')
            plt.title(f'Temperature at t={n*dt:.1f}s')
            plt.tight_layout()
            plt.savefig(f'results/temperature_t{n}.png', dpi=180)
            plt.close()

    peak = float(T.max()-Tb)
    plt.figure(figsize=(6,5))
    plt.imshow(T.T, origin='lower', cmap='inferno')
    plt.colorbar(label='°C')
    plt.title('Final temperature')
    plt.tight_layout()
    plt.savefig('results/temperature_final.png', dpi=180)
    plt.close()
    return {"peak_delta_c": peak}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--intensity_w_m2', type=float, required=True)
    ap.add_argument('--alpha_np_m', type=float, required=True)
    ap.add_argument('--exposure_s', type=float, default=120.0)
    args = ap.parse_args()
    os.makedirs('results', exist_ok=True)
    metrics = bioheat(args.intensity_w_m2, args.alpha_np_m, args.exposure_s)
    print("Bioheat metrics:", metrics)

if __name__ == "__main__":
    main()
```


## 4\) Parameter Sweeping and Comparisons

### What you’ll do

  * Loop over different frequencies and drive pressures.
  * For each combo: run acoustic → read PNP and intensity → run bubble → run heat.
  * Save a summary CSV and comparison plots.

### Full Guide

**Create script:** `scripts/param_sweep.py` (code below).

**Run:**

```bash
python scripts/param_sweep.py --ct_nifti data/head_ct.nii.gz --slice_index 80
```

**Outputs:**

  * `results/sweep_summary.csv`
  * A bubble plot and temperature map per last run (you can extend to save all).

### Code: `scripts/param_sweep.py`

```python
import subprocess, json, os, argparse, csv

def run_acoustic(ct_nifti, slice_index, f0_khz, drive_pnp_kpa):
    out_prefix = f'results/acoustic_slice{slice_index}_f{int(f0_khz)}_p{int(drive_pnp_kpa)}'
    cmd = [
        'python','scripts/acoustic_kspace_2d.py',
        '--ct_nifti', ct_nifti,
        '--slice_index', str(slice_index),
        '--f0_khz', str(f0_khz),
        '--drive_pnp_kpa', str(drive_pnp_kpa),
        '--out_prefix', out_prefix
    ]
    subprocess.run(cmd, check=True)
    with open(f'{out_prefix}_metrics.json','r') as f:
        m = json.load(f)
    return m, out_prefix

def run_bubble(pnp_pa, f0_hz, tag):
    out = f'results/bubble_radius_{tag}.png'
    cmd = [
        'python','scripts/bubble_marmottant.py',
        '--pnp_pa', str(pnp_pa),
        '--f0_hz', str(f0_hz),
        '--r0_um', '2.0',
        '--out_path', out
    ]
    subprocess.run(cmd, check=True)

def run_bioheat(intensity_w_m2, tag, alpha_np_m=0.5, exposure_s=120):
    cmd = [
        'python','scripts/bioheat_2d.py',
        '--intensity_w_m2', str(intensity_w_m2),
        '--alpha_np_m', str(alpha_np_m),
        '--exposure_s', str(exposure_s)
    ]
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ct_nifti', type=str, required=True)
    ap.add_argument('--slice_index', type=int, default=80)
    args = ap.parse_args()
    os.makedirs('results', exist_ok=True)

    freqs = [220, 350, 500, 700]  # kHz
    drives = [200, 300, 400, 500] # kPa (pre-skull)
    rows = []

    for f0 in freqs:
        for p in drives:
            tag = f'f{f0}_p{p}'
            m, prefix = run_acoustic(args.ct_nifti, args.slice_index, f0, p)
            pnp_pa = m['peak_pnp_pa']
            intensity = m['intensity_w_m2']
            f0_hz = f0*1e3

            run_bubble(pnp_pa, f0_hz, tag)
            run_bioheat(intensity, tag)

            rows.append({
                'freq_khz': f0, 'drive_pnp_kpa': p,
                'peak_pnp_pa': m['peak_pnp_pa'],
                'p_rms_pa': m['p_rms_pa'],
                'intensity_w_m2': m['intensity_w_m2']
            })
            print(f"[{tag}] PNP={m['peak_pnp_pa']:.0f} Pa, I={m['intensity_w_m2']:.3f} W/m2")

    with open('results/sweep_summary.csv','w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    main()
```

-----

## Parameter Tables with Literature References

### Acoustic Simulation Parameters

| Parameter | Value / Range | Why it matters | Reference |
| :--- | :--- | :--- | :--- |
| **Brain speed $c$** | \~1540 m/s | Sets wave speed in brain | Vendel et al., 2019 (Fluids Barriers CNS) |
| **Skull speed $c$** | 2800–3500 m/s | Faster wave, phase errors | Aubry et al., 2003 (JASA); Jones & Hynynen, 2016 (Phys Med Biol) |
| **Brain density $\rho$** | \~1030 kg/m³ | Used in intensity | IT’IS Foundation Tissue Properties |
| **Skull density $\rho$** | 1850–2000 kg/m³ | Impedance & intensity | IT’IS Foundation Tissue Properties |
| **Brain attenuation $\alpha_0$** | 0.5–1.0 dB/cm/MHz | Energy loss | He et al., 2018 (Cells, review) |
| **Skull attenuation $\alpha_0$** | \~30 dB/cm/MHz @ \~0.5 MHz | Major loss/heating | Jones & Hynynen, 2016 |
| **Frequency $f_0$** | 220–700 kHz | Penetration vs focus | Clinical BBB arrays ≈220 kHz; preclinical up to ≈700 kHz |
| **Drive PNP (pre‑skull)** | 200–500 kPa | To reach 0.2–0.6 MPa post‑skull | O’Reilly & Hynynen, 2012; Wu et al., 2021 (review) |

### Bubble Simulation (Marmottant) Parameters

| Parameter | Value / Range | Why it matters | Reference |
| :--- | :--- | :--- | :--- |
| **Bubble radius $R_0$** | 1.5–2.5 µm | Resonance & response | Definity/Sonovue datasheets |
| **Gas polytropic $\gamma$** | 1.4 | Bubble gas compressibility | Marmottant et al., 2005 (JASA) |
| **Viscosity $\mu$** | 0.001 Pa·s | Damping | Standard water |
| **Shell elasticity $\chi$** | 0.1–1.0 N/m | Coating stiffness | Marmottant et al., 2005 |
| **Buckling/Rupture** | 0.9 / 1.2 $R_0$ | Surface tension regime | Marmottant et al., 2005 |

### Thermal Simulation Parameters

| Parameter | Value / Range | Why it matters | Reference |
| :--- | :--- | :--- | :--- |
| **$\rho$ (brain)** | 1030 kg/m³ | Heat capacity term | IT’IS Foundation |
| **$c_p$ (brain)** | 3600 J/(kg·K) | Heat capacity | IT’IS Foundation |
| **$k$ (brain)** | 0.5 W/(m·K) | Heat conduction | IT’IS Foundation |
| **Perfusion $\omega_b$** | 0.01 s⁻¹ | Cooling | Bioheat literature (Pennes-type) |
| **Blood $c_b$** | 4180 J/(kg·K) | Cooling capacity | Standard |
| **Baseline $T_b$** | 37 °C | Physiological baseline | Standard |
| **Attenuation $\alpha$** | from acoustic map | Converts intensity to heat source Q | Coupling equation above |

-----

## Recaps: Inputs, Outputs, Equations

**Acoustic (k‑space, single focus)**

  * **Inputs:** CT slice (HU), frequency ($f_0$), drive PNP (pre‑skull), grid spacing, focus coordinate, source line.
  * **Outputs:** Pressure maps, peak negative pressure (PNP) at focus, $p_{\text{rms}}$, intensity.
  * **Equations:** HU→($\rho,c,\alpha$); k‑space damped wave; phase focus; $I = p_{\text{rms}}^2/(\rho c)$.

**Bubble (Marmottant)**

  * **Inputs:** PNP (Pa) and $f_0$ (Hz) from acoustic sim; $R_0$; fluid & shell parameters.
  * **Outputs:** Radius vs time, max excursion (µm).
  * **Equations:** Marmottant surface tension + modified Rayleigh–Plesset.

**Thermal (bioheat)**

  * **Inputs:** Intensity ($I$) (W/m²) from acoustic sim; attenuation ($\alpha$); exposure time; thermal and perfusion params.
  * **Outputs:** Temperature maps; peak ΔT (°C).
  * **Equations:** Bioheat PDE, $Q_{\text{ac}} = 2\alpha I$.

-----

## One last thing: exact commands to get started fast

**Activate environment:**

```bash
venv\Scripts\activate
```

**Run acoustic (NIfTI):**

```bash
python scripts/acoustic_kspace_2d.py --ct_nifti data/head_ct.nii.gz --slice_index 80 --f0_khz 500 --drive_pnp_kpa 300
```

**Read acoustic metrics (JSON) and plug into bubble:**

```bash
python scripts/bubble_marmottant.py --pnp_pa <peak_pnp_pa_from_json> --f0_hz 500000 --r0_um 2.0
```

**Run heat (use intensity from acoustic JSON):**

```bash
python scripts/bioheat_2d.py --intensity_w_m2 <intensity_from_json> --alpha_np_m 0.5 --exposure_s 120
```

**Run sweeps:**

```bash
python scripts/param_sweep.py --ct_nifti data/head_ct.nii.gz --slice_index 80
```
