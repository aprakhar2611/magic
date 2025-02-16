from magic import *
import re
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor, as_completed

#############################################################
def get_namelist_value(var_name="m_max", filename="input.nml"):
    """Generalized function to extract the value of any specified variable from a namelist file."""
    if not os.path.isfile(filename):
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    pattern = rf'^\s*{var_name}\s*=\s*(.*?)[,\n!]'  
    value = None
    
    with open(filename, 'r') as file:
        for line in file:
            if re.search(r'^\s*!', line):  # Skip commented lines
                continue
            match = re.search(pattern, line)
            if match:
                value = match.group(1).strip()
                break
    
    if value is None:
        print(f"Warning: '{var_name}' not found or commented out in the file.")
    else:
        print(f"{var_name} = {value}")
    
    return value

#############################################################
def gauleg(sinThMin, sinThMax, theta_ord, gauss):
    """Legendre-Gauss integration to compute nodes and weights."""
    eps = 10.0 * np.finfo(float).eps
    n_th_max = theta_ord.size
    m = (n_th_max + 1) // 2
    sinThMean = 0.5 * (sinThMax + sinThMin)
    sinThDiff = 0.5 * (sinThMax - sinThMin)

    for i in range(1, m + 1):
        z = np.cos(np.pi * (i - 0.25) / (n_th_max + 0.5))
        z1 = z + 10.0 * eps
        while np.abs(z - z1) > eps:
            p1 = 1.0
            p2 = 0.0
            for j in range(1, n_th_max + 1):
                p3 = p2
                p2 = p1
                p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j
            pp = n_th_max * (z * p1 - p2) / (z * z - 1.0)
            z1 = z
            z = z1 - p1 / pp
        theta_ord[i - 1] = np.arccos(sinThMean + sinThDiff * z)
        theta_ord[n_th_max - i] = np.arccos(sinThMean - sinThDiff * z)
        gauss[i - 1] = 2.0 * sinThDiff / ((1.0 - z * z) * pp * pp)
        gauss[n_th_max - i] = gauss[i - 1]

def process_checkpoint(checkpoint_file, l_cutoff, loopindex):
    """Process each checkpoint file and save the results."""
    print(f"Processing {checkpoint_file}")
    chk = MagicCheckpoint(filename=checkpoint_file)
    sh = SpectralTransforms(l_max=chk.l_max, minc=chk.minc, m_max=chk.m_max, n_theta_max=chk.n_theta_max)

    # Obtain density (rho0) and read other required variables from checkpoint file
    rho0 = np.ones(chk.n_r_max)
    wpol = chk.wpol
    ztor = chk.ztor
    bpol = chk.bpol
    btor = chk.btor
    l_expanded = sh.ell
    m_expanded = sh.m

    # CREATE A MASK FOR FILTERING HIGHER MODES (l>l_cutoff)
    mask = np.ones(chk.lm_max, dtype=int)
    for filtr in range(chk.lm_max):
        l = l_expanded[filtr]
        m = m_expanded[filtr]
        if l > l_cutoff or m > l_cutoff:
            mask[filtr] = 0

    ztor = ztor * mask
    wpol = wpol * mask
    bpol = bpol * mask
    btor = btor * mask

    vr = np.zeros((chk.n_theta_max * 2, chk.n_theta_max, chk.n_r_max), np.float32)
    vt = np.zeros((chk.n_theta_max * 2, chk.n_theta_max, chk.n_r_max), np.float32)
    vp = np.zeros((chk.n_theta_max * 2, chk.n_theta_max, chk.n_r_max), np.float32)
    br = np.zeros((chk.n_theta_max * 2, chk.n_theta_max, chk.n_r_max), np.float32)
    bt = np.zeros((chk.n_theta_max * 2, chk.n_theta_max, chk.n_r_max), np.float32)
    bp = np.zeros((chk.n_theta_max * 2, chk.n_theta_max, chk.n_r_max), np.float32)
    for i in range(chk.n_r_max):
        vfield1 = wpol[i, :] * l_expanded * (l_expanded + 1) / chk.radius[i]**2 / rho0[i]
        bfield1 = wpol[i, :] * l_expanded * (l_expanded + 1) / chk.radius[i]**2
        vr[:, :, i] = sh.spec_spat(vfield1)
        br[:, :, i] = sh.spec_spat(bfield1)

    dwdr = rderavg(wpol.T, chk.radius).T
    dbpoldr = rderavg(bpol.T, chk.radius).T

    for i in range(chk.n_r_max):
        vfield2, vfield3 = sh.spec_spat(dwdr[i, :], ztor[i, :])
        bfield2, bfield3 = sh.spec_spat(dbpoldr[i, :], btor[i, :])
       
        vt[:, :, i] = vfield2 / chk.radius[i] / rho0[i]
        vp[:, :, i] = vfield3 / chk.radius[i] / rho0[i]
        bt[:, :, i] = bfield2 / chk.radius[i]
        bp[:, :, i] = bfield3 / chk.radius[i]

    vScale = 2
    vr = vr * vScale
    vt = vt * vScale
    vp = vp * vScale
    "for now not changing any scale for magnetic field. If any, I will have to check first from other place where it has been calculated just like I did for velocity filed"

    th3D = np.zeros_like(vp)
    rr3D = np.zeros_like(th3D)
    ph3D = np.zeros_like(vp)
    ph = np.linspace(0, 2 * np.pi, chk.n_phi_tot)
    colatitude = sh.colat
    for i in range(chk.n_theta_max):
        th3D[:, i, :] = colatitude[i]
    for i in range(chk.n_r_max):
        rr3D[:, :, i] = chk.radius[i]
    for i in range(chk.n_phi_tot):
        ph3D[i, :, :] = ph[i]

    s3D = rr3D * np.sin(th3D)
    
    uz = vr * np.cos(th3D) - vt * np.sin(th3D)
    uz_flattened = uz.flatten()
    filename = f"uz_{loopindex}.dat"
    np.savetxt(filename, uz_flattened)
    time = chk.time

    bz = br * np.cos(th3D) - bt * np.sin(th3D)
    bz_flattened = bz.flatten()
    filename = f"bz_{loopindex}.dat"
    np.savetxt(filename, bz_flattened)
    time = chk.time
    return time#, rr3D, th3D, ph3D
#############################################################
# Find and sort all files with names of the form 'checkpoint_t=0_000000.test'
files = sorted([f for f in os.listdir() if re.match(r'checkpoint_t=0_\d{6}\.test', f)])
print("Found files:", files)

# Prompt for l_cutoff (only once)
l_cutoff = int(input("Enter the value of cutoff mode: "))

# Get user input for the number of parallel processes
num_processes = int(input("Enter the number of parallel processes: "))

# Initialize arrays to store results
time_values = []
rr3D_list = []
th3D_list = []
ph3D_list = []

# Process each checkpoint file in sequence but with a limited number of threads
with ProcessPoolExecutor(max_workers=num_processes) as executor:
    futures = []
    
    for loopindex, checkpoint_file in enumerate(files):
        future = executor.submit(process_checkpoint, checkpoint_file, l_cutoff, loopindex)
        futures.append((future, loopindex))  # Store future along with its index
    
    # Collect results in order
    for future, loopindex in futures:
        try:
           # time, rr3D, th3D, ph3D = future.result()
            time = future.result()
            time_values.append(time)
           # rr3D_list.append(rr3D)
           # th3D_list.append(th3D)
           # ph3D_list.append(ph3D)
        except Exception as e:
            print(f"Error processing file {files[loopindex]}: {e}")

# Save the time array
time_array = np.array(time_values)
np.savetxt('stimes.dat', time_array, fmt='%f')

#rtp_filename = 'rtp.dat'
#rr_flattened = np.concatenate([rr.flatten() for rr in rr3D_list])
#th_flattened = np.concatenate([th.flatten() for th in th3D_list])
#ph_flattened = np.concatenate([ph.flatten() for ph in ph3D_list])

# Stack the flattened arrays into columns (2D array with 3 columns)
#flattened_data = np.column_stack((rr_flattened, th_flattened, ph_flattened))

# Save to a .dat file with space-separated values
#np.savetxt(rtp_filename, flattened_data, fmt='%e', delimiter=' ', header='rr3D th3D ph3D')

