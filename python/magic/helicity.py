from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from magic import *
import sys

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
#############################################################

def gauss_weights(n_points):
    # Step 1: Obtain nodes and weights on the interval [-1, 1]
    nodes, weights = np.polynomial.legendre.leggauss(n_points)

    # Step 2: Map nodes from [-1, 1] to [0, pi]
    theta_nodes = 0.5 * (nodes + 1) * np.pi
    theta_weights = 0.5 * weights * np.pi

    # Step 3: Incorporate sin(theta) term into the weights
    theta_weights *= np.sin(theta_nodes)

    return theta_nodes, theta_weights
#############################################################

# Define the function that processes a single file
def process_file(checkpoint_file, l_cutoff):
    chk = MagicCheckpoint(filename=checkpoint_file)
    sh = SpectralTransforms(l_max=chk.l_max, minc=chk.minc, m_max=chk.m_max, n_theta_max=chk.n_theta_max)

    r = MagicRadial(field='anel', iplot=False)
    rho0 = r.rho0

    wpol = chk.wpol
    ztor = chk.ztor
    l_expanded = sh.ell
    m_expanded = sh.m

    # CREATE A MASK FOR FILTERING HIGHER MODES (l > l_cutoff)
    mask = np.ones(chk.lm_max, dtype=int)
    for filtr in range(chk.lm_max):
        l = l_expanded[filtr]
        m = m_expanded[filtr]
        if l > l_cutoff or m > l_cutoff:
            mask[filtr] = 0

    ztor = ztor * mask
    wpol = wpol * mask

    vr = np.zeros((chk.n_theta_max * 2, chk.n_theta_max, chk.n_r_max), np.float32)
    vt = np.zeros((chk.n_theta_max * 2, chk.n_theta_max, chk.n_r_max), np.float32)
    vp = np.zeros((chk.n_theta_max * 2, chk.n_theta_max, chk.n_r_max), np.float32)

    for i in range(chk.n_r_max):
        field1 = wpol[i, :] * l_expanded * (l_expanded + 1) / chk.radius[i]**2 / rho0[i]
        vr[:, :, i] = sh.spec_spat(field1)

    dwdr = rderavg(wpol.T, chk.radius).T

    for i in range(chk.n_r_max):
        field2, field3 = sh.spec_spat(dwdr[i, :], ztor[i, :])
        vt[:, :, i] = field2 / chk.radius[i] / rho0[i]
        vp[:, :, i] = field3 / chk.radius[i] / rho0[i]
    vScale = 2
    vr *= vScale
    vt *= vScale
    vp *= vScale

    th3D = np.zeros_like(vp)
    rr3D = np.zeros_like(th3D)
    colatitude = sh.colat
    for i in range(chk.n_theta_max):
        th3D[:, i, :] = colatitude[i]
    for i in range(chk.n_r_max):
        rr3D[:, :, i] = chk.radius[i]
    s3D = rr3D * np.sin(th3D)

    wp = 1. / chk.radius * (rderavg(chk.radius * vt, chk.radius) - thetaderavg(vr))
    wt = 1. / s3D * phideravg(vr, chk.minc) - 1. / chk.radius * rderavg(chk.radius * vp, chk.radius)
    wr = 1. / s3D * (thetaderavg(np.sin(th3D) * vp) - phideravg(vt, chk.minc))

    hel3D = vr * wr + vt * wt + vp * wp

    tmp_gauss = np.zeros(chk.n_theta_max)
    theta_ord = np.zeros(chk.n_theta_max)
    n_theta_cal2ord = np.zeros_like(theta_ord)
    gauleg(-1.0, 1.0, theta_ord, tmp_gauss)
    gauss = np.zeros_like(tmp_gauss)
    for n_theta in range(1, (chk.n_theta_max // 2) + 1):
        n_theta_cal2ord[2 * n_theta - 2] = n_theta
        n_theta_cal2ord[2 * n_theta - 1] = chk.n_theta_max - n_theta + 1
        gauss[2 * n_theta - 2] = tmp_gauss[n_theta - 1]
        gauss[2 * n_theta - 1] = tmp_gauss[chk.n_theta_max - n_theta]

    theta_nodes, theta_weights = gauss_weights(chk.n_theta_max)
    hel1D = np.zeros_like(chk.radius)
    hemisphere_limit = int(chk.n_theta_max / 2)
    print(chk.n_theta_max /2)
    print(hemisphere_limit)
    for i in range(chk.n_phi_tot):
        for j in range(hemisphere_limit):
            hel1D += hel3D[i,j,:] * theta_weights[j]
#            nTh = n_theta_cal2ord[j]
#            if nTh <= chk.n_theta_max / 2:
#                hel1D += hel3D[i, j, :] * gauss[j]

    hel0D = intcheb(hel1D * chk.radius * chk.radius, chk.n_r_max - 1, chk.radius[chk.n_r_max - 1], chk.radius[0])
    hel0D = hel0D * (-1)  # I am calculating north hemisphere only which is negative, but for clarity, changed it to positive. alternatively,I can calculate for south hemisphere.
    vol = 4.0/3 * np.pi * (chk.radius[0] ** 3 - chk.radius[chk.n_r_max - 1] ** 3)

    hel0D /= vol
    print(f"hel0D for {checkpoint_file}: {hel0D}")
    return hel0D

# Main execution block
folder_name = input("Enter the folder name: ")
l_cutoff = int(input("Enter the value of cutoff mode: "))
num_processes = int(input("Enter the number of parallel processes: "))
# Check if the folder exists
if not os.path.isdir(folder_name):
    print(f"Error: Folder '{folder_name}' not found.")
    sys.exit(1)

# Navigate to the folder and find all checkpoint files
os.chdir(folder_name)
files = sorted([f for f in os.listdir() if re.match(r'checkpoint_t=0_\d{6}\.test', f)])
print("Found files:", files)

# Initialize list to store hel0D values
hel0D_values = []

# Process files in parallel
with ProcessPoolExecutor(max_workers=num_processes) as executor:
    futures = {executor.submit(process_file, f, l_cutoff): f for f in files}
    for future in as_completed(futures):
        hel0D = future.result()
        hel0D_values.append(hel0D)

# Return to the parent directory
os.chdir("..")

# Ensure values are saved in order
hel0D_values = [x for _, x in sorted(zip(files, hel0D_values))]

# Save hel0D values to a file
output_filename = f"hel_l{l_cutoff}.txt"

with open(output_filename, 'w') as f:
    for value in hel0D_values:
        f.write(f"{value}\n")


# Plot the hel0D values
plt.figure(figsize=(10, 6))
plt.plot(hel0D_values, marker='o', linestyle='-', color='b', label='hel0D values')
plt.xlabel('Checkpoint File Index')
plt.ylabel('hel0D')
plt.title('hel0D Values across Checkpoint Files in Folder')
plt.grid(True)
plt.legend()
output_filename = f"plot_l{l_cutoff}.eps"
plt.savefig(output_filename, format='eps', dpi=300)
plt.show()

#np.savetxt("hel0D.txt", hel0D_values)

