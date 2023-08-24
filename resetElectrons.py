# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 05:40:34 2022

@author: Racha
"""
import sys
import tables
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Don't display the plot
import matplotlib.pyplot as plt

filename_e = sys.argv[1]
# filename_e = 'D:/Puffin_BUILD/Puffin_BIN/examples/simple/1D/single_spike_resetelectrons/new_reset1/SSS_e_500.h5'

hdf5_file = tables.open_file(filename_e, mode='r+')

def cal_energy_spread(e_arr, gamma):
    m_GAMMA = e_arr[:, 5]
    m_Z = e_arr[:, 2]
    gamma_j = gamma * m_GAMMA
    energy_spread = (gamma_j - gamma) * 100 / gamma
    return m_Z, energy_spread

# read electrons data set
e_data = hdf5_file.root.electrons.read()
m_Z = e_data[:, 2] # electron position in Z2
m_GAMMA = e_data[:, 5] # electron Gamma_j

# Calculate energy spread
gamma_r = hdf5_file.root.runInfo._v_attrs.gamma_r
gamma_j = gamma_r * m_GAMMA
energy_spread = (gamma_j - gamma_r) * 100 / gamma_r # in the unit of percentage
rho = hdf5_file.root.runInfo._v_attrs.rho

meshsizeZ2 = hdf5_file.root.runInfo._v_attrs.sLengthOfElmZ2
nZ2 = hdf5_file.root.runInfo._v_attrs.nZ2
print("z2 mesh =", meshsizeZ2)
print("nz2 =", nZ2)

modelLengthInZ2 = nZ2*meshsizeZ2
Lc = hdf5_file.root.runInfo._v_attrs.Lc
lambda_r = hdf5_file.root.runInfo._v_attrs.lambda_r
modelLengthInlambda = modelLengthInZ2/(4*np.pi*rho)
print("model length in lambda =", modelLengthInlambda)

numelec = np.shape(e_data)[0]
print("total number of electrons =", numelec)

MPsPerWave = np.int64(np.round(numelec/modelLengthInlambda))
print("MPsPerWave =", MPsPerWave)

# generate the equally space with in z2 model length
inElectronSpace = np.float64(np.linspace((nZ2-1)*meshsizeZ2/(2*numelec), (nZ2-1)*meshsizeZ2*(1-1.0/(2*numelec)), numelec))

numEdit = 180*MPsPerWave # unit of z2
resetposition = inElectronSpace[-numEdit:][0]

print("number of initial ending electrons =", numEdit) # initial edit length

# sort the electrons using column #2 (electron_z)
sorted_elec = e_data[e_data[:,2].argsort()]

# Check if there are any all-zero rows
if np.any(np.all(sorted_elec == 0, axis=1)):
    # Split the array into two parts: zero and non-zero
    zero_e = sorted_elec[np.all(sorted_elec == 0, axis=1)]
    zero_e = zero_e.astype(float)
    zero_e[zero_e == 0] = np.nan
    non_zero_e = sorted_elec[~np.all(sorted_elec == 0, axis=1)]
else:
    # If there are no all-zero rows, the entire array is non-zero
    zero_e = np.array([])  # Empty array
    non_zero_e = sorted_elec

non_zero_e[-numEdit:,5] = 1.0
non_zero_e[-numEdit:,4] = 0.0
non_zero_e[-numEdit:,3] = 0.0
non_zero_e[-numEdit:,2] = inElectronSpace[-numEdit:]
non_zero_e[-numEdit:,1] = 0.0
non_zero_e[-numEdit:,0] = 0.0

# Find the index where values in the second column are greater than resetposition
split_index = np.where(non_zero_e[:, 2] >= resetposition)[0][0]
# Split the array at the found index
first_part, second_part = np.split(non_zero_e, [split_index])

print("number of actual ending electrons =", len(second_part)) # above the condition
new_start = resetposition

print("z2 start position to reset =", new_start)


if len(second_part) < numEdit:
    resetLength = 0
    numEdit = len(second_part)
else:
    resetLength = len(second_part) - numEdit

print("number of zero gamma electrons =", resetLength)

first_part[:numEdit,5] = 1.0
first_part[:numEdit,2] = inElectronSpace[:numEdit] # reset position # e_z
first_part[:numEdit,4] = 0.0
first_part[:numEdit,3] = 0.0
first_part[:numEdit,1] = 0.0
first_part[:numEdit,0] = 0.0

if resetLength == 0:
    first_part = first_part
else:
    first_part[-resetLength:,:] = np.nan
    """
    first_part_0, first_part_1 = np.split(first_part, [-resetLength])

    first_part_1[:,6] = np.nan
    first_part_1[:,5] = np.nan
    first_part_1[:,2] = np.nan # reset position # e_z
    first_part_1[:,4] = np.nan
    first_part_1[:,3] = np.nan
    first_part_1[:,1] = np.nan
    first_part_1[:,0] = np.nan

    first_part = np.vstack((first_part_0, first_part_1))
    """
second_part[:,5] = 1.0 # e_gamma
second_part[:,4] = 0.0 # p_y
second_part[:,3] = 0.0 # p_x
# second_part[-numEdit*1:,2] = inElectronSpace[-numEdit*1:] # reset position # e_z
# second_part[:setzeroLength,2] = resetposition[0] # e_z
second_part[:,2] = inElectronSpace[-len(second_part):]
second_part[:,1] = 0.0 # e_y
second_part[:,0] = 0.0 # e_x



if np.any(np.all(sorted_elec == 0, axis=1)):
    rearranged_e = np.vstack((first_part, second_part))
    rearranged_e =  np.vstack((rearranged_e, zero_e))
else:
    rearranged_e = np.vstack((first_part, second_part))
# rearranged_e[:len(second_part)] = second_part

# Assuming cal_energy_spread returns two numpy arrays x, y
x, y = cal_energy_spread(e_data, gamma_r)
# Define your ranges
num_intervals = len(x) // MPsPerWave  # number of intervals, each of length MPsPerWave
ranges = [(i*MPsPerWave, (i+1)*MPsPerWave) for i in range(num_intervals)]
colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple']  # and these with your colors
num_colors_needed = len(ranges)
colors *= num_colors_needed // len(colors) + 1

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
# Plot energy spread
ax1.plot(cal_energy_spread(e_data, gamma_r)[0], cal_energy_spread(e_data, gamma_r)[1], '.', markersize=0.5, color='tab:red', label='Energy spread')
ax1.plot(cal_energy_spread(rearranged_e, gamma_r)[0], cal_energy_spread(rearranged_e, gamma_r)[1], 'x', markersize=0.5, color='tab:green', label='Energy spread')
"""
for i, (start, end) in enumerate(ranges):
    # plot this section with the corresponding color
    ax1.plot(x[start:end], y[start:end], '.', markersize=10.0, color=colors[i], label=f'Index range {start}-{end}')
"""
ax1.set_xlim(0)
# ax1.set_ylim(-80,80)
ax1.set_xlabel(r'$z_2$')
ax1.set_ylabel('$e^-$ % energy spread')
# Adjust the plot layout
plt.tight_layout()
# Save the plot as a PNG file
print("Saving the plot...")
output_filename = sys.argv[2]
# output_filename = filename_e[:-3] +'_spread'+ '.png'
plt.savefig(output_filename, dpi=300)

e_dataset = hdf5_file.get_node('/electrons')
e_dataset[:] = rearranged_e[rearranged_e[:,2].argsort()]
hdf5_file.root.time._v_attrs.vsStep = 0
hdf5_file.root.time._v_attrs.vsTime = 0.0
hdf5_file.root.electrons._v_attrs.iCsteps = 0
hdf5_file.root.electrons._v_attrs.iL = 0
hdf5_file.root.electrons._v_attrs.iWrite_cr = 0
hdf5_file.root.electrons._v_attrs.istep = 0
hdf5_file.root.electrons._v_attrs.time = 0.0
hdf5_file.root.electrons._v_attrs.zInter = 0.0
hdf5_file.root.electrons._v_attrs.zLocal = 0.0
hdf5_file.root.electrons._v_attrs.zTotal = 0.0
hdf5_file.root.electrons._v_attrs.zbarInter = 0.0
hdf5_file.root.electrons._v_attrs.zbarLocal = 0.0
hdf5_file.root.electrons._v_attrs.zbarTotal = 0.0

hdf5_file.root.runInfo._v_attrs.iCsteps = 0
hdf5_file.root.runInfo._v_attrs.iL = 0
hdf5_file.root.runInfo._v_attrs.iWrite_cr = 0
hdf5_file.root.runInfo._v_attrs.istep = 0
hdf5_file.root.runInfo._v_attrs.time = 0.0
hdf5_file.root.runInfo._v_attrs.zInter = 0.0
hdf5_file.root.runInfo._v_attrs.zLocal = 0.0
hdf5_file.root.runInfo._v_attrs.zTotal = 0.0
hdf5_file.root.runInfo._v_attrs.zbarInter = 0.0
hdf5_file.root.runInfo._v_attrs.zbarLocal = 0.0
hdf5_file.root.runInfo._v_attrs.zbarTotal = 0.0

hdf5_file.flush()
hdf5_file.close()
