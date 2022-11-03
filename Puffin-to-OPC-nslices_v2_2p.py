# -*- coding: utf-8 -*-
"""
Created first version on Wed May 15 10:38:37 2019 

@author: Racha Pongchalee
"""
# noted only x polarization of the Aperp field will be converted to OPC format
import numpy as np
import time
import tables, gc
from scipy.signal import hilbert
from scipy.fftpack import next_fast_len
import sys

filename = sys.argv[1] # retreive the base name
# filename = "D://Puffin_BUILD/Puffin_BIN/examples/simple/3D/DXFEL/dxfel_aperp_0"
h5name = filename + ".h5"
binname_x = filename + "_x.dfl"
paramname_x = filename + "_x.param"
binname_y = filename + "_y.dfl"
paramname_y = filename + "_y.param"

print ("Reading aperp file ..." + h5name + "\n")
h5f = tables.open_file(h5name, mode='r')

# Read the HDF5 file (Puffin_aperp file)
aperps = h5f.root.aperp.read()
Aperp_x = np.array(aperps[0]) # x-polarised field
Aperp_y = np.array(aperps[1]) # y-polarised field
print ("Getting file attributes ... \n")
wavelength = h5f.root.runInfo._v_attrs.lambda_r
nx = h5f.root.runInfo._v_attrs.nX
ny = h5f.root.runInfo._v_attrs.nY
nz = h5f.root.runInfo._v_attrs.nZ2
Lc = h5f.root.runInfo._v_attrs.Lc
Lg = h5f.root.runInfo._v_attrs.Lg
rho = h5f.root.runInfo._v_attrs.rho
meshsizeX = h5f.root.runInfo._v_attrs.sLengthOfElmX
meshsizeY = h5f.root.runInfo._v_attrs.sLengthOfElmY
meshsizeZ2 = h5f.root.runInfo._v_attrs.sLengthOfElmZ2
meshsizeXSI = meshsizeX*np.sqrt(Lc*Lg)
meshsizeYSI = meshsizeY*np.sqrt(Lc*Lg)
meshsizeZSI = meshsizeZ2*Lc
zsep = meshsizeZSI/wavelength

# binary_x = np.zeros((nz,ny,nx), dtype=np.complex_)
# binary_x_imag = np.zeros((nz,ny,nx))
print("Getting the complex envelope from x-field ...")
print("Processing the Hilbert transform ..")
start = time.time()
fast_len = next_fast_len(len(Aperp_x))
Aperp_x_complex = hilbert(Aperp_x, fast_len, 0)[:len(Aperp_x),:,:]
end = time.time()

# Aperp_x_hilbert = Hilbertfromfft(Aperp_x)
print("Hilbert transform x ... DONE ...   " + str(end - start) + " seconds" +"\n")
del(Aperp_x)

start = time.time()
fast_len = next_fast_len(len(Aperp_y))
Aperp_y_complex = hilbert(Aperp_y, fast_len, 0)[:len(Aperp_y),:,:]
end = time.time()

# Aperp_x_hilbert = Hilbertfromfft(Aperp_x)
print("Hilbert transform y ... DONE ...   " + str(end - start) + " seconds" +"\n")
del(Aperp_y)
h5f.close()
gc.collect()

print("Re-ordering/correcting the phase of the complex field into the OPC format")
# reorder the data to binary file for X-polarised field

start = time.time()
phasex = 1j*-(np.unwrap(np.angle(Aperp_x_complex)))
bin_x = np.reshape(np.abs(Aperp_x_complex)*np.exp(phasex),nx*ny*nz)
#bin_x = envelope_x*np.exp(1j*-(phase_x))
#bin_x = Aperp_x_complex
del(Aperp_x_complex)
end = time.time()
print("Re-order the complex field x ... DONE ...   " + str(end - start) + " seconds" +"\n")

start = time.time()
phasey = 1j*-(np.unwrap(np.angle(Aperp_y_complex)))
bin_y = np.reshape(np.abs(Aperp_y_complex)*np.exp(phasey),nx*ny*nz)
#bin_x = envelope_x*np.exp(1j*-(phase_x))
#bin_x = Aperp_x_complex
del(Aperp_y_complex)
end = time.time()
print("Re-order the complex field y ... DONE ...   " + str(end - start) + " seconds" +"\n")
gc.collect()

# dfl_x = list(itertools.chain(*zip(np.real(bin_x), np.imag(bin_x)))) 
# interleave real and imaginary part
def interArray(A, B):
    C = np.empty((A.size + B.size,), dtype=np.float64)
    C[0::2] = A
    C[1::2] = B
    return C

bin_x = interArray(np.real(bin_x), np.imag(bin_x))
print("Saving x-field to binary file ..." + " binary data length = "+ str(len(bin_x)))
start = time.time()
with open(binname_x, "wb") as f:
        bin_x.tofile(f)
del(bin_x)
f.close()
end = time.time()
print("Save file x ... DONE ...   " + str(end - start) + " seconds" +"\n")

gc.collect()

bin_y = interArray(np.real(bin_y), np.imag(bin_y))
print("Saving y-field to binary file ..." + " binary data length = "+ str(len(bin_y)))
start = time.time()
with open(binname_y, "wb") as f:
        bin_y.tofile(f)
del(bin_y)
f.close()
end = time.time()
print("Save file y ... DONE ...   " + str(end - start) + " seconds" +"\n")

# save binary file
# write the parameter file for physical interpretation
print("Generating OPC parameter file x ... ")
param_x = open(paramname_x, 'w')
param_x.write(" $optics\n")
param_x.write(" nslices = " + str(nz) +"\n")
print(" nslices = " + str(nz))
param_x.write(" zsep = " + str(zsep) +"\n")
print(" zsep = " + str(zsep))
# for 1D data or 3D
if nx-1 == 0:
    param_x.write(" mesh_x = " + str(1) +"\n")
    param_x.write(" mesh_y = " + str(1) +"\n")
else:
    param_x.write(" mesh_x = " + str(meshsizeXSI) +"\n")
    param_x.write(" mesh_y = " + str(meshsizeYSI) +"\n")

param_x.write(" npoints_x = " + str(nx) +"\n")
print(" npoints_x = " + str(nx))
param_x.write(" npoints_y = " + str(ny) +"\n")
print(" npoints_y = " + str(ny))
param_x.write(" Mx = " + str(1) +"\n")
param_x.write(" My = " + str(1) +"\n")
param_x.write(" lambda = " + str(wavelength) +"\n")
param_x.write(" field_next = 'none'" + "\n")
param_x.write(" /\n")
param_x.write(" $puffin\n")
param_x.write(" Lc = " + str(Lc) +"\n")
param_x.write(" Lg = " + str(Lg) +"\n")
param_x.write(" nX = " + str(nx) +"\n")
param_x.write(" nY = " + str(ny) +"\n")
param_x.write(" nZ2 = " + str(nz) +"\n")
param_x.write(" rho = " + str(rho) +"\n")
param_x.write(" sLengthOfElmX = " + str(meshsizeX) +"\n")
param_x.write(" sLengthOfElmY = " + str(meshsizeY) +"\n")
param_x.write(" sLengthOfElmZ2 = " + str(meshsizeZ2) +"\n")
param_x.write(" /\r")
param_x.close()

print("Generating OPC parameter file y ... ")
param_y = open(paramname_y, 'w')
param_y.write(" $optics\n")
param_y.write(" nslices = " + str(nz) +"\n")
print(" nslices = " + str(nz))
param_y.write(" zsep = " + str(zsep) +"\n")
print(" zsep = " + str(zsep))
# for 1D data or 3D
if ny-1 == 0:
    param_y.write(" mesh_x = " + str(1) +"\n")
    param_y.write(" mesh_y = " + str(1) +"\n")
else:
    param_y.write(" mesh_x = " + str(meshsizeXSI) +"\n")
    param_y.write(" mesh_y = " + str(meshsizeYSI) +"\n")

param_y.write(" npoints_x = " + str(nx) +"\n")
print(" npoints_x = " + str(nx))
param_y.write(" npoints_y = " + str(ny) +"\n")
print(" npoints_y = " + str(ny))
param_y.write(" Mx = " + str(1) +"\n")
param_y.write(" My = " + str(1) +"\n")
param_y.write(" lambda = " + str(wavelength) +"\n")
param_y.write(" field_next = 'none'" + "\n")
param_y.write(" /\n")
param_y.write(" $puffin\n")
param_y.write(" Lc = " + str(Lc) +"\n")
param_y.write(" Lg = " + str(Lg) +"\n")
param_y.write(" nX = " + str(nx) +"\n")
param_y.write(" nY = " + str(ny) +"\n")
param_y.write(" nZ2 = " + str(nz) +"\n")
param_y.write(" rho = " + str(rho) +"\n")
param_y.write(" sLengthOfElmX = " + str(meshsizeX) +"\n")
param_y.write(" sLengthOfElmY = " + str(meshsizeY) +"\n")
param_y.write(" sLengthOfElmZ2 = " + str(meshsizeZ2) +"\n")
param_y.write(" /\r")
param_y.close()

print("DONE\n")