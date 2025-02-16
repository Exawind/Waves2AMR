import subprocess
import os, sys
import numpy as np
import pandas as pd
import math
from functions_for_iterative_loop import *

exe = "~/HOS-NWT/bin/HOS-NWT"
ref_data = "IrregularWave.Elev"
tskip = 50. # seconds
n_4waves_max = 2
n_1wave_max = 1
tol = 1.0

make_directories = True
update_inputs = False
if (make_directories):
    os.makedirs("init_irregular")
    os.makedirs("phase_shift0")
    os.makedirs("phase_shift1")
    os.makedirs("phase_shift2")
    os.makedirs("phase_shift3")
    os.makedirs("no_decomp")
if (make_directories or update_inputs):
    subprocess.call("cp case4_HOS-NWT.inp ./init_irregular",shell=True)
    subprocess.call("cp probe_2.inp ./init_irregular",shell=True)
    subprocess.call("cp case3_HOS-NWT.inp ./phase_shift0",shell=True)
    subprocess.call("cp wavemaker.cfg ./phase_shift0",shell=True)
    subprocess.call("cp probe_1.inp ./phase_shift0",shell=True)
    subprocess.call("cp case3_HOS-NWT.inp ./phase_shift1",shell=True)
    subprocess.call("cp wavemaker.cfg ./phase_shift1",shell=True)
    subprocess.call("cp probe_1.inp ./phase_shift1",shell=True)
    subprocess.call("cp case3_HOS-NWT.inp ./phase_shift2",shell=True)
    subprocess.call("cp wavemaker.cfg ./phase_shift2",shell=True)
    subprocess.call("cp probe_1.inp ./phase_shift2",shell=True)
    subprocess.call("cp case3_HOS-NWT.inp ./phase_shift3",shell=True)
    subprocess.call("cp wavemaker.cfg ./phase_shift3",shell=True)
    subprocess.call("cp probe_1.inp ./phase_shift3",shell=True)
    subprocess.call("cp case3_HOS-NWT.inp ./no_decomp",shell=True)
    subprocess.call("cp wavemaker.cfg ./no_decomp",shell=True)
    subprocess.call("cp probe_1.inp ./no_decomp",shell=True)

# Do spectrum-based simulation as initial guess for loop
os.chdir("init_irregular")
subprocess.run(exe + " case4_HOS-NWT.inp",shell=True)

# Read in the probe file (has 2)
nheader_probfile = 1
nprobes_probfile = 2
t_init_irr, elev_init_irr = read_elev("Results/probes.dat", tskip, nheader_probfile, nprobes_probfile)

# Convert to spectra
s_init_in = convert_to_spectrum(t_init_irr, elev_init_irr[:,0])
s_init_out = convert_to_spectrum(t_init_irr, elev_init_irr[:,1])

# Compute dt of simulations
dt = t_init_irr[1] - t_init_irr[0]
length_of_signal = len(t_init_irr)

# Read in target time series from experiment and scale
modelscale_time = np.sqrt(50)
nheader_experiment = 1
nprobes_experiment= 1
t_exp, elev_exp = read_elev("../" + ref_data, tskip * modelscale_time, nheader_experiment, nprobes_experiment)
t_exp /= modelscale_time
modelscale_length = 50.
elev_exp /= modelscale_length

# Get target spectrum from experiment
#s_exp = convert_to_spectrum(t_exp, elev_exp)
# To match size
s_exp = convert_to_spectrum(t_exp[0:(len(t_init_irr))], elev_exp[0:(len(t_init_irr))])

# For first input spectrum for loops
s_in = get_input_spectrum(s_init_in, s_init_out, s_exp)

# Begin loop for 4-wave decomposition
print("Begin loop for 4-wave decomposition, n_max = " + str(n_4waves_max))
n = 0
err = 1.5 * tol
while (n < n_4waves_max and err > tol):
    s_in_old = s_in
    n += 1
    # Loop over phase shifts
    for i in range(4):
        # Move to directory
        os.chdir("../phase_shift"+ str(i))

        # Add phase shift and write to file
        # pass also a time array to get frequency info
        # write_input_spectrum(s_in, i)
        write_input_spectrum(s_in, i, length_of_signal)

        # Run the HOS-NWT simulation
        subprocess.run(exe + " case3_HOS-NWT.inp",shell=True)

        # Read in the probe file
        t_probe, elev_probe = read_elev("Results/probes.dat", tskip, 1, 1)

        # Rename probe and wavemaker files to avoid overwriting
        subprocess.call("mv Results/probes.dat Results/probes" + str(n) + ".dat",shell=True)
        subprocess.call("mv wavemaker.dat wavemaker" + str(n) + ".dat",shell=True)

        # If phase shift = 0, compare time series to target
        if (i == 0):
            wave_elev_norm = time_series_difference(t_exp, elev_exp, t_probe, eleve_probe)
            print("Time series norm: " + str(wave_elev_norm))

        # Convert to spectrum
        if (i == 0):
            s0 = convert_to_spectrum(t_probe, elev_probe)
        elif (i == 1):
            s1 = convert_to_spectrum(t_probe, elev_probe)
        elif (i == 2):
            s2 = convert_to_spectrum(t_probe, elev_probe)
        else:
            s3 = convert_to_spectrum(t_probe, elev_probe)

    # Use 4 probe time series to generate linear "out" spectrum
    s_out = generate_4wave_out_spectrum(s0, s1, s2, s3)

    # Generate new input spectrum
    s_in = get_input_spectrum(s_in, s_out, s_exp)

    # Compare current input spectrum to old input spectrum
    err = measure_convergence(s_in, s_in_old)

    print("Iteration " + str(n) + " | Error = " + str(err))

# Move to directory
subprocess.run("cd ../no_decomp")
# Begin loop for single wave, no decomposition
print("Begin loop for single wave, no decomposition")
n = 0
err = 1.5 * tol
while (n < n_1wave_max and err > tol):
    s_in_old = s_in
    n += 1

    # Write latest input spectrum to file
    #write_input_spectrum(s_in, 0)
    write_input_spectrum(s_in, 0, length_of_signal)

    # Run the HOS-NWT simulation
    subprocess.run(exe + " case3_HOS-NWT.inp",shell=True)

    # Read in the probe file
    t_probe, elev_probe = read_elev("Results/probes.dat", tskip, 1, 1)

    # Rename probe file and wavemaker file to avoid overwriting
    subprocess.call("mv Results/probes.dat Results/probes" + str(n) + ".dat",shell=True)
    subprocess.call("mv wavemaker.dat wavemaker" + str(n) + ".dat",shell=True)

    # Compare time series
    wave_elev_norm = time_series_difference(t_exp, elev_exp, t_probe, eleve_probe)
    print("Time series norm: " + str(wave_elev_norm))

    # Convert to spectrum
    s_out = convert_to_spectrum(t_probe, elev_probe)

    # Generate new input spectrum
    s_in = get_input_spectrum(s_in, s_out, s_exp)

    # Compare current input spectrum to old input spectrum
    err = measure_convergence(s_in, s_in_old)

    print("Iteration " + str(n) + " | Error = " + str(err))

# Print final input spectra
#write_input_spectrum(s_in, 0)
write_input_spectrum(s_in, 0, dt, length_of_signal)

