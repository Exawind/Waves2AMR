import subprocess
import os, sys
import numpy as np
import pandas as pd
import math
from functions_for_iterative_loop import *

exe = "~/HOS-NWT/bin/HOS-NWT"
ref_data = "IrregularWave.Elev"
tskip = 50. # seconds
n_1wave_max = 1
tol = 1.0

make_directories = True
update_inputs = False
if (make_directories):
    os.makedirs("init_irregular")
    os.makedirs("no_decomp")
if (make_directories or update_inputs):
    subprocess.call("cp case4_HOS-NWT.inp ./init_irregular",shell=True)
    subprocess.call("cp probe_2.inp ./init_irregular",shell=True)
    subprocess.call("cp case3_HOS-NWT.inp ./no_decomp",shell=True)
    subprocess.call("cp wavemaker.cfg ./no_decomp",shell=True)
    subprocess.call("cp probe_1.inp ./no_decomp",shell=True)

# Do spectrum-based simulation as initial guess for loop
os.chdir("init_irregular")
subprocess.run(exe + " case4_HOS-NWT.inp",shell=True)

# Read in the probe file (has 2)
nheader_probfile = 47
nprobes_probfile = 2
t_init_irr, elev_init_irr = read_elev("Results/probes.dat", tskip, nheader_probfile, nprobes_probfile)

# Convert to spectra
s_init_in = convert_to_spectrum(t_init_irr, elev_init_irr[:,0]) # at wavemaker (close to it)
s_init_out = convert_to_spectrum(t_init_irr, elev_init_irr[:,1]) # at target
print("s_init_in = ",s_init_in)
print("s_init_out = ",s_init_out)

# Read in target time series from experiment and scale
modelscale_time = np.sqrt(50)
nheader_experiment = 0
nprobes_experiment = 1
t_exp, elev_exp = read_elev(ref_data, tskip * modelscale_time, nheader_experiment, 1)
t_exp /= modelscale_time
modelscale_length = 50.
elev_exp /= modelscale_length

# Get target spectrum from experiment
# To match size
s_exp = convert_to_spectrum(t_exp[0:(len(t_init_irr))], elev_exp[0:(len(t_init_irr)),0])

print("s_exp = ",s_exp)
s_in = s_init_in
print("s_in_start = ", s_in)

# Move to directory
os.chdir("../no_decomp/")
# Begin loop for single wave, no decomposition
print("Begin loop for single wave, no decomposition")
n = 0
err = 1.5 * tol
while (n < n_1wave_max and err > tol):
    s_in_old = s_in
    n += 1

    # Write latest input spectrum to file
    write_input_spectrum(s_in, 0, t_init_irr)

    # Run the HOS-NWT simulation
    subprocess.run(exe + " case3_HOS-NWT.inp",shell=True)

    # Read in the probe file
    if n==1:
        t_probe, elev_probe = read_elev("Results/probes.dat", tskip, nheader_probfile, 1)
        print("PROBE: ", elev_probe)
    else:
        t_probe, elev_probe = read_elev("Results/probes" + str(n-1) + ".dat", tskip, nheader_probfile, 1)

    # Rename probe file and wavemaker file to avoid overwriting
    subprocess.call("mv Results/probes.dat Results/probes" + str(n) + ".dat",shell=True)
    subprocess.call("mv wavemaker.dat wavemaker" + str(n) + ".dat",shell=True)

    # Compare time series
    wave_elev_norm = time_series_difference(t_exp[0:(len(s_in))], elev_exp[0:(len(s_in)),0], t_probe[0:(len(s_in))], elev_probe[0:(len(s_in)),0])
    print("Time series norm: " + str(wave_elev_norm))

    # Convert to spectrum
    #s_out = convert_to_spectrum(t_probe[0:(len(t_init_irr))], elev_probe[0:(len(t_init_irr)),0])

    # Generate new input spectrum
    #s_in = get_input_spectrum(s_in, s_out, s_exp)

    # Compare current input spectrum to old input spectrum
    #err = measure_convergence(s_in, s_in_old)

    #print("Iteration " + str(n) + " | Error = " + str(err))

# Print final input spectra
# write_input_spectrum(s_in, 0, t_init_irr)

