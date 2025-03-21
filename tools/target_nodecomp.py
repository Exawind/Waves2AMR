import subprocess
import os
import numpy as np
from functions_for_iterative_loop import *

np.set_printoptions(threshold=np.inf)

exe = "~/HOS-NWT/bin/HOS-NWT"
ref_data = "IrregularWave.Elev"
tskip = 50. # seconds
n_1wave_max = 2
tol = 1.0

make_directories = True
update_inputs = False
if (make_directories):
    os.makedirs("no_decomp")
if (make_directories or update_inputs):
    subprocess.call("cp case3_HOS-NWT.inp ./no_decomp",shell=True)
    subprocess.call("cp wavemaker.cfg ./no_decomp",shell=True)
    subprocess.call("cp probe_1.inp ./no_decomp",shell=True)

# Do spectrum-based simulation as initial guess for loop
os.chdir("init_irregular")
subprocess.run(exe + " case4_HOS-NWT.inp",shell=True)

# Read in the probe file (has 2 probes)
nheader_probfile = 47
nprobes_probfile = 2
t_init_irr, elev_init_irr = read_elev("init_irregular/Results/probes.dat", tskip, nheader_probfile, nprobes_probfile)

# Convert to spectra
s_init_in = convert_to_spectrum(t_init_irr, elev_init_irr[:,0]) # at wavemaker (close to it)
np.savetxt("ampli_init_in_data.txt", np.abs(s_init_in))
np.savetxt("phase_init_in_data.txt", np.angle(s_init_in))

s_init_out = convert_to_spectrum(t_init_irr, elev_init_irr[:,1]) # at target
np.savetxt("ampli_init_out_data.txt", np.abs(s_init_out))
np.savetxt("phase_init_out_data.txt", np.angle(s_init_out))

# Read in target time series from experiment and scale
modelscale_time = np.sqrt(50)
nheader_experiment = 0
nprobes_experiment = 1
t_exp, elev_exp = read_elev(ref_data, tskip * modelscale_time, nheader_experiment, 1)
t_exp /= modelscale_time
modelscale_length = 50.
elev_exp /= modelscale_length

# Get target spectrum from experiment
s_exp = convert_to_spectrum(t_exp[0:len(t_init_irr)], elev_exp[0:len(t_init_irr),0])
np.savetxt("ampli_exp_spectrum_data.txt", np.abs(s_exp))
np.savetxt("phase_exp_spectrum_data.txt", np.angle(s_exp))

# Initial guess
# s_init_out has the closest spectrum to the experimental one
s_in = s_init_out
np.savetxt("ampli_input_spectrum_data.txt", np.abs(s_in))
np.savetxt("phase_input_spectrum_data.txt", np.angle(s_in))

os.chdir("./no_decomp/")
# Begin loop for single wave, no decomposition
print("Begin loop for single wave, no decomposition")
n = 0
err = 1.5 * tol
t_probe = t_init_irr
np.savetxt("time_array.txt", np.abs(t_probe))

while (n < n_1wave_max and err > tol):
    s_in_old = s_in
    n += 1

    # Write latest input spectrum to file
    write_input_spectrum(s_in_old, 0, t_probe)

    # Run the HOS-NWT simulation
    subprocess.run(exe + " case3_HOS-NWT.inp",shell=True)

    # Read in the probe file
    t_probe, elev_probe = read_elev("Results/probes.dat", tskip, nheader_probfile, 1)

    # Rename probe file and wavemaker file to avoid overwriting
    subprocess.call("cp -r Results Results" + str(n),shell=True)
    subprocess.call("rm -r Results",shell=True)
    subprocess.call("cp wavemaker.dat wavemaker" + str(n) + ".dat",shell=True)
    subprocess.call("rm wavemaker.dat",shell=True)

    # Compare time series
    wave_elev_norm, error_vec = time_series_difference(t_exp[0:(len(t_probe))], elev_exp[0:(len(t_probe)),0], t_probe[0:(len(t_probe))], elev_probe[0:(len(t_probe)),0])

    # Convert to spectrum
    s_out = convert_to_spectrum(t_probe[0:(len(t_probe))], elev_probe[0:(len(t_probe)),0])

    np.savetxt("ampli_out_iteration_data.txt", np.abs(s_out))
    np.savetxt("phase_out_iteration_data.txt", np.angle(s_out))
    subprocess.call("mv ampli_out_iteration_data.txt ampli_out_iteration_data_" + str(n) + ".txt",shell=True)
    subprocess.call("mv phase_out_iteration_data.txt phase_out_iteration_data_" + str(n) + ".txt",shell=True)

    # Generate new input spectrum
    s_in = get_input_spectrum(s_in_old, s_out, s_exp)
    np.savetxt("ampli_input_spectrum_iteration_data.txt", np.abs(s_in))
    np.savetxt("phase_input_spectrum_iteration_data.txt", np.angle(s_in))
    subprocess.call("mv ampli_input_spectrum_iteration_data.txt ampli_input_spectrum_iteration_data_" + str(n) + ".txt",shell=True)
    subprocess.call("mv phase_input_spectrum_iteration_data.txt phase_input_spectrum_iteration_data_" + str(n) + ".txt",shell=True)

    # Compare current input spectrum to old input spectrum
    err = measure_convergence(s_in, s_in_old)

    print("Iteration " + str(n) + " | Error = " + str(err))

# Print final input spectra
write_input_spectrum(s_in, 0, t_exp)

