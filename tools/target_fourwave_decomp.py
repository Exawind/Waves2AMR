import subprocess
import os
import numpy as np
from functions_for_iterative_loop import *

np.set_printoptions(threshold=np.inf)

exe = "~/HOS-NWT/bin/HOS-NWT"
ref_data = "IrregularWave.Elev"
tskip = 50. # seconds
n_4waves_max = 2
tol = 1.0

make_directories = True
update_inputs = False
if (make_directories):
    os.makedirs("phase_shift0")
    os.makedirs("phase_shift1")
    os.makedirs("phase_shift2")
    os.makedirs("phase_shift3")
if (make_directories or update_inputs):
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

# Do spectrum-based simulation as initial guess for loop
os.chdir("init_irregular")
subprocess.run(exe + " case4_HOS-NWT.inp",shell=True)

# Read in the probe file (has 2 probes)
nheader_probfile = 47
nprobes_probfile = 2
t_init_irr, elev_init_irr = read_elev("Results/probes.dat", tskip, nheader_probfile, nprobes_probfile)

# Convert to spectra
s_init_in = convert_to_spectrum(t_init_irr, elev_init_irr[:,0]) # at wavemaker (close to it)
s_init_out = convert_to_spectrum(t_init_irr, elev_init_irr[:,1]) # at target

# Read in target time series from experiment and scale
modelscale_time = np.sqrt(50)
nheader_experiment = 0
nprobes_experiment = 1
t_exp, elev_exp = read_elev(ref_data, tskip * modelscale_time, nheader_experiment, 1)
t_exp /= modelscale_time
modelscale_length = 50.
elev_exp /= modelscale_length

# Get target spectrum from experiment
s_exp = convert_to_spectrum(t_exp[0:(len(t_init_irr))], elev_exp[0:(len(t_init_irr)),0])

# Initial guess
# s_init_out has the closest spectrum to the experimental one
s_in = s_init_out

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
        write_input_spectrum(s_in_old, i, t_init_irr)

        # Run the HOS-NWT simulation
        subprocess.run(exe + " case3_HOS-NWT.inp",shell=True)

        # Read in the probe file
        t_probe, elev_probe = read_elev("Results/probes.dat", tskip, nheader_probfile, 1)
        
        # Rename probe and wavemaker files to avoid overwriting
        subprocess.call("cp -r Results Results" + str(n),shell=True)
        subprocess.call("rm -r Results",shell=True)
        subprocess.call("cp wavemaker.dat wavemaker" + str(n) + ".dat",shell=True)
        subprocess.call("rm wavemaker.dat",shell=True)

        # If phase shift = 0, compare time series to target
        if (i == 0):
            wave_elev_norm, error_vec = time_series_difference(t_exp[0:(len(t_init_irr))], elev_exp[0:(len(t_init_irr)),0], t_probe[0:(len(t_init_irr))], elev_probe[0:(len(t_init_irr)),0])
            print("Time series norm: " + str(wave_elev_norm))

        # Convert to spectrum
        if (i == 0):
            s0 = convert_to_spectrum(t_probe[0:(len(t_init_irr))], elev_probe[0:(len(t_init_irr)),0])
        elif (i == 1):
            s1 = convert_to_spectrum(t_probe[0:(len(t_init_irr))], elev_probe[0:(len(t_init_irr)),0])
        elif (i == 2):
            s2 = convert_to_spectrum(t_probe[0:(len(t_init_irr))], elev_probe[0:(len(t_init_irr)),0])
        else:
            s3 = convert_to_spectrum(t_probe[0:(len(t_init_irr))], elev_probe[0:(len(t_init_irr)),0])

    # Use 4 probe time series to generate linear "out" spectrum
    s_out = generate_4wave_out_spectrum(s0, s1, s2, s3)
    np.savetxt("ampli_out_4wavedecomp_iteration_data.txt", np.abs(s_out))
    np.savetxt("phase_out_4wavedecomp_iteration_data.txt", np.angle(s_out))
    subprocess.call("mv ampli_out_4wavedecomp_iteration_data.txt ampli_out_4wavedecomp_iteration_data_" + str(n) + ".txt",shell=True)
    subprocess.call("mv phase_out_4wavedecomp_iteration_data.txt phase_out_4wavedecomp_iteration_data_" + str(n) + ".txt",shell=True)

    # Generate new input spectrum
    s_in = get_input_spectrum(s_in_old, s_out, s_exp)
    np.savetxt("ampli_input_spectrum_4wavedecomp_iteration_data.txt", np.abs(s_in))
    np.savetxt("phase_input_spectrum_4wavedecomp_iteration_data.txt", np.angle(s_in))
    subprocess.call("mv ampli_input_spectrum_4wavedecomp_iteration_data.txt ampli_input_spectrum_4wavedecomp_iteration_data_" + str(n) + ".txt",shell=True)
    subprocess.call("mv phase_input_spectrum_4wavedecomp_iteration_data.txt phase_input_spectrum_4wavedecomp_iteration_data_" + str(n) + ".txt",shell=True)

    # Compare current input spectrum to old input spectrum
    err = measure_convergence(s_in, s_in_old)

    print("Iteration " + str(n) + " | Error = " + str(err))