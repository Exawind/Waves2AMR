using Plots
using Interpolations
using DelimitedFiles
using Statistics
include("FFT_smoothing.jl")
include("functions.jl")

#path_to_executable = "~/Local/Projects/GIT/HOS-NWT/bin/HOS-NWT"
t_skip = 50. # seconds (model scale)
n_4waves_max = 1 # original is 2. Just trying to debug now
tol = 1.0

# create the necessary directories
mkdir("phase_shift_0")
mkdir("phase_shift_1")
mkdir("phase_shift_2")
mkdir("phase_shift_3")

# copy the wavemaker and probe 
# No shift
cp("case3_HOS-NWT.inp","./phase_shift_0/case3_HOS-NWT.inp")
cp("wavemaker.cfg","./phase_shift_0/wavemaker.cfg")
cp("probe_1.inp","./phase_shift_0/probe_1.inp")
# shift by pi/2
cp("case3_HOS-NWT.inp","./phase_shift_1/case3_HOS-NWT.inp")
cp("wavemaker.cfg","./phase_shift_1/wavemaker.cfg")
cp("probe_1.inp","./phase_shift_1/probe_1.inp")
# shift by pi
cp("case3_HOS-NWT.inp","./phase_shift_2/case3_HOS-NWT.inp")
cp("wavemaker.cfg","./phase_shift_2/wavemaker.cfg")
cp("probe_1.inp","./phase_shift_2/probe_1.inp")
# shift by 3pi/2
cp("case3_HOS-NWT.inp","./phase_shift_3/case3_HOS-NWT.inp")
cp("wavemaker.cfg","./phase_shift_3/wavemaker.cfg")
cp("probe_1.inp","./phase_shift_3/probe_1.inp")

# Get the results from Case 4: two probes, one at the wavemaker location and one at the target location
cd("init_irregular/")
case4_time_series = readdlm("Results/probes.dat")
# first column is time, second column elevation at probe_1 and third column is elevation at probe_2
rows_to_skip = 49
case4_time_series = case4_time_series[rows_to_skip:end,1:3]
# convert from Any to Float
case4_time_series = Float64.(case4_time_series)

# get rid of elements that are duplicates (issue with time-step in HOS-NWT)
# I first noticed this when I put a warning that stated that the time-step was not constant. So the original time series had issues
time_original = Interpolations.deduplicate_knots!(case4_time_series[:,1]) # this allows for the de-duplication of data so that the elements in the array are different (essentially it avoids having repeated times) # only works in the interpolation package
itp_wavemaker = interpolate((time_original,), case4_time_series[:,2], Gridded(Linear()))
itp_target = interpolate((time_original,), case4_time_series[:,3], Gridded(Linear()))

# equispaced time array
dt = 1.0/141.421
t_final = 1777.744229648911
time = collect(0:dt:t_final)

# wave-elevation data
wave_elevation_at_wavemaker = itp_wavemaker.(time) # at wavemaker
wave_elevation_at_target = itp_target.(time) # at target

# convert to spectrum
# Define smoothing type and relevant parameters
# Options are: 
# Ensemble Averaging -> smoothing_type = 1; smoothing_parameter = 15000
# Constant Filter -> smoothing_type = 2; smoothing_parameter = 30
# Gaussian -> smoothing_type = 3; smoothing_parameter = 30
smoothing_type = 3
smoothing_parameter = -1
if smoothing_type==1
    smoothing_parameter = 100
else
    smoothing_parameter = 50
end
bSmoothing_initial = true
bPlotting_initial = true

frequencies_input, onesided_input_spectrum, initial_input_spectrum, frequencies_input_conjugates, twosided_input_spectrum = FFT_smoothing(time,wave_elevation_at_wavemaker,bSmoothing_initial,bPlotting_initial,smoothing_type,smoothing_parameter,"initial_input")
frequencies_output, onesided_output_spectrum, initial_output_spectrum,frequencies_output_conjugates, twosided_output_spectrum = FFT_smoothing(time,wave_elevation_at_target,bSmoothing_initial,bPlotting_initial,smoothing_type,smoothing_parameter,"initial_output")

# move back one directory
cd("../")

# get experiment data
rescaling = 50.
modelscale_time = sqrt(rescaling)
experimental_data = readdlm("IrregularWave.dat")
time_experiment = experimental_data[2:end,1]
# take out transient data
indices = findall(time_experiment->(time_experiment>(t_skip*modelscale_time)),time_experiment)
time_experiment = time_experiment[indices]
wave_elevation_experiment = experimental_data[2:end,2]
wave_elevation_experiment = wave_elevation_experiment[indices]
# convert to model scale, since the simulations are run in model scale
time_experiment /= modelscale_time
modelscale_length = rescaling
wave_elevation_experiment /= modelscale_length

# convert experimental time series to experimental spectrum
frequencies_experiment, onesided_experiment_spectrum, experiment_spectrum,frequencies_experiment_conjugates, twosided_experiment_spectrum = FFT_smoothing(time_experiment,wave_elevation_experiment,bSmoothing_initial,bPlotting_initial,smoothing_type,smoothing_parameter,"experiment")

# find shortest array to avoid dimensionality issues later on during vector operations
spectrum_length = 0
if length(experiment_spectrum)<=length(initial_input_spectrum)
    spectrum_length = length(experiment_spectrum)
else
    spectrum_length = length(initial_input_spectrum)
end

# cutting out what appears to be numerical artifacts at very low-frequencies from the spectrum
input_spectrum = get_input_spectrum(initial_input_spectrum[128:spectrum_length],initial_output_spectrum[128:spectrum_length],experiment_spectrum[128:spectrum_length])
input_plot = plot(frequencies_experiment[1:length(input_spectrum)],abs.(input_spectrum),xlims=(0,10))
savefig(input_plot,"check_input.pdf")

# now we have the input spectrum, and we can try to use the four-wave decomposition
for iteration_count=1:n_4waves_max
    # initialize spectrum
    old_input_spectrum = input_spectrum
    # declare the four spectra for the decomposition: s_0, s_1, s_2 and s_3 follow the notation from the relevant paper
    s_0 = zeros(size(old_input_spectrum))
    s_1 = zeros(size(old_input_spectrum))
    s_2 = zeros(size(old_input_spectrum))
    s_3 = zeros(size(old_input_spectrum))

    # frequencies
    f_0 = []
    f_1 = []
    f_2 = []
    f_3 = []

    # Loop over phase shifts
    for phase_shift_index=0:3
        # Move up to directory for shift
        cd("./phase_shift_" * string(phase_shift_index))

        # create wavemaker file
        write_wavemaker_file(old_input_spectrum, phase_shift_index, frequencies_experiment[1:length(old_input_spectrum)])

        # Run the HOS-NWT simulation
        run(`/Users/mpolimen/Local/Projects/GIT/HOS-NWT/bin/HOS-NWT case3_HOS-NWT.inp`)

        # Read in the probe file
        iteration_data = readdlm("Results/probes.dat")
        # Get relevant data
        iteration_data = iteration_data[rows_to_skip:end,1:2] # there is only one probe at the target location
        iteration_data = Float64.(iteration_data)

        # Clean up
        time_iteration = Interpolations.deduplicate_knots!(iteration_data[:,1]) # this allows for the de-duplication of data so that the elements in the array are different (essentially it avoids having repeated times) # only works in the interpolation package
        itp_iteration = interpolate((time_iteration,), iteration_data[:,2], Gridded(Linear()))
        wave_elevation_iteration = itp_iteration.(time) # evenly-spaced time series

        # Rename Results directory and wavemaker file to avoid overwriting
        cp("wavemaker.dat","./Results/wavemaker.dat") # save old wavemaker file for record keeping
        mv("Results", "Results" * string(iteration_count) * ".dat")
        mv("wavemaker.dat","wavemaker" * string(iteration_count) * ".dat")

        # Convert to spectrum
        # use 'time' instead of 'time iteration' for the eavenly-spaced time series
        if (phase_shift_index == 0) # this one should look the same as the original one
            f_00 , _ , s_0, _ , _ = FFT_smoothing(time,wave_elevation_iteration,bSmoothing_initial,true,smoothing_type,smoothing_parameter,"s_0")
            println("##### s_0 has been generated #####")
            f_0 = push!(vec(f_00))
        elseif (phase_shift_index == 1)
            f_11 , _ , s_1, _ , _ = FFT_smoothing(time,wave_elevation_iteration,bSmoothing_initial,true,smoothing_type,smoothing_parameter,"s_1")
            println("##### s_1 has been generated #####")
            f_1 = push!(vec(f_11))
        elseif (phase_shift_index == 2)
            f_22 , _ , s_2, _ , _ = FFT_smoothing(time,wave_elevation_iteration,bSmoothing_initial,true,smoothing_type,smoothing_parameter,"s_2")
            println("##### s_2 has been generated #####")
            f_2 = push!(vec(f_22))
        else
            f_33 , _ , s_3, _ , _ = FFT_smoothing(time,wave_elevation_iteration,bSmoothing_initial,true,smoothing_type,smoothing_parameter,"s_3")
            println("##### s_3 has been generated #####")
            f_3 = push!(vec(f_33))
        end
        # move back to main directory
        cd("../")
    end
    # Use four-wave decomposition algorithm to generate linear "output" spectrum
    output_spectrum = zeros(size(old_input_spectrum))
    println("##### Four-wave decomposition ######")
    output_spectrum = generate_4wave_out_spectrum(s_0, s_1, s_2, s_3) # The max amplitude returns 10^-8. . Actually, this might not be good, because this goes into the denominator of the linear relationship and so it makes the updated one blow up
    println("##### OUTPUT spectrum has been generated #####")

    # Generate new input (this one currently generates a massive amplitude)
    global input_spectrum = get_input_spectrum(old_input_spectrum, output_spectrum[1:length(old_input_spectrum)], experiment_spectrum[1:length(old_input_spectrum)])
    println("##### INPUT spectrum has been generated #####")
    input_spectrum = input_spectrum./(141.42) # see if this helps. It should not be there, however since the normalization is already being performed in the FFT.. 

    # plot the four shifted spectra
    four_spectra = plot(f_0,abs.(s_0),xlims=(0,10),label="\$s_0\$")
    plot!(f_1,abs.(s_1),label="\$s_1\$")
    plot!(f_2,abs.(s_2),label="\$s_2\$")
    plot!(f_3,abs.(s_3),label="\$s_3\$")
    savefig(four_spectra,"four_spectra_shifts_iteration_" * string(iteration_count) * ".pdf")

    # note: add plot of input spectrum as well as s_0, s_1, s_2, s_3
    plotting_length_old_input =  (length(old_input_spectrum) <= length(frequencies_experiment)) ? length(old_input_spectrum) : length(frequencies_experiment)
    plotting_length_new_input =  (length(input_spectrum) <= length(frequencies_experiment)) ? length(input_spectrum) : length(frequencies_experiment)
    plotting_length_output =  (length(output_spectrum) <= length(frequencies_experiment)) ? length(output_spectrum) : length(frequencies_experiment)

    all_spectra_plot = plot(frequencies_experiment[1:plotting_length_old_input],abs.(old_input_spectrum[1:plotting_length_old_input]),xlims=(0,10),label="Old Input Spectrum")
    plot!(frequencies_experiment[1:plotting_length_new_input],abs.(input_spectrum[1:plotting_length_new_input]),label="Input Spectrum - four-wave decomposition")
    plot!(frequencies_experiment[1:plotting_length_output],abs.(output_spectrum[1:plotting_length_output]),label="Output Spectrum - four-wave decomposition")
    savefig(all_spectra_plot,"Spectra_from_fourwave_decomp_iteration_" * string(iteration_count) * ".pdf")
end

# let us create the final wavemaker file and plot the spectrum
plotting_length_final_input =  (length(input_spectrum) <= length(frequencies_experiment)) ? length(input_spectrum) : length(frequencies_experiment)
write_wavemaker_file(input_spectrum[1:plotting_length_final_input],0,frequencies_experiment[1:plotting_length_final_input])