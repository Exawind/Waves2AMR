using DelimitedFiles
using LinearAlgebra
using Plots
using FFTW

function get_input_spectrum_amplitude(s_input,s_output,s_target)

    """
    Eq.(3) from paper
    """
    s_input = vec(s_input)
    s_output = vec(s_output)
    s_target = vec(s_target)

    # define amplitudes
    ampli_in = abs.(s_input)
    ampli_out = abs.(s_output)
    ampli_tg = abs.(s_target)
    # Multiply for amplitudes
    a_input_new = ampli_in.*(ampli_tg./ampli_out) # ampli_out should be computed close to wavemaker position (amplitude matching)
    #a_input_new ./= 141.42 # normalizing by the frequency resolution to try and recover correct power information. See https://www.hep.ucl.ac.uk/~rjn/saltStuff/fftNormalisation.pdf for reference

    return vec(a_input_new)
end

function get_input_spectrum_phase(s_input,s_output,s_target)

    """
    Eq.(3) from paper
    """
    s_input = vec(s_input)
    s_output = vec(s_output)
    s_target = vec(s_target)

    # Subtract for phases # phi_out should be computed at target position (phase focussing)
    phase_input_new = angle.(s_input) .+ (angle.(s_target) .- angle.(s_output))

    return vec(phase_input_new)
end

function get_input_spectrum(s_input,s_output,s_target)

    """
    Eq.(3) from paper
    """
    s_input = vec(s_input)
    s_output = vec(s_output)
    s_target = vec(s_target)

    # define amplitudes
    ampli_in = abs.(s_input)
    ampli_out = abs.(s_output)
    ampli_tg = abs.(s_target)
    # Multiply for amplitudes
    a_input_new = ampli_in.*(ampli_tg./ampli_out) # ampli_out should be computed close to wavemaker position (amplitude matching)
    #a_input_new ./= 141.42 # normalizing by the frequency resolution to try and recover correct power information. See https://www.hep.ucl.ac.uk/~rjn/saltStuff/fftNormalisation.pdf for reference

    # Subtract for phases # phi_out should be computed at target position (phase focussing)
    phase_input_new = angle.(s_input) .+ (angle.(s_target) .- angle.(s_output))


    # compute new input spectrum
    s_input_new = (a_input_new.*exp.(1im.*phase_input_new))

    println("Max amplitude for input spectrum from four-wave = ",maximum(abs.(s_input_new)))

    return vec(s_input_new)
end

function write_wavemaker_file(s_input,i_shift,frequencies_experiment)

    phase_original = angle.(s_input)
    phase_shift = zeros(size(phase_original))

    s_input = vec(s_input)
    frequencies_experiment = vec(frequencies_experiment)

    # Implementing the shifts in a floating-point-arithmetics friendly form
    if i_shift==0
        phase_shift = exp.(1im.*phase_original) # no shift
    elseif i_shift==1
        phase_shift = 1im.*exp.(1im.*phase_original) # shift by pi/2
    elseif i_shift==2
        phase_shift = -exp.(1im.*phase_original) # shift by pi
    else
        phase_shift = -1im.*exp.(1im.*phase_original) # shift by 3pi/2
    end
    #println("phase_shift_" * string(i_shift) * " = ",phase_shift)

    # define shifted spectrum
    s_input_new = (abs.(s_input).*phase_shift)
    # plot as a check # something is off. They all appear to be the same
    #s_plot = plot(frequencies_experiment,abs.(s_input_new),xlims=(0,10))
    #savefig(s_plot,"spectrum_shift_" * string(i_shift) * ".pdf")

    # df from wavemaker.cfg
    df = 38.0/(2.0^16)
    # frequencies for the wavemaker
    frequencies_list = zeros(size(s_input))
    for ii=0:length(frequencies_list)-1
        if (ii*df<(90.0)) # cutoff frequency
            frequencies_list[ii+1] = ii*df
        else
            break
        end
    end
    max_location = findmax(frequencies_list)
    max_index = max_location[2]

    # create the wavemaker array
    # to allow for mixed types to be written to the same array
    harmonics = Vector{Real}(undef,max_index)
    for jj=1:max_index
        harmonics[jj] = jj-1 # harmonics (integers starting at 0)
    end
    wavemaker_data = zeros(max_index,4) # Note: column three will be all zeros (angles)
    wavemaker_data[:,1] = frequencies_list[1:max_index] # frequencies
    wavemaker_data[:,2] = abs.(s_input_new[1:max_index])/141.42 # amplitudes
    wavemaker_data[:,4] = angle.(s_input_new[1:max_index]) # phases

    # building the wavemaker file
    wavemaker = hcat(harmonics,wavemaker_data)

    writedlm("wavemaker.dat",wavemaker,',')

end

function generate_4wave_out_spectrum(s_0, s_1, s_2, s_3)

    # No need to actually solve the linear system
    s_0 = vec(s_0)
    s_1 = vec(s_1)
    s_2 = vec(s_2)
    s_3 = vec(s_3)

    #a_0 = zeros(size(s_0))
    a_1z = zeros(size(s_0))

    #a_0 = (s_0.+s_1.+s_2.+s_3)./4.0
    a_1z = (s_0.-(1im.*s_1).-(s_2).+(1im.*s_3))./4.0

    println("Max amplitude for output spectrum from four-wave = ",maximum(abs.(a_1z)))
    #s_out = a_0 + a_1z
    s_out = a_1z

    return vec(s_out)

end

function generate_2wave_out_spectrum(s_0, s_1)

    # No need to actually solve the linear system
    s_0 = vec(s_0)
    s_1 = vec(s_1)

    a_1z = (s_0.-s_1)./2.0

    s_out = a_1z

    return vec(s_out)

end

# Gerchberg-Saxton
#algorithm Gerchberg–Saxton(Source, Target, Retrieved_Phase) is
#    A := IFT(Target)
#    while error criterion is not satisfied
#        B := Amplitude(Source) × exp(i × Phase(A))
#        C := FT(B)
#        D := Amplitude(Target) × exp(i × Phase(C))
#        A := IFT(D)
#    end while
#    Retrieved_Phase = Phase(A)

# https://en.wikipedia.org/wiki/Phase_retrieval
function gerchberg_saxton(timeseries_probe2,wave_exp)

    A_t = timeseries_probe2
    A = fft(A_t)
    A_iter = A
    ft = fft(wave_exp)
    phase_ft = angle.(ft)

    tol = 1e-4
    counter = 0
    while counter<10
        B = abs.(ft).*exp.(1im*angle.(A_iter))
        C = ifft(B)
        # put constraints
        phase_t = angle.(B)
        for ii in eachindex(C)
            if norm(phase_ft[ii]-phase_t[ii]) > tol
                C[ii] -= (0.8*C[ii])
            end
        end
        A_iter = fft(C)
        counter = counter + 1
    end
    A = A_iter
    return vec(angle.(A)) # I guess the idea, for us, would be: Return all of A (spectrum), feed amplitudes and phases to the wavemaker. Update timeseries probe2 with new result. Repeat until convergence (whatever that is)
end

function gerchberg_saxton_spectrum(timeseries_probe2,wave_exp)

    A_t = timeseries_probe2
    A = fft(A_t)
    A_iter = A
    B = zeros(size(A_iter))
    ft = fft(wave_exp)
    phase_ft = angle.(ft)

    tol = 1e-4
    counter = 0
    while counter<10
        B = abs.(ft).*exp.(1im*angle.(A_iter))
        C = ifft(B)
        # put constraints
        phase_t = angle.(B)
        for ii in eachindex(C)
            if norm(phase_ft[ii]-phase_t[ii]) > tol
                C[ii] -= (0.8*C[ii])
            end
        end
        A_iter = fft(C)
        counter = counter + 1
    end
    #maybe I should return B
    #A = A_iter
    return B/length(timeseries_probe2) #vec(A) # I guess the idea, for us, would be: Return all of A (spectrum), feed amplitudes and phases to the wavemaker. Update timeseries probe2 with new result. Repeat until convergence (whatever that is)
end