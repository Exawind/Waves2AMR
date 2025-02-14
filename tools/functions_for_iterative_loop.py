import numpy as np

def read_elev(fname,time_skip,nheader,nprobes):

    """
    Extract spatio-temporal data from file
    """
    spatiotemporal_data = np.loadtxt(fname, skiprows=nheader)
    # convert to array to be able to manipulate it more easily
    spatiotemporal_data = np.array(spatiotemporal_data)
    # take out the nheader row(s)
    #spatiotemporal_data = spatiotemporal_data[nheader:,:]
    # take out transiet time part
    spatiotemporal_data = spatiotemporal_data[time_skip:,:]
    # from first column you get dt (time[1]-time[0])
    # and then time_skip is in seconds
    # and so you get the index of the first element that you want
    # separate time and space data
    time = spatiotemporal_data[:,0]
    elevation = spatiotemporal_data[:,1:nprobes]

    return time, elevation

def convert_to_spectrum(time, elevation):

    """
    Implement Fast-Fourier Transform
    to convert spatial-temporal data
    to frequency data
    """
    spectrum = np.fft.fft(elevation,len(time))
    spectrum = 2.*spectrum/len(time) # normalizing by the length of the signal
    spectrum[0] = spectrum[0]/2.
    # take only positive frequencies (first half of the signal, since the rest will be complex conjugates of the others)
    spectrum = spectrum[0:len(spectrum)//2]

    # we might need to fftshift to bring 0 frequency to the center of the signal
    # should not change much though
    spectrum = np.fft.fftshift(spectrum)

    return spectrum

def get_input_spectrum(s_input, s_output, s_target):

    """
    Eq.(3) from paper
    """

    # Multiply for amplitudes
    a_input_new = np.abs(s_input)*np.abs(s_target)/np.abs(s_output) 
    # Subtract for phases
    phase_input_new = np.angle(s_input) + (np.angle(s_target)-np.angle(s_output))

    s_input_new = a_input_new*np.exp(phase_input_new)

    return s_input_new

# pass a time array and get a frequency resolution from there so that we can get the column of frequencies in the file
# and the index / high frequency
def write_input_spectrum(s_input, i_shift, dt, length_of_signal):
    # Write to wavemaker.dat
    # need to think about the format a bit more, but here is the idea
    s = np.zeros(s_input.shape)

    if i_shift==0:
        s = s_input #s0
    elif i_shift==1:
        s = 0.5*np.pi*s_input #s1
    elif i_shift==2:
        s = np.pi*s_input #s2
    else:
        s = 1.5*np.pi*s_input #s3
    
    # and here we write s to the file
    # are we writing n_harmo, frequency, amplitude, angle and phase?
    df = 1./dt
    flist = np.linspace(0.,length_of_signal,df)
    nf = len(flist)
    amplitudelist = np.abs(s)
    phaselist = np.angle(s)
    with open(WritePath+".dat","w") as wave_maker:
        for ifreq in range(nf): # in nf
            line = '%d'%ifreq+',%.3f'%flist[ifreq]+',%.3f'%amplitudelist[ifreq]
            line += ',%.2f'%anglelist[ifreq]=0+',%.2f'%phaselist[ifreq]
            wave_maker.write(line+'\n') 


def time_series_difference(t0, elev0, t1, elev1):
    # any norm that makes sense
    # unclear to me why we are passing the time info
    # if we are comparing the whole thing in the end

    elev_diff_norm = np.abs(elev0 - elev1)

    return elev_diff_norm

def measure_convergence(s, s_old):
    # use also any norm that makes sense

    spectrum_norm = np.abs(s-s_old)

    return spectrum_norm

def generate_4wave_out_spectrum(s0, s1, s2, s3):

    # No need to actually solve the linear system
    # a0 = (s0+s1+s2+s3)/4
    a1z = (s0-1j*s1-s2+1j*s3)/4.

    s_out = a1z

    return s_out