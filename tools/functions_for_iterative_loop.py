import numpy as np
import pandas as pd

def read_elev(fname,time_skip,nheader,nprobes):

    """
    Extract spatio-temporal data from file
    """
    # take out the nheader row(s)
    spatiotemporal_data = pd.read_csv(fname,skiprows=nheader,sep='\s+')
    # convert to array for data manipulation purposes
    spatiotemporal_data = spatiotemporal_data.to_numpy()
    # take out transiet time part
    dt = float(spatiotemporal_data[1,0]) - float(spatiotemporal_data[0,0])
    ntimesteps_skip = int(time_skip/dt)
    spatiotemporal_data = spatiotemporal_data[(ntimesteps_skip+1):,:]
    # separate time and space data
    time = spatiotemporal_data[:,0]
    elevation = spatiotemporal_data[:,1:nprobes+1]

    return time, elevation

def convert_to_spectrum(time, elevation):

    """
    Implement Fast-Fourier Transform
    to convert spatial-temporal data
    to frequency data
    """
    spectrum = np.fft.fft(elevation)
    spectrum = 2.*spectrum/len(time) # normalizing by the length of the signal
    spectrum[0] = spectrum[0]/2.
    # take only positive frequencies (first half of the signal, since the rest will be complex conjugates of the others)
    spectrum = spectrum[0:len(spectrum)//2]

    # we might need to fftshift to bring 0 frequency to the center of the signal
    # should not change much though
    # spectrum = np.fft.fftshift(spectrum) # taking it out to avoid possible issues with frequency order in wave_maker.dat file

    return spectrum.flatten()

def get_input_spectrum(s_input, s_output, s_target):

    """
    Eq.(3) from paper
    """

    # Multiply for amplitudes
    a_input_new = np.abs(s_input)*np.abs(s_target)/np.abs(s_output) 
    # Subtract for phases
    phase_input_new = np.angle(s_input) + (np.angle(s_target)-np.angle(s_output))

    s_input_new = a_input_new*np.exp(1j*phase_input_new)

    return s_input_new.flatten()

# pass a time array and get a frequency resolution from there so that we can get the column of frequencies in the file
# and the index / high frequency
def write_input_spectrum(s_input, i_shift, length_of_signal):
    # Write to wavemaker.dat
    # need to think about the format a bit more, but here is the idea
    s = np.zeros(s_input.shape)

    s = s_input*np.exp(1j*0.5*np.pi*i_shift)
    
    # and here we write s to the file
    frequencies = np.fft.fftfreq(length_of_signal)
    df = 38./(2.**16) # hard-coded for now. Correct way is to parse it from .cgf file
    flist = np.arange(0.,max(frequencies),df)
    nf = int(len(flist)/2.) # only half the frequencies matter
    amplitudelist = np.abs(s)
    phaselist = np.angle(s)
    angle = 0.0
    with open("wavemaker.dat",'w+') as wave_maker:
        for ifreq in range(nf): # in nf
            line = '%d'%ifreq+',%.5e'%flist[ifreq]+',%.5e'%amplitudelist[ifreq]
            line += ',%.5e'%angle+',%.5e'%phaselist[ifreq]
            wave_maker.write(line+'\n') 


def time_series_difference(t0, elev0, t1, elev1):
    # any norm that makes sense
    # unclear to me why we are passing the time info
    # if we are comparing the whole thing in the end

    elev_diff_norm = np.max(np.abs(elev0 - elev1))

    return elev_diff_norm

def measure_convergence(s, s_old):
    # use also any norm that makes sense

    spectrum_norm = np.max(np.abs(s-s_old))

    return spectrum_norm

def generate_4wave_out_spectrum(s0, s1, s2, s3):

    # No need to actually solve the linear system
    # a0 = (s0+s1+s2+s3)/4
    a1z = (s0-1j*s1-s2+1j*s3)/4.

    s_out = a1z

    return s_out.flatten()