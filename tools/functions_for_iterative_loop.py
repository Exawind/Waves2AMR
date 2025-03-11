import numpy as np
import scipy as sp
import pandas as pd

def read_elev(fname,time_skip,nheader,iprobe):

    """
    Extract spatio-temporal data from file
    """
    # take out the nheader row(s)
    spatiotemporal_data = pd.read_csv(fname,skiprows=nheader,sep='\s+') # hard coded until better solution is found
    # convert to array for data manipulation purposes
    spatiotemporal_data = spatiotemporal_data.to_numpy()
    spatiotemporal_data = spatiotemporal_data[:,0:iprobe+1]
    # take out transiet time part
    dt = float(spatiotemporal_data[1,0]) - float(spatiotemporal_data[0,0])
    ntimesteps_skip = int(time_skip/dt)
    spatiotemporal_data = spatiotemporal_data[(ntimesteps_skip+1):,:]
    # separate time and space data
    time = spatiotemporal_data[:,0]
    #time = time.reshape(len(time),1)
    elevation = spatiotemporal_data[:,1:]

    return time, elevation

def convert_to_spectrum(time, elevation):

    """
    Implement Fast-Fourier Transform
    to convert spatial-temporal data
    to frequency data
    """
    spectrum = np.fft.fft(elevation,100500)
    spectrum = spectrum[0:(len(spectrum)//2 + 1)]

    return spectrum.flatten()

def get_input_spectrum(s_input, s_output, s_target):

    """
    Eq.(3) from paper
    """

    # Multiply for amplitudes
    ampli_in = np.abs(s_input)
    ampli_tg = np.abs(s_target)
    ampli_out = np.abs(s_output)

    a_input_new = ampli_in*ampli_tg/ampli_out

    # Subtract for phases
    phase_input_new = np.arctan2(np.imag(s_input),np.real(s_input)) + (np.arctan2(np.imag(s_target),np.real(s_target))-np.arctan2(np.imag(s_output),np.real(s_output)))

    s_input_new = a_input_new*np.exp(1j*phase_input_new)

    return s_input_new.flatten()

# pass a time array and get a frequency resolution from there so that we can get the column of frequencies in the file
# and the index / high frequency
def write_input_spectrum(s_input, i_shift, time):
    # Write to wavemaker.dat
    # need to think about the format a bit more, but here is the idea
    s = np.zeros(s_input.shape)

    s = s_input*np.exp(1j*0.5*np.pi*i_shift)
    df = 38./(2.**16)
    f_max = 141.42/2. # based on Nyquist Criterion, this is the best we can do FFT wise for frequencies

    flist = np.zeros(s_input.shape)
    for ii in range(len(flist)):
        flist[ii] = ii*df

    amplitudelist = np.sqrt(2*np.pi)*np.abs(s)/len(time)
    amplitudelist[1:] = 2.*amplitudelist[1:]
    phaselist = np.arctan2(np.imag(s),np.real(s))
    angle = 0.0
    with open("wavemaker.dat",'w+') as wave_maker:
        for ifreq in range(nf): # in nf
            line = '%d'%ifreq+',%.5e'%flist[ifreq]+',%.5e'%amplitudelist[ifreq]
            line += ',%.5e'%angle+',%.5e'%phaselist[ifreq]
            wave_maker.write(line+'\n') 


def time_series_difference(t0, elev0, t1, elev1):

    elev_diff_norm = np.max(np.abs(elev0 - elev1))

    return elev_diff_norm

def measure_convergence(s, s_old):

    spectrum_norm = np.max(np.abs(s-s_old))

    return spectrum_norm

def generate_4wave_out_spectrum(s0, s1, s2, s3):

    # No need to actually solve the linear system
    # a0 = (s0+s1+s2+s3)/4
    a1z = (s0-1j*s1-s2+1j*s3)/4.

    s_out = a1z

    return s_out.flatten()