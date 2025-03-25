import numpy as np
import pandas as pd

def waveNumber(g, omega, d):
    k0 = 1
    err = 1
    count = 0
    while (err >= 10e-8 and count <= 100):
        f0 = omega*omega - g*k0*np.tanh(k0*d)
        fp0 = -g*np.tanh(k0*d)-g*k0*d*(1-np.tanh(k0*d)*np.tanh(k0*d))
        k1 = k0 - f0/fp0
        err = abs(k1-k0)
        k0 = k1
        count += 1

    if (count >= 100):
        print('Can\'t find solution for dispersion equation!')
        exit()
    else:
        return(k0)

def read_elev(fname,time_skip,nheader,iprobe):

    """
    Extract spatio-temporal data from file
    """
    # take out the nheader row(s)
    # convert to array for data manipulation purposes
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
    elevation = spatiotemporal_data[:,1:]

    return time, elevation

def convert_to_spectrum(time, elevation):

    """
    Implement Fast-Fourier Transform
    to convert spatial-temporal data
    to frequency data
    """

    # the number of points was chosen to try and find a balance between frequency resolution and time resolution
    spectrum = np.fft.fft(elevation,100500)

    return spectrum.flatten()


def get_input_spectrum(s_input, s_output, s_target):
    """
    Eq.(3) from paper
    """

    # initalize amplitudes
    ampli_in = np.zeros(s_input.shape)
    ampli_out = np.zeros(s_output.shape)
    ampli_tg = np.zeros(s_target.shape)

    # define amplitudes
    ampli_in = np.abs(s_input)
    ampli_out = np.abs(s_output)
    ampli_tg = np.abs(s_target)
    # Multiply for amplitudes
    a_input_new = ampli_in*(ampli_tg/ampli_out) # ampli_out should be computed close to wavemaker position (amplitude matching)
    a_input_new /= (141.42) # normalizing by the frequency resolution to try and recover correct power information. See https://www.hep.ucl.ac.uk/~rjn/saltStuff/fftNormalisation.pdf for reference

    # initialize phases
    phase_input_new = np.zeros(s_input.shape)
    # Subtract for phases # phi_out should be computed at target position (phase focussing)
    phase_input_new = np.arctan2(np.imag(s_input),np.real(s_input)) + (np.arctan2(np.imag(s_target),np.real(s_target))-np.arctan2(np.imag(s_output),np.real(s_output)))


    # compute new input spectrum
    s_input_new = a_input_new*np.exp(1j*phase_input_new)

    return s_input_new.flatten()

def write_input_spectrum(s_input, i_shift, time):
    # Write to wavemaker.dat

    phase_original = np.zeros(s_input.shape)
    phase_original = np.arctan2(np.imag(s_input),np.real(s_input))
    phase_shifted = np.zeros(phase_original.shape)

    # Implementing the shifts in a floating-point arithmetics friendly form
    if i_shift==0:
        phase_shifted = np.exp(1j*phase_original) # no shift
    elif i_shift==1:
        phase_shifted = 1j*np.exp(1j*phase_original) # shift by pi/2
    elif i_shift==2:
        phase_shifted = -np.exp(1j*phase_original) # shift by pi
    else:
        phase_shifted = -1j*np.exp(1j*phase_original) # shift by 3pi/2
    
    # Define updated spectrum
    s = np.abs(s_input)*phase_shifted
    # taking out complex conjugates
    s = s[0:(len(s)//2 + 1)]

    # df from wavemaker.cfg
    df = 38./(2.**16)

    # create list of frequencies for wavemaker
    flist = np.zeros(s.shape)
    for ii in range(len(flist)):
        if ((ii*df) < np.sqrt(2.*np.pi)): # Limiting factor to avoid "big frequency" errors, based on Benchmark case https://gitlab.com/lheea/HOS-NWT/-/blob/master/Benchmark/Irreg_2D/Hs4_Tp10.dat?ref_type=heads
            flist[ii] = ii*df

    # largest frequency allowed
    idx = np.argmax(flist)
    nf = len(flist[:idx])

    # compute and normalize amplitudes
    amplitudelist = np.abs(s)/len(time)
    amplitudelist[1:] = 2.*amplitudelist[1:]

    # compute phases
    phaselist = np.arctan2(np.imag(s),np.real(s))
    phaselist *= np.sqrt(2*np.pi) # trying this based on some post processing analysis

    # creating wavemaker.dat file
    angle = 0.0
    with open("wavemaker.dat",'w+') as wave_maker:
        for ifreq in range(nf):
            line = '%d'%ifreq+',%.5e'%flist[ifreq]+',%.5e'%amplitudelist[ifreq]
            line += ',%.5e'%angle+',%.5e'%phaselist[ifreq]
            wave_maker.write(line+'\n') 


def time_series_difference(t0, elev0, t1, elev1):

    elev_diff_norm = np.max(np.abs(elev0 - elev1))
    error_vec = np.abs(elev0 - elev1)

    return elev_diff_norm, error_vec

def measure_convergence(s, s_old):

    spectrum_norm = np.max(np.abs(s-s_old))

    return spectrum_norm

def generate_4wave_out_spectrum(s0, s1, s2, s3):

    # No need to actually solve the linear system
    a0 = np.zeros(s0.shape,dtype=complex)
    a1z = np.zeros(a0.shape,dtype=complex)
    #a0 = (s0+s1+s2+s3)/4.
    a1z = (s0-1j*s1-s2+1j*s3)/4.

    #s_out = a0 + a1z
    s_out = a1z

    return s_out.flatten()