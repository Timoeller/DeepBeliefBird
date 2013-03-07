'''
Module to pre-process wav files 
- highpass (simple noise reduction)
- spectrogram
- filterbank (not used in main() )
- data compression with DCT

author: Timo Moeller
'''

import numpy as np
import scipy.signal as ss
from scipy.io import wavfile
import pylab as pl
import cPickle


def highpass(s,stopBand,fs):
    ''' 
    highpass signal for noise reduction
    finch songdata should be in range 350Hz to 11kHz 
    '''
    N,wn = ss.buttord(wp=2*np.pi*(stopBand+100.)/fs,ws=2*np.pi*stopBand*1.0/fs,gpass=2.0, gstop=30.0,analog=0)
    b, a = ss.butter(N, wn,btype='high',analog=0)  
    filteredsong = ss.lfilter(b, a, s)
    filteredsong = np.round(filteredsong).astype('int16') # scipy wav needs 16 bit depth
    return filteredsong

def hamming(n):
    '''
    Generate a hamming window of n points
    '''
    return 0.54 - 0.46 * np.cos(2 * np.pi / n * (np.arange(n) + 0.5))
    
def dctmtx(n):
    """
    Return the DCT-II matrix of order n
    """
    x,y = np.meshgrid(range(n), range(n))
    D = np.sqrt(2.0/n) * np.cos(np.pi * (2*x+1) * y / (2*n))
    D[0] /= np.sqrt(2)
    return D

def compute_spectrogram(s,window,nfft,fs,hopfactor=2):
    '''
    split signal into frames and compute fft
    len(window) <= nfft
    nfft should be power of 2
    hopfactor describes stepsize ( len(window)/hopfactor )
    '''
    wl = len(window)
    frame_shift=np.round(wl/hopfactor)
    frames=[]
    for i in range( ((s.shape[0]-wl) / frame_shift) +1):
        windowed = s[i*frame_shift:i*frame_shift+wl]*window
        #windowed[1:] -= windowed[:-1] * 0.95 # pre-emphasis, basically high pass filtering like a douche (signal should already be highpass filtered)
        power_spectrum = (np.abs(np.fft.fft(windowed, nfft)[:nfft/2+1]))**2
        frames.append(power_spectrum)
    frames = np.row_stack(frames) 
    
    return frames

def trfbank(fs, nfft, lowfreq, linsc, logsc, nlinfilt, nlogfilt):
    ''''
    Compute triangular filterbank
    only good if transforming fft on to log (or mel) scale. 
    pure linear scale --> kind of data compression, better with DCT (inverse DCT works nice)
    '''
    # Total number of filters
    nfilt = nlinfilt + nlogfilt

    #------------------------
    # Compute the filter bank
    #------------------------
    # Compute start/middle/end points of the triangular filters in spectral
    # domain
    freqs = np.zeros(nfilt+2)
    freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linsc
    freqs[nlinfilt:] = freqs[nlinfilt-1] * logsc ** np.arange(1, nlogfilt + 3)
    heights = 2./(freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = np.zeros((nfilt, nfft))
    # FFT bins (in Hz)
    nfreqs = np.arange(nfft) / (1. * nfft) * fs
    for i in range(nfilt):
        low = freqs[i]
        cen = freqs[i+1]
        hi = freqs[i+2]

        lid = np.arange(np.floor(low * nfft / fs) + 1,
                        np.floor(cen * nfft / fs) + 1, dtype=np.int)
        lslope = heights[i] / (cen - low)
        rid = np.arange(np.floor(cen * nfft / fs) + 1,
                        np.floor(hi * nfft / fs) + 1, dtype=np.int)
        rslope = heights[i] / (hi - cen)
        fbank[i][lid] = lslope * (nfreqs[lid] - low)
        fbank[i][rid] = rslope * (hi - nfreqs[rid])

    return fbank, freqs

def main(song,fs,hpFreq=250,nfft=1024,hopfactor=2,filterbank=False,numCoeffs= 30,plotting=False,DCT=True):
    '''
    function that performs preprocessing on audiodata (.wav) in song by
    1. highpass filtering with butterworth filter (noise reduction)
    2. compute spectrogram:  - window (hamming) of size nfft
                            - fft of size nfft
                            - take only square of positive Frequencies (power spectrum)
                            - take log
    3. apply filterbank (optional)
    4. apply discrete cosine transform as data compression, use only numCoeffs coefficients (similar to PCA)
    5. zero mean and normalize each coefficient over time
                            
    returns
    newspec: processed data
    invD: inverse DCT matrix
    mu,sigma: used for zero mean and normalization
    '''
    #highpass
    filteredsong=highpass(song,hpFreq,fs)
    #spectrogram
    window=hamming(nfft)
    spectrogram= compute_spectrogram(filteredsong,window,nfft,fs,hopfactor=hopfactor)
    spectrogram[spectrogram < 1e-100]=1e-100 #log 0 avoid
    spectrogram=np.log(spectrogram)
    #filterbank
    if filterbank:
        # we want to filter frequencies for zebra finch from 250Hz to 11kHz
        lowfreq=250
        highfreq=11000
        
        nlinfilt=30
        linsc=(highfreq*1.0-lowfreq)/(nlinfilt-1)
        logsc=1.
        nlogfilt=0
        filters= trfbank(fs, nfft, lowfreq, linsc, logsc, nlinfilt, nlogfilt)
        triF=filters[0][:,:nfft/2+1].T
        triF=triF/np.sum(triF[:,0])
        
        spectrogram=np.dot(spectrogram,triF)
        
    #DCT
    if DCT:
        D = dctmtx(spectrogram.shape[1])[0:numCoeffs,:]
        invD = np.linalg.inv(dctmtx(spectrogram.shape[1]))[:,0:numCoeffs]
        newspec = np.dot(D,spectrogram.T).T
    
    #normalization of each channel through time
    mu = np.mean(newspec,axis=0)
    sigma=np.std(newspec,axis=0)
    newspec=(newspec-mu)/sigma
    
    

    
    
    if plotting:
        PINVtriF=np.linalg.pinv(triF)
        #INVtrifilt[np.isnan(INVtrifilt)]=0
        #trispec= np.dot(spectrogram,triF)
        oldtrispec= np.dot(spectrogram,PINVtriF)
        oldspec = np.dot(invD,((newspec*sigma)+mu).T).T
    
        pl.figure()
        pl.subplot(4,1,1)
        pl.plot(np.arange(0,filteredsong.shape[0]*1.0/fs,1.0/fs),filteredsong)
        pl.title('Highpass filtered Song',fontsize=30)
        #pl.xlabel('time [s]')
        pl.xticks([])
        pl.yticks([])
        
        pl.subplot(4,1,2)
        pl.imshow((spectrogram).T,origin='lower',aspect='auto')
        pl.title('filtered spectrogram',fontsize=30)
        pl.ylabel('nfft bin')
        pl.xticks([])
        
        pl.subplot(4,1,3)
        pl.imshow(oldspec.T,origin='lower',aspect='auto')
        pl.title('Cepstrum inverse',fontsize=30)
        #pl.xlabel('frame number')
        pl.ylabel('Coefficient')
        pl.xticks([])
        
        pl.subplot(4,1,4)
        pl.imshow(newspec.T,origin='lower',aspect='auto')
        pl.title('cepstrum',fontsize=30)
        #pl.xlabel('frame number')
        pl.ylabel('Coefficient')
        pl.xticks([])
        
        
        pl.show()
    
    if filterbank:
        return newspec,invD,mu,sigma,triF
    else:
        return newspec,invD,mu,sigma
    
if __name__ == '__main__':
    inputfile= 'test1.wav'
    [fs, songwhole]=wavfile.read(inputfile)
    if len(songwhole.shape)==2:
        song=np.mean(songwhole[:],axis=1)
    else:
        song = songwhole[5*fs:15*fs]
    #===========================================================================
    # pklfile=open('38_concat.pkl','rb')
    # temp=cPickle.load(pklfile)
    # song=temp[0]
    # fs=temp[1]
    #===========================================================================
    test_data,invD,mu,sigma= main(song,fs,hpFreq=250,nfft=1024,hopfactor=2,filterbank=True,numCoeffs= 12,plotting=True,DCT=True)
    






