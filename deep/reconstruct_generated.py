'''
take generated output by DBN and reconstruct into spectogram
save as mat file to use with matlab catbox: spectogram to wav file

author: Timo Moeller
'''

import cPickle
import numpy as np
import pylab as pl
from scipy.io import savemat

def unnormalize(s,mu,sigma):
    '''
    reverse normalisation, normalization = standard variance and zero mean for each coefficient (columns) through time!!! (rows)
    '''
    return (s*sigma)+mu

def main(gen_series,test_data,invD,mu,sigma,savednamed,delay,hidden_sizes,plotting=True):
    '''
    undo preprocessing up until spectogram, generate m-file to generate WAV with CATbox V0.1
    in:
        stuff
    '''    
    gen_series=unnormalize(gen_series,mu,sigma) # unnormalise each channel with given mu[channel] sigma[channel] 
    gen_reco=np.sqrt(np.exp(np.dot(gen_series,invD.T))) #
    
    test_data=unnormalize(test_data,mu,sigma)
    data_reco=np.sqrt(np.exp(np.dot(test_data,invD.T)))
    
    recon={}
    recon['gen']=gen_reco
    recon['orig']=data_reco
    savemat('//home/timom/Downloads/catbox/CATbox v0.1/LSEE/input/' + savednamed,recon)
    
    if plotting:
        """
        Show the spectrum reconstructed from cepstum as an image.
        """
        pl.figure()
        pl.subplot(2,1,1)
        
        pl.imshow(np.log10(gen_reco.T), aspect="auto", origin="lower")
        pl.yticks(np.linspace(0,gen_reco.shape[1],5).astype('int'),(np.linspace(0,22050,5)).astype('int'))
        
        pl.title('Spectrum of Generated Data || TADBN delay:%i, num neurons:%i' %(delay,hidden_sizes[0]))
        pl.xlabel("Frame")
        pl.ylabel("Freq [Hz]")
        
        pl.subplot(2,1,2)
        
        pl.imshow(np.log10(data_reco.T), aspect="auto", origin="lower")
        pl.yticks(np.linspace(0,data_reco.shape[1],5).astype('int'),(np.linspace(0,22050,5)).astype('int'))
        pl.title("Spectrum of Training Data || window and nfft = %i || first %i DCT coeffs kept" %((invD.shape[0]-1)*2,invD.shape[1]))
        pl.xlabel("Frame")
        pl.ylabel("Freq [Hz]")
        pl.show()