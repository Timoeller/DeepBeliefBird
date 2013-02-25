import pylab as pl
from scipy.io import wavfile

import sys
sys.path.insert(0, '//home/timom/git/deep')
from models.tadbn import TADBN

import numpy as np
import theano
import cPickle

import reconstruct_generated as recon

sys.path.insert(0, '//home/timom/git/DeepBeliefBird/')
import SongFeatures.preprocessing as pp
import SongFeatures.birdsongUtil as bsu

''' 
paramter specs
'''
inputpath= '//home/timom/git/DeepBeliefBird/SongFeatures/Motifs/1189/'
nfft=512
delay = 25
hidden_layers_sizes = [50]
savednamed= '%i_%i_1189' %(nfft,delay) 

songs,fs,filenames=bsu.readSongs(inputpath)  

seqlen=[]
for i,song in enumerate(songs):
    cepstrum,invD,mu,sigma= pp.main(song,fs,hpFreq=250,nfft=nfft,hopfactor=2,M=False,numCoeffs= 30)
    seqlen.append(cepstrum.shape[0])
    if i == 0:
        test_data=cepstrum
    else:
        
        test_data = np.concatenate((test_data,cepstrum))

print 'time slizes: %i || input dimensions: %i || window size:%i' %(test_data.shape[0],test_data.shape[1],nfft)


batchdata = np.asarray(test_data, dtype=theano.config.floatX)


#seqlen = [ test_data.shape[0] ]  #for multiple files after one another: takes as first input the signal at later time, so delayed training corresponds

numpy_rng = np.random.RandomState(123)
n_dim = [test_data.shape[1]]

dbn_tadbn = TADBN(numpy_rng=numpy_rng, n_ins=[n_dim],
         hidden_layers_sizes=hidden_layers_sizes,
         sparse=0.0, delay=delay, learning_rate=0.01)

dbn_tadbn.pretrain(batchdata, plot_interval=5, static_epochs=80,
                  save_interval=10, ae_epochs=80, all_epochs=50,
                  batch_size=5,seqlen=seqlen)

output = open('trained_models/' + savednamed + '.pkl', 'wb')
cPickle.dump(dbn_tadbn, output)
output.close()

# sanity check
generated_series = dbn_tadbn.generate(batchdata, n_samples=300)[0,:,:]
output = open('output/gen_' + savednamed + '.pkl', 'wb')
cPickle.dump([generated_series,test_data,delay,hidden_layers_sizes,invD,mu,sigma], output)
output.close()
recon.main(generated_series,test_data,invD,mu,sigma,savednamed,delay,hidden_layers_sizes,plotting=True)


