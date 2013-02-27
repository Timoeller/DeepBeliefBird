import pylab as pl
from scipy.io import wavfile

import sys
sys.path.insert(0, '//home/timom/git/deep')
from models.tadbn import TADBN

import numpy as np
import theano
import cPickle

import reconstruct_generated as recon

sys.path.insert(0, '//home/timom/git/DeepBeliefBird/SongFeatures')
import preprocessing as pp
import birdsongUtil as bsu



#===============================================================================
# inputfile= 'test1.wav'
# [fs, songwhole]=wavfile.read(inputfile)
# if len(songwhole.shape)==2:
#    song=np.mean(songwhole[:],axis=1)
# else:
#    song = songwhole[5*fs:15*fs]
#===============================================================================
inputpath= '//home/timom/git/DeepBeliefBird/SongFeatures/Motifs/1189'

songs,fs,filenames=bsu.readSongs(inputpath)  

savednamed= '512_12_1189_concat' 
test_data,invD,mu,sigma,triF= pp.main(songs,fs,hpFreq=250,nfft=512,hopfactor=2,filterbank=True,numCoeffs= 12,DCT=True)

print 'time slizes: %i || input dimensions: %i || window size:%i' %(test_data.shape[0],test_data.shape[1],(invD.shape[0]-1)*2)

oldspec = np.dot(invD,((test_data*sigma)+mu).T)
pl.figure()
pl.imshow(oldspec,origin='lower',aspect='auto')
pl.show()


batchdata = np.asarray(test_data, dtype=theano.config.floatX)

delay = 30
#seqlen = [ test_data.shape[0] ]  #for multiple files after one another: takes as first input the signal at later time, so delayed training corresponds
hidden_layers_sizes = [10]
numpy_rng = np.random.RandomState(123)
n_dim = [test_data.shape[1]]

dbn_tadbn = TADBN(numpy_rng=numpy_rng, n_ins=[n_dim],
         hidden_layers_sizes=hidden_layers_sizes,
         sparse=0.0, delay=delay, learning_rate=0.01)

dbn_tadbn.pretrain(batchdata, plot_interval=5, static_epochs=80,
                  save_interval=10, ae_epochs=80, all_epochs=50,
                  batch_size=5,seqlen=seqlen)

generated_series = dbn_tadbn.generate(batchdata, n_samples=300)[0,:,:]

output = open('output/gen_' + savednamed + '.pkl', 'wb')
cPickle.dump([generated_series,test_data,delay,hidden_layers_sizes,invD,mu,sigma], output)
output.close()

output = open('trained_models/' + savednamed + '.pkl', 'wb')
cPickle.dump(dbn_tadbn, output)
output.close()

recon.main(generated_series,test_data,invD,mu,sigma,savednamed,delay,hidden_layers_sizes,plotting=True)

#===============================================================================
# plt.figure()
# plt.subplot(211)
# plt.plot(test_data[:generated_series.shape[1]])
# plt.subplot(212)
# plt.plot(generated_series[0])
# plt.show()
#===============================================================================
pl.show()
