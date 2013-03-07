import pylab as pl
from scipy.io import wavfile
import sys
import numpy as np
import theano
import cPickle

sys.path.insert(0, '//home/timom/git/deep')
from models.tadbn import TADBN

sys.path.insert(0, '//home/timom/git/DeepBeliefBird/')
sys.path.append('..')
import SongFeatures.preprocessing as pp
import SongFeatures.birdsongUtil as bsu
import reconstruct_generated as recon

''' 
parameter specs
'''
inputpath= '../SongFeatures/Motifs/1189/'
nfft_list=[1024]
delay_list = [1]
hidden_layers_sizes_list =[[300]]
sparse_list=[0.05]


for nfft in nfft_list:
    for delay in delay_list:
        for hidden_layers_sizes in hidden_layers_sizes_list:
            for sparse in sparse_list:
                
                savedname= '%i_%i_%i_%.3f_FB_1189' %(nfft,delay,hidden_layers_sizes[0],sparse) 
                print savedname
                songs,fs,filenames=bsu.readSongs(inputpath)  
                
                seqlen=[]
                for i,song in enumerate(songs):
                    cepstrum,invD,mu,sigma,triF= pp.main(song,fs,hpFreq=250,nfft=nfft,hopfactor=2,filterbank=True,numCoeffs=12,DCT=True)
                    seqlen.append(cepstrum.shape[0])
                    if i == 0:
                        test_data=cepstrum
                    else:
                        
                        test_data = np.concatenate((test_data,cepstrum))
                
                #print 'time slizes: %i || input dimensions: %i || window size:%i' %(test_data.shape[0],test_data.shape[1],nfft)
                
                
                batchdata = np.asarray(test_data, dtype=theano.config.floatX)
                
                
                #seqlen = [ test_data.shape[0] ]  #for multiple files after one another: takes as first input the signal at later time, so delayed training corresponds
                
                numpy_rng = np.random.RandomState(123)
                n_dim = [test_data.shape[1]]
                
                dbn_tadbn = TADBN(numpy_rng=numpy_rng, n_ins=[n_dim],
                         hidden_layers_sizes=hidden_layers_sizes,
                         sparse=sparse, delay=delay, learning_rate=0.01)
                
                dbn_tadbn.pretrain(batchdata, plot_interval=5, static_epochs=80,
                                  save_interval=10, ae_epochs=80, all_epochs=50,
                                  batch_size=5,seqlen=seqlen)
                
                output = open('trained_models/' + savedname + '.pkl', 'wb')
                cPickle.dump(dbn_tadbn, output)
                output.close()
                #===============================================================================
                # 
                # # sanity check
                # generated_series = dbn_tadbn.generate(batchdata, n_samples=300)[0,:,:]
                # output = open('output/gen_' + savednamed + '.pkl', 'wb')
                # cPickle.dump([generated_series,test_data,delay,hidden_layers_sizes,invD,mu,sigma], output)
                # output.close()
                # recon.main(generated_series,test_data,invD,mu,sigma,savednamed,delay,hidden_layers_sizes,plotting=True)
                #===============================================================================


