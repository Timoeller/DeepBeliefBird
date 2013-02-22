'''
Module to classify Birdsong data
- classify only pre processed data (cepstrum)
- learn TARBM, use propup() and classify on transformed data

author: Timo Moeller
'''

import numpy as np
import pylab as pl
import sys
import cPickle
sys.path.insert(0, '//home/timom/git/DeepBeliefBird')
import SongFeatures.BirdsongUtil as bsu
import SongFeatures.preprocessing as pp
import deep.reconstruct_generated as rg
#import deep.test_tadbn 
from sklearn import linear_model 
from sklearn import cross_validation
import theano

PEGGY_DRIFT=0.01 # in s

def createData(path,nfft=256,hopF=2,numCoeffs=30,batchsize=1,batchshift=1,method='normal',tadbn_file=None):
    '''
    method to read wav files and labels from path to generate data for classification
    in: 
        path: path to folder with .wav files and xls file (file needs specific format, read in Birdsongutil.createLabelArray())
        nfft,hopF,numCoeffs: params for pre-processing
        batchsize: how many samples (from pre processing) should be put together for classification
        mode: 'normal' or 'tarbm'
    out:
        data [num_samples, input dim]
        labels [num_samples]
    '''
    songs,fs,filenames,seqlen=bsu.readSongs(path)        
    labels= bsu.createLabelArray(path,PEGGY_DRIFT,fs=fs,windowL=nfft,hopF=hopF) 
    
    #loop through all files
    for i,filename in enumerate(filenames):
        #getting cepstrum and labels (labels <= cepstrum)
        cepstrum,invD,mu,sigma= pp.main(songs[i],fs,hpFreq=250,nfft=nfft,hopfactor=hopF,M=False,numCoeffs= numCoeffs) 
        cur_labels= np.asarray(labels[filename[:-4]])
        
        
        if method=='tadbn':
            tadbn_file = '//home/timom/git/DeepBeliefBird/deep/trained_models/1_38_concat.pkl'
            pklfile=open(tadbn_file,'rb')
            dbn_tadbn=cPickle.load(pklfile)
            #create zero array of size delay to put in front of data and cast into theano array 
            preCepstrum=np.zeros((dbn_tadbn.delay,numCoeffs))
            cepstrum=np.asarray(np.vstack((preCepstrum,cepstrum)), dtype=theano.config.floatX)
            cepstrum= dbn_tadbn.propup(cepstrum)[0,:,:]#1st Dim: output of hidden before (0) and after (1) sigmoid activation, my xp: 0 works better
            
            
        #getting batches, indexing only from 0 to len(labels) 
        #no 0 padding of labels, cause sometimes there are unmarked chirps at the end of wav file
        indexing = [np.asarray(range(batchsize))+x for x in np.arange(0,cur_labels.shape[0]-batchsize,batchshift)]

        if method =='tadbn':    
            cepstrum= cepstrum[indexing,:].reshape((-1,cepstrum.shape[1]*batchsize)) # cepstrum is in hidden space of dbn != numcoeffs
        else:
            cepstrum= cepstrum[indexing,:].reshape((-1,numCoeffs*batchsize)) # stretch batches to 1D array
        cur_labels = np.median(cur_labels[indexing,],axis=1)
            
        if i ==0:
            data=cepstrum
            targets=cur_labels
        else:
            data=np.vstack((data,cepstrum))
            targets=np.hstack((targets,cur_labels))
    targets=targets.T

    return data,targets
    
    
    
    
def normal_logit(path,batchsize=1,crossValidation=5):
    nfft=256*4
    hopF=2
    numCoeffs=30
    
    songs,fs,filenames,seqlen=bsu.readSongs(path)        
    labels= bsu.createLabelArray(path,PEGGY_DRIFT,fs=fs,windowL=nfft,hopF=hopF)      
    
    for i,filename in enumerate(filenames):
        #getting cepstrum and labels (labels mostly shorter than cepstrum)
        cepstrum,invD,mu,sigma= pp.main(songs[i],fs,hpFreq=250,nfft=nfft,hopfactor=hopF,M=False,numCoeffs= numCoeffs) 
        cur_labels= np.asarray(labels[filename[:-4]])
        
        #getting batches, indexing only with len(labels)
        indexing = [np.asarray(range(batchsize))+x for x in np.arange(0,cur_labels.shape[0]-batchsize,batchsize)] # indexing either shifting by batchsize or change to 1
        cepstrum= cepstrum[indexing,:].reshape((-1,numCoeffs*batchsize)) # stretch batches to 1D array
        cur_labels = np.median(cur_labels[indexing,],axis=1)
        if i ==0:
            data=cepstrum
            targets=cur_labels
        else:
            data=np.vstack((data,cepstrum))
            targets=np.hstack((targets,cur_labels))
    targets=targets.T
    
    logit = linear_model.LogisticRegression()
    if crossValidation:
        print 'num examples: %i || input dimensions:%i'%(data.shape[0],data.shape[1])

        print 'number of different targets: %i ' %(np.max(targets)+1) #0 is target as well
        scores = cross_validation.cross_val_score(logit, data, targets, cv=crossValidation)
        print scores
        print 'mean score: %.3f' %np.mean(scores)
        
    #===========================================================================
    # seqlen = [len(x) for x in songs]
    # print seqlen
    #===========================================================================

    print filenames
    if batchsize == 1:
        pl.figure()
        pl.imshow(np.dot(rg.unnormalize(data,mu,sigma),invD.T).T, aspect="auto", origin="lower")
        pl.plot(targets*invD.shape[0]/10)
        pl.show()


def TARBM_logit(path,crossValidation=5):
    inputfile= '//home/timom/git/DeepBeliefBird/deep/trained_models/1_38_concat.pkl'
    pklfile=open(inputfile,'rb')
    dbn_tadbn=cPickle.load(pklfile)
    nfft=256
    hopF=2
    numCoeffs=30
    songs,fs,filenames,seqlen=bsu.readSongs(path)        
    labels= bsu.createLabelArray(path,PEGGY_DRIFT,fs=fs,windowL=nfft,hopF=hopF)      
    preCepstrum=np.zeros((dbn_tadbn.delay,numCoeffs))
    for i,filename in enumerate(filenames):
        
        cepstrum,invD,mu,sigma= pp.main(songs[i],fs,hpFreq=250,nfft=nfft,hopfactor=hopF,M=False,numCoeffs= numCoeffs)
        cepstrum=np.asarray(np.vstack((preCepstrum,cepstrum)), dtype=theano.config.floatX)
        #DO TARBM propup
        tarbm_thoughts= dbn_tadbn.propup(cepstrum)[0,:,:] #1st Dim: output of hidden before (0) and after (1) sigmoid activation
        cur_labels= np.array(labels[filename[:-4]])
        if i ==0:
            data=tarbm_thoughts[:np.size(cur_labels)]
            targets=cur_labels
        else:
            data=np.vstack((data,tarbm_thoughts[:np.size(cur_labels)]))
            targets=np.hstack((targets,cur_labels))
        
    targets=targets.T
    
    #n-fold cross validation
    if crossValidation:
        print 'num examples: %i || input dimensions:%i'%(data.shape[0],data.shape[1])
        print targets.shape
        print 'number of different targets: %i ' %(np.max(targets)+1) #0 is target as well
        logit = linear_model.LogisticRegression()
        scores = cross_validation.cross_val_score(logit, data, targets, cv=crossValidation)
        print scores
        print 'mean score: %.3f' %np.mean(scores)
    
    
    
    
if __name__ == '__main__':
    path= '/home/timom/git/DeepBeliefBird/SongFeatures/Motifs/38/'
    TARBM_logit(path,crossValidation=5)
    #normal_logit(path,batchsize=3,crossValidation=5)
    data,targets=createData(path,method='tadbn')
    print data.shape
    print targets.shape
