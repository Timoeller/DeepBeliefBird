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
#import deep.test_tadbn 
from sklearn import linear_model 
from sklearn import cross_validation
import theano

def normalRegressionOnSingleWindow(path):
    nfft=1024
    hopF=2
    songs,fs,filenames=bsu.readSongs(path)        
    labels= bsu.createLabelArray(path,fs=fs,windowL=nfft,hopF=hopF)      
    
    for i,filename in enumerate(filenames):
        cepstrum,invD,mu,sigma= pp.main(songs[i],fs,hpFreq=250,nfft=nfft,hopfactor=hopF,M=False,numCoeffs= 30)
        cur_labels= np.array(labels[filename[:-4]])
        if i ==0:
            data=cepstrum[:np.size(cur_labels)]
            targets=cur_labels
        else:
            data=np.vstack((data,cepstrum[:np.size(cur_labels)]))
            targets=np.hstack((targets,cur_labels))
        
    targets=targets.T
    print 'num examples: %i || input dimensions:%i'%(data.shape[0],data.shape[1])
    print 'number of different targets: %i ' %(np.max(targets)+1) #0 is target as well
    logit = linear_model.LogisticRegression()
    scores = cross_validation.cross_val_score(logit, data, targets, cv=10)
    print scores
    print 'mean score: %.3f' %np.mean(scores)


def TARBMREgressionOnSingleWindow(path):
    inputfile= '//home/timom/git/DeepBeliefBird/deep/trained_models/1_38_concat.pkl'
    pklfile=open(inputfile,'rb')
    dbn_tadbn=cPickle.load(pklfile)
    
    nfft=1024
    hopF=2
    numCoeffs=30
    songs,fs,filenames=bsu.readSongs(path)        
    labels= bsu.createLabelArray(path,fs=fs,windowL=nfft,hopF=hopF)      
    preCepstrum=np.zeros((dbn_tadbn.delay,numCoeffs))
    for i,filename in enumerate(filenames):
        print i
        cepstrum,invD,mu,sigma= pp.main(songs[i],fs,hpFreq=250,nfft=nfft,hopfactor=hopF,M=False,numCoeffs= numCoeffs)
        cepstrum=np.asarray(np.vstack((preCepstrum,cepstrum)), dtype=theano.config.floatX)
        #DO TARBM propup
        tarbm_thoughts= dbn_tadbn.propup(cepstrum)[1,:,:] #1D: output of hidden,2D shifted by delay,3D all hidden variables
        cur_labels= np.array(labels[filename[:-4]])
        if i ==0:
            data=tarbm_thoughts[:np.size(cur_labels)]
            targets=cur_labels
        else:
            data=np.vstack((data,tarbm_thoughts[:np.size(cur_labels)]))
            targets=np.hstack((targets,cur_labels))
        
    targets=targets.T
    print 'num examples: %i || input dimensions:%i'%(data.shape[0],data.shape[1])
    print targets.shape
    print 'number of different targets: %i ' %(np.max(targets)+1) #0 is target as well
    logit = linear_model.LogisticRegression()
    scores = cross_validation.cross_val_score(logit, data, targets, cv=10)
    print scores
    print 'mean score: %.3f' %np.mean(scores)
    
    
    
    
if __name__ == '__main__':
    path= '/home/timom/git/DeepBeliefBird/SongFeatures/Motifs/38/'
    TARBMREgressionOnSingleWindow(path)
    normalRegressionOnSingleWindow(path)
