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
import SongFeatures.birdsongUtil as bsu
import SongFeatures.preprocessing as pp
import deep.reconstruct_generated as rg
#import deep.test_tadbn 
from sklearn import linear_model 
from sklearn import cross_validation
import theano
import os

PEGGY_DRIFT=0.01 # in s

def createData(path,nfft=1024,hopF=2,numCoeffs=12,batchsize=1,batchshift=1,method='normal',tadbn_file=None,filterbank=True):
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
    songs,fs,filenames=bsu.readSongs(path)        
    labels= bsu.createLabelArray(path,PEGGY_DRIFT=PEGGY_DRIFT,fs=fs,windowL=nfft,hopF=hopF) 
    
    #loop through all files
    for i,filename in enumerate(filenames):
        #getting cepstrum and labels (labels <= cepstrum)
        cepstrum,invD,mu,sigma,triF= pp.main(songs[i],fs,hpFreq=250,nfft=nfft,hopfactor=hopF,filterbank=filterbank,numCoeffs= numCoeffs) 
        cur_labels= np.asarray(labels[filename[:-4]])
        
        
        if method == 'tadbn' or method == 'tarbm':    
            if not(tadbn_file):
                tadbn_file='/home/timom/git/DeepBeliefBird/deep/trained_models/old/512_25_1189.pkl'
            if len(tadbn_file) < 35:
                tadbn_file='/home/timom/git/DeepBeliefBird/deep/trained_models/' + tadbn_file
            pklfile=open(tadbn_file,'rb')
            dbn_tadbn=cPickle.load(pklfile)
            #create zero array of size delay to put in front of data and cast into theano array 
            preCepstrum=np.zeros((dbn_tadbn.delay,numCoeffs))
            theano.config.floatX='float32'
            cepstrum=np.asarray(np.vstack((preCepstrum,cepstrum)), dtype=theano.config.floatX)
            cepstrum= dbn_tadbn.propup(cepstrum)[1,:,:]#1st Dim: output of hidden before (0) and after (1) sigmoid activation
            
            
        #getting batches, indexing only from 0 to len(labels) 
        #no 0 padding of labels, cause sometimes there are unmarked chirps at the end of wav file
        indexing = [np.asarray(range(batchsize))+x for x in np.arange(0,cur_labels.shape[0]-batchsize,batchshift)]

        if method == 'tadbn' or method == 'tarbm':    
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
    
def doLogit(data,targets):
    logit = linear_model.LogisticRegression()
    
    num_samples = data.shape[0]
    num_training=num_samples-200#int(num_samples/1.5)
    training=data[0:num_training,:]
    trainTargets=targets[0:num_training]
    

    
    logit.fit(training,trainTargets)
    prediction=logit.predict(data)
    
    
    print 'training correct = %.3f' %(np.sum(prediction[:num_training] == targets[:num_training])/(num_training*1.0))
    pl.figure()
    for i in range(num_training-20,len(targets)):
        if i == num_training:
            pl.plot([i,i],[-1,np.max(targets)+1],label='unseen ->')
        
        if targets[i] == prediction[i]:
            pl.plot(i,targets[i],'|g',ms=15,mew=2)
        else:
            pl.plot(i,targets[i],'|b',ms=15,mew=2)
            pl.plot(i,prediction[i],'|r',ms=15,mew=2)
            
    pl.legend()
    pl.xlabel('sample number')
    pl.ylabel('class number')
    pl.title('Logistic Regression')
    pl.show()
   
def doSparseLogit(data,targets):
    lg = linear_model.LogisticRegression(penalty='l1', C=1)    
    lg.fit(data, targets)
    print lg.coef_.shape
    
    #look at coeffs and 
    out = open('logitcoeffs_1189_C=1.pkl','wb')
    cPickle.dump(lg.coef_,out)
    out.close()

    pl.imshow(lg.coef_)
    pl.colorbar()
    pl.show()
         
def loopThroughTADBN(TADBNpath):
    batchsize=3
    files = [ f for f in os.listdir(TADBNpath) if os.path.isfile(os.path.join(TADBNpath,f)) and f.endswith('.pkl')]
    allscores = {}
    for i,TADBNfile in enumerate(files):
        #get parameters from filename 1024_10_10_0.0_FB_1189.pkl
        print TADBNfile
        temp=TADBNfile.split('_',5)
        nfft=int(temp[0])
        delay=int(temp[1])
        hidden_size=int(temp[2])
        sparsity=float(temp[3])
        birdnum=int(temp[5].split('.',1)[0])
        songpath= '/home/timom/git/DeepBeliefBird/SongFeatures/Motifs/' + str(birdnum) + '/'
        
        #create DAta
        data,targets = createData(songpath,nfft=nfft,batchsize=batchsize,method='tadbn',tadbn_file=TADBNfile,filterbank=True)
        logit = linear_model.LogisticRegression(penalty='l1', C=1)
        kf = cross_validation.KFold(len(targets), k=5, shuffle=False, indices=True)
        scores = cross_validation.cross_val_score(logit, data, targets, cv=kf)
        
        #print scores
        print 'mean score: %.3f' %np.mean(scores)
        allscores[TADBNfile[0:-4]] = [scores, np.mean(scores)]
    
    allscores=sorted(allscores.items(), key=lambda t: t[1][1])
    output = open('allscores_FB_normal_batch3_aftersigmoid.pkl', 'wb')
    cPickle.dump(allscores, output)
    output.close()
    print allscores
 
if __name__ == '__main__':
    path= '/home/timom/git/DeepBeliefBird/SongFeatures/Motifs/1189/'
    tadbn_file='//home/timom/git/DeepBeliefBird/deep/trained_models/1024_25_300_0.05_FB_1189.pkl'
    data,targets=createData(path,tadbn_file =tadbn_file ,method='tadbn',nfft=1024,hopF=2,batchsize=1,filterbank=True)
    
    #TADBNpath= '/home/timom/git/DeepBeliefBird/deep/trained_models'
    #loopThroughTADBN(TADBNpath)  
    #doSparseLogit(data,targets)
    
    logit = linear_model.LogisticRegression(penalty='l1', C=1)
    kf = cross_validation.KFold(len(targets), k=5, shuffle=False, indices=True) #no shuffling! of data, cause one timepoint contains info of previous data
    scores = cross_validation.cross_val_score(logit, data, targets, cv=kf)
    print scores
    print 'mean score: %.3f' %np.mean(scores)
    
    

