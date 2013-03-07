import cPickle
import theano
import numpy as np
import pylab as pl
import sys
import time
from sklearn import linear_model 

sys.path.insert(0, '//home/timom/git/DeepBeliefBird')
import SongFeatures.birdsongUtil as bsu
import SongFeatures.preprocessing as pp
import Classifyem.logisticBirds as lb

        

def calcSimilarityMatrix(coef,n=5,plotting=True):
    '''
    calculates Similarity matrix based on regression coefficients. because hidden representation
    is a higher order rep and slightly different inputs should activate the same hidden unit
    in:
        coef - regression coefficients [numClasses*numhidden]
        n - n most positive and n most negative coefficients to consider
        plotting ...
    out:
        SM - Similarity Matrix
    '''
    SM = np.zeros((coef.shape[0],coef.shape[0]))   
    s=np.argsort(coef,axis=1)
    for i in range(coef.shape[0]):
        for j in range(coef.shape[0]):
            if i!=j:
                SM[i,j]=np.sum(np.in1d(s[i,-n:],s[j,-n:],True))
                SM[i,j]+=np.sum(np.in1d(s[i,:n],s[j,:n],True))
                SM[i,j]/=(n*2.0)
    
    if plotting:
        pl.figure()
        pl.imshow(SM,interpolation='nearest',cmap='Greys')
        pl.title('Similarity Matrix with Diagonal set to 0 || %i out of %i hidden units compared' %(n*2,coef.shape[1]))
        pl.colorbar()
        pl.xlabel('Class')
        pl.ylabel('Class')
        pl.show()
    return SM
     
    
def visual_frames(data,targets,showCoef=False):
    '''
    visualizes for each individual frame some interesting hidden neurons
    interesting: large regression coefficients (absolute terms)
    in:
        data: [numFrames*numHidden]
        targets: [numFrames] with class (Syllable) labels corresponding to data
        
    out:
        nuescht
    '''
    
    converter={'0':0,'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7} #0 reserved for no syllable
    inv_converter=dict([(v,k) for (k,v) in converter.items()])\
    
    
    lg = linear_model.LogisticRegression(penalty='l1', C=1)    
    lg.fit(data, targets)
    numClasses=int(np.max(targets)+1) #+1 for 0 class
    
    calcSimilarityMatrix(lg.coef_)
    
    if showCoef:
        pl.figure()
        pl.subplot(211)
        pl.imshow(lg.coef_,interpolation='nearest')
        pl.colorbar()
        
        
        
        lg = linear_model.LogisticRegression(penalty='l1', C=1)    
        lg.fit(data, targets)
        pl.subplot(212)
        pl.imshow(lg.coef_,interpolation='nearest')
        pl.colorbar()
        
        pl.show()
    

    
    
    numInteresting= 2
    int_hidden=np.argsort(lg.coef_,axis=1)
    hidden_ex= np.zeros((numInteresting,numClasses))
    hidden_inh= np.zeros((numInteresting,numClasses))
    

    
    pl.ion()
    pl.figure()
    pl.title('Hidden Neurons with large regression coefficients')
    pl.hold(False)
    
    for i in range(targets.shape[0]):
        for j in range(numClasses):
            
            hidden_ex[:,j]=data[i,int_hidden[j,-1:-numInteresting-1:-1]]
            hidden_inh[:,j]=data[i,int_hidden[j,:numInteresting]]
        pl.subplot(121)
        pl.imshow(hidden_ex.T,interpolation='nearest',cmap='Greys')
        pl.xticks(range(numInteresting),range(numInteresting))
        pl.ylabel('Classes')
        pl.xlabel('"excitatory" hidden units')
        
        #=======================================================================
        # if i == 0:
        #    pl.colorbar()
        #=======================================================================
        pl.text(-1.5,int(targets[i])+0.3,inv_converter[int(targets[i])] , bbox=dict(boxstyle="round", fc="0.8",facecolor='white', alpha=0.7),fontsize=30)
        
        pl.subplot(122)
        pl.imshow(hidden_inh.T,interpolation='nearest',cmap='Greys')
        pl.xticks(range(numInteresting),range(numInteresting))
        #pl.ylabel('Classes')
        pl.yticks([])
        pl.xlabel('"Inhibitory" hidden units')
        pl.draw()
        time.sleep(0.03)
    
    



if __name__ == '__main__':

    path= '/home/timom/git/DeepBeliefBird/SongFeatures/Motifs/1189/'
    tadbn_file='//home/timom/git/DeepBeliefBird/deep/trained_models/1024_25_300_0.05_FB_1189.pkl'
    #===========================================================================
    # data,targets=lb.createData(path,tadbn_file =tadbn_file ,method='tadbn',nfft=1024,hopF=2,batchsize=1,filterbank=True)
    # out = open('datatargets.pkl', 'wb')
    # cPickle.dump([data,targets],out)
    # out.close()
    #===========================================================================
    
    pkl = open('datatargets.pkl', 'rb')
    temp=cPickle.load(pkl)
    data=temp[0]
    targets=temp[1]
    visual_frames(data,targets)
    #visualize_frames(songpath)
    #visualize(songpath)