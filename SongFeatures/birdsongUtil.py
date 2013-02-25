'''
Module to handle Birdsong wav files: 
- produce labels according to excel file
- create train and test sets for cross validation 

author: Timo Moeller
'''

import numpy as np
import pylab as pl
import os
import xlrd
from scipy.io import wavfile
import cPickle
import preprocessing as pp



def convertSyl2Num(syllables):
    converter={'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7} #0 reserved for no syllable
    s=np.zeros(syllables.shape[0])
    for i in range(syllables.shape[0]):
        s[i]=converter[syllables[i][0]] # taking just first letter in case something like A1 A2 exists...
    return s

def createLabelArray(path,PEGGY_DRIFT=0.01,fs=44100,windowL=1024,hopF=2):
    ''' method that takes the xls file and creates for each song a array with labels
    in:
      path: path containing excel file: column t1 = start of Syllable | t2 = end of Syl
      PEGGY_DRIFT: apperently labeling of data should be shifted to later times
      windowL,hopF,fs: raw input (with fs sample rate) is windowed and shifted by windowL/hopF, 
                      so labeled array has to look accordingly
    out:
        dict of key:songname and value: labeled array
    '''
    os.chdir(path)
    for files in os.listdir("."):
        if files.endswith(".xls"):
            xlsfile=files
            break
    frame_shift=windowL*1.0/hopF
    ALL_labels={'windowL':windowL,'hopF':hopF}

    worksheet = xlrd.open_workbook(xlsfile).sheet_by_index(0)
    num_rows = worksheet.nrows
    num_columns = worksheet.ncols
    
    #get column number where info is placed
    columns = {}
    for i in range(2):
        for j in range(num_columns):
            if worksheet.cell_value(i, j) == 't1':
                columns['start'] = j
            elif worksheet.cell_value(i, j) == 't2':
                columns['end'] = j
            elif worksheet.cell_value(i, j) == 't2-t1':
                columns['duration'] = j
            elif worksheet.cell_type(i, j) == 1:
                if len(worksheet.cell_value(i, j)) <= 2: #sometimes A1 etc exists
                    columns['syllable'] = j
                else:
                    columns['song'] = j
                    
    skiprows=1 # skip first row because there's no data
    
    ALL_labels['minDuration']= np.floor(np.min(np.asarray(worksheet.col_values(columns['duration'],start_rowx=skiprows)))*(fs/frame_shift))
    ALL_labels['maxDuration']= np.floor(np.max(np.asarray(worksheet.col_values(columns['duration'],start_rowx=skiprows)))*(fs/frame_shift))
    t1 = np.asarray(worksheet.col_values(columns['start'],start_rowx=skiprows)) + PEGGY_DRIFT
    t2 = np.asarray(worksheet.col_values(columns['end'],start_rowx=skiprows)) + PEGGY_DRIFT
    
    t1 = np.floor(t1*fs/frame_shift).astype('int') - 1
    t2 = np.ceil(t2*fs/frame_shift).astype('int') - 1 # both shifted for index starts at 0
    
    syllables = np.asarray(worksheet.col_values(columns['syllable'],start_rowx=skiprows))
    sylAsNums= convertSyl2Num(syllables)
    
    temp = np.asarray(worksheet.col_types(columns['song'],start_rowx=skiprows))
    idx = np.nonzero(temp)[0]
    idx= np.append(idx,t2.shape[0])#appending end reference
    for i in range(np.sum(temp)):
        labels = np.zeros((t2[idx[i+1]-1]))
        for j in range(idx[i],idx[i+1]): #looping through labels of 1 file
            labels[t1[j]:t2[j]]=sylAsNums[j]
        ALL_labels[worksheet.cell_value(idx[i]+skiprows,columns['song'])]=labels
    
    return ALL_labels

def readSongs(mypath):
    '''
    reads all .wav files in mypath with same samplerate
    file should look like '7 276.wav' || sorting done on first number, second number ident bird
    in:
        path to look for .wav files. no traversing of dir
    out: 
        songs: list of songs, 
        fs: sample rate of all songs 
        onlyfiles: strings of files, same index as songs
    '''
    onlyfiles = [ f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f)) and f.endswith('.wav')]
    recordnum = np.asarray([int(x[:x.find(' ')]) for x in onlyfiles])
    ind= np.argsort(recordnum)
    onlyfiles = [onlyfiles[i] for i in ind] #files sorted after first number in name
    songs=[]
    filenames=[]
    fsAssigned=False
    for i,myfile in enumerate(onlyfiles):
        this=True
        try:
            [fs, song]=wavfile.read(mypath+myfile)
        except Exception, e:
            print e + myfile
            #print myfile + ' is not working properly'
            this=False
        if not(fsAssigned) and this: #tricky if first file is corrupted
            FSall = fs
        elif this and fs != FSall: #if sample rate mismatched from other files, don't add to list and remove from onlyfiles
            print 'mismatched sampling rates in folder %s' %mypath
            this=False
        if this:
            fsAssigned=True
            filenames.append(myfile)
            songs.append(song)
    #seqlen = [len(x) for x in songs]
            
    return songs,FSall,filenames    

if __name__ == '__main__':
    path= '//home/timom/git/DeepBeliefBird/SongFeatures/Motifs/1189/'
    plotting=True
    os.chdir(path)
    for files in os.listdir("."):
        if files.endswith(".xls"):
            xlsfile=files 
            
    nfft=512
    songs,fs,filenames=readSongs(path)        
    labels= createLabelArray(path,PEGGY_DRIFT=0.01,fs=fs,windowL=nfft,hopF=2)  
    print filenames      
    print len(songs)
    songnumber = 4
    print filenames[songnumber][:-4]
    newspec,invD,mu,sigma= pp.main(songs[songnumber],fs,hpFreq=250,nfft=nfft,hopfactor=2,M=False,numCoeffs= 30,plotting=False)
    print newspec.shape
    
    if plotting:
        oldspec = np.dot(invD,((newspec*sigma)+mu).T)
        pl.figure()
        pl.imshow(oldspec,origin='lower',aspect='auto')
        label=labels[filenames[songnumber][:-4]]
        pl.plot(label*oldspec.shape[0]/20.0,'xb')
        pl.xlim(0,oldspec.shape[1])
        
        pl.show()

            
            
    