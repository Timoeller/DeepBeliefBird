import numpy as np
import pylab as pl
import os
import xlrd

def convertSyl2Num(syllables):
    converter={'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7} #0 reserved for no syllable
    s=np.zeros(syllables.shape[0])
    for i in range(syllables.shape[0]):
        s[i]=converter[syllables[i][0]] # taking just first letter in case something like A1 A2 exists...
    return s

def createLabelArray(xlsfile,fs=44100,windowL=1024,hopF=2):
    ''' method that takes the xls file and creates for each song a array with labels
    input:
          xlsfile: excel file containing column t1 = start of Syllable | t2 = end of Syl
          windowL,hopF,fs: raw input (with fs sample rate) is windowed and shifted by windowL/hopF, 
                          so labeled array has to look accordingly
    out:
        dict of key:songname and value: labeled array
    '''
    frame_shift=windowL*1.0/hopF
    ALL_labels={'windowL':windowL,'hopF':hopF}

    worksheet = xlrd.open_workbook(xlsfile).sheet_by_index(0)
    num_rows = worksheet.nrows - 1
    num_columns = worksheet.ncols - 1
    
    #get column number where info is placed
    columns = {}
    for i in range(2):
        for j in range(num_columns):
            if worksheet.cell_value(i, j) == 't1':
                columns['start'] = j
            elif worksheet.cell_value(i, j) == 't2':
                columns['end'] = j
            elif worksheet.cell_type(i, j) == 1:
                if len(worksheet.cell_value(i, j)) <= 2: #sometimes A1 etc exists
                    columns['syllable'] = j
                else:
                    columns['song'] = j
    
    skiprows=1 # skip first row because theres no data
    t1 = np.asarray(worksheet.col_values(columns['start'],start_rowx=skiprows))
    t2 = np.asarray(worksheet.col_values(columns['end'],start_rowx=skiprows))
    max_time = np.max(t2)
    t1 = np.floor(t1*fs/frame_shift).astype('int')
    t2 = np.ceil(t2*fs/frame_shift).astype('int')
    
    syllables = np.asarray(worksheet.col_values(columns['syllable'],start_rowx=skiprows))
    sylAsNums= convertSyl2Num(syllables)
    
    temp = np.asarray(worksheet.col_types(columns['song'],start_rowx=skiprows))
    idx = np.nonzero(temp)[0]
    idx= np.append(idx,t2.shape[0])
    for i in range(np.sum(temp)):
        labels = np.zeros((t2[idx[i+1]-1]))
        for j in range(idx[i],idx[i+1]):
            labels[t1[j]:t2[j]]=sylAsNums[j]
        ALL_labels[worksheet.cell_value(idx[i]+skiprows,columns['song'])]=labels
    
    return ALL_labels


if __name__ == '__main__':
    path= 'D:\\computational neuroscience\\HIWI TADBN\\labeled Motifs\\3718\\'
    
    os.chdir(path)
    songnames=[]
    for files in os.listdir("."):
        if files.endswith(".wav"):
            songnames.append(files)
        elif files.endswith(".xls"):
            xlsfile=files
    print createLabelArray(path+xlsfile)        
            
            
    