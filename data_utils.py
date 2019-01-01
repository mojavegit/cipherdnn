import numpy as np
import tensorflow as tf
import csv
import re

class Data(object):
    
    def __init__(self,
                 data_source,
                 alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
                 l0 = 1014,
                 batch_size = 128,
                 no_of_classes = 20):
        
        self.alphabet = alphabet
        self.alphabet_size = len(self.alphabet)
        self.dict = {}
        self.no_of_classes = no_of_classes
        
        with open(data_source, "r") as bf:
                data1 = bf.read()

        for i, c in enumerate(data1):
            self.dict[hex(ord(c))] = i+1


        index = 1
        for vocab_key in self.dict.keys():
            self.dict[vocab_key]=index
            index = index + 1

        with open('test_map.csv', 'w') as f:
            for key in self.dict.keys():
                f.write("%s,%s\n"%(key,self.dict[key]))
                
        print(len(self.dict))
        self.alphabet_size = len(self.dict)
        
        self.length = l0
        self.batch_size = batch_size
        self.data_source = data_source


    def loadData1(self):
        data = []
        with open(self.data_source, 'r') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            for row in rdr:
                txt = ""
                for s in row[1:]:
                    txt = txt + " " + re.sub("^\s*(.-)\s*$", "%1", s).replace("\\n", "\n")
                
                cipher_text = txt.encode("utf-8")
                data.append ((int(row[0]), cipher_text))

        self.data = np.asarray(data)
        self.shuffled_data = self.data
        
    def loadData(self):
        data = []
        with open(self.data_source, 'r') as f:
            
             rdr = csv.reader(f, delimiter=',', quotechar='"')
            
             for row in rdr:
                txt = row[2]
                cipher_text = txt.encode("utf-8")
                txt2 = row[3]
                data.append ((int(txt2), cipher_text))
                
        print(len(data))
        self.data = np.asarray(data)
        #self.data = data
        self.shuffled_data = self.data  
        
    def loadDatatest(self):
        data = []
        with open(self.data_source, 'r') as f:
            
             rdr = csv.reader(f, delimiter=',', quotechar='"')
            
             for row in rdr:
                txt = row[2]
                cipher_text = txt.encode("utf-8")
                #data.append ((int(0), cipher_text))
                data.append ((cipher_text))
                

        self.data = np.asarray(data)
        #self.data = data
        self.shuffled_data = self.data  
         

    def shuffleData(self):
        np.random.seed(235)
        data_size = len(self.data)
        
        shuffle_indices = np.random.permutation(np.arange(data_size))
        self.shuffled_data = self.data[shuffle_indices]         
        



    def getBatch(self, batch_num=0):
        data_size = len(self.data)
        start_index = batch_num * self.batch_size
        end_index = min((batch_num + 1) * self.batch_size, data_size)
        return self.shuffled_data[start_index:end_index]

        

    def getBatchToIndices(self, batch_num = 0):
        data_size = len(self.data)
        start_index = batch_num * self.batch_size
        end_index = data_size if self.batch_size == 0 else min((batch_num + 1) * self.batch_size, data_size)
        batch_texts = self.shuffled_data[start_index:end_index]
        batch_indices = []
        one_hot = np.eye(self.no_of_classes, dtype='int64')
        classes = []
        for c, s in batch_texts:
            #print(s)
            #print("******")
            #print(c)
            batch_indices.append(self.strToIndexs(s))
            c = int(c) - 1
            classes.append(one_hot[c])

        return np.asarray(batch_indices, dtype='int64'), classes
            
        
        

    def strToIndexs(self, s):
   
        #s = s.lower()
        m = len(s)
        n = min(m, self.length)
        str2idx = np.zeros(self.length, dtype='int64') 
        k = 0
        #print(s)
        for i in range(0, m):
            c = s[i]
            if hex(c) in self.dict:
                str2idx[i] = self.dict[hex(c)]
        return str2idx


    def getLength(self):
        return len(self.data)





if __name__ == '__main__':
    data = Data("/Users/dal/kaggle_comp/train_noheader.csv")
    #E = np.eye(4)
    #img = np.zeros((4, 15))
    #idxs = data.strToIndexs('aghgbccdahbaml')
    #print(idxs)
    
    data.loadData()
    
    with open("train_trans.vec", "w") as fo:
        for i in range(data.getLength()):
            c = data.data[i][0]
            txt = data.data[i][1]
            vec =  " ".join(map(str, data.strToIndexs(txt)))
            
            fo.write("{},{}\n".format(c, vec))
        
    
    
    for i in range(3):
        data.shuffleData()
        batch_x, batch_y = data.getBatchToIndices()
        print (batch_x[1], batch_y[1])
        

            
        
        
        
