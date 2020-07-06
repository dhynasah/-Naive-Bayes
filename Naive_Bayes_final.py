"""
Date: March 20 2019

@author: Dhynasah Cakir
"""

import pandas as pd 
import numpy as np 
from collections import defaultdict
from pprint import pprint
    


def addToDict(document,dict_index,word_dicts):
    
    '''
        Parameters:
        1. a single document  
        2. dict_index - implies to which category this document belongs to

        What the function does:
        -----------------------
        It splits the document on the basis of space as a tokenizer and adds every tokenized word to
        its corresponding dictionary

        Returns:
        ---------
        Nothing
    
   '''
    
    if isinstance(document,np.ndarray): document=document[0]
 
    for token_word in document: #for every word in document
      word_dicts[dict_index][token_word]+=1 #increment in its count
      
def train(dataset,labels,classes):
    
    '''
        Parameters:
        1. dataset
        2. labels
        3. unique classes

        What the function does:
        -----------------------
        This is the training function which will train the Naive Bayes Model i.e compute a dictionary for each
        category/class. 

        Returns:
        ---------
        category information i.e prior probability and denominator value for each class
    
    '''

    documents=dataset
    labels=labels
    word_dicts=np.array([defaultdict(lambda:0) for index in range(classes.shape[0])])
    
    if not isinstance(documents,np.ndarray): documents=np.array(documents)
    if not isinstance(labels,np.ndarray): labels=np.array(labels)
        
    #constructing dictionary for each category
    for cat_index,cat in enumerate(classes):
      
        all_cat_docs=documents[labels==cat] #filter all documents of category == cat
        
        
        all_cat_docs=pd.DataFrame(data=all_cat_docs)
        
        #now costruct dictionary of this particular category
        np.apply_along_axis(addToDict,1,all_cat_docs,cat_index,word_dicts)
    prob_classes=np.empty(classes.shape[0])
    all_words=[]
    cat_word_counts=np.empty(classes.shape[0])
    for cat_index,cat in enumerate(classes):
       
        #Calculating prior probability p(c) for each class
        prob_classes[cat_index]=np.sum(labels==cat)/float(labels.shape[0]) 
        

        cat_word_counts[cat_index]=np.sum(np.array(list(word_dicts[cat_index].values())))+1 
        
        #get all words of this category                                
        all_words+=word_dicts[cat_index].keys()
                                                 
    
    #combine all words of every category & make them unique to get vocabulary -V- of entire training set
    
    vocab=np.unique(np.array(all_words))
    vocab_length=vocab.shape[0]
                              
    #computing denominator value                                      
    denoms=np.array([cat_word_counts[cat_index]+vocab_length+1 for cat_index,cat in enumerate(classes)])                                                                          
    cats_info=[(word_dicts[cat_index],prob_classes[cat_index],denoms[cat_index]) for cat_index,cat in enumerate(classes)]                               
    cats_info=np.array(cats_info) 
    return cats_info     
def docProb(test_doc,classes,cats_info):                                

    '''
        Parameters:
        -----------
        1. a single test document 
        2. list of unique classes
        3. category information containing prior probability and denominatory for each category
        
        what the function does:
        -----------------------
        Function that estimates posterior probability of the given test document

        Returns:
        ---------
        probability of test document in all classes
    '''                                      
                                          
    likelihood_prob=np.zeros(classes.shape[0]) #to store probability w.r.t each class
    
    #finding probability w.r.t each class of the given test document
    for cat_index,cat in enumerate(classes): 
                         
        for test_token in test_doc.split(): #split the test document and get p of each test word
            
            ####################################################################################
                                          
            #This loop computes : for each word w [ count(w|c)+1 ] / [ count(c) + |V| + 1 ]                               
                                          
            ####################################################################################                              
            
            #get total count of this test token from it's respective training dict to get numerator value                           
            test_token_counts=cats_info[cat_index][0].get(test_token,0)+1
            
            #now get likelihood of this test_token word                              
            test_token_prob=test_token_counts/float(cats_info[cat_index][2])                              
            
            #To prevent underflow
            likelihood_prob[cat_index]+=np.log(test_token_prob)
                                          
    # we have likelihood estimate of the given document against every class but we need posterior probility
    post_prob=np.empty(classes.shape[0])
    for cat_index,cat in enumerate(classes):
        post_prob[cat_index]=likelihood_prob[cat_index]+np.log(cats_info[cat_index][1])                                  
  
    return post_prob

def test(test_set,classes,cats_info):
  
    '''
        Parameters:
        -----------
        1. A complete test set of shape (m,)
        2. list of unique classes
        3. category information: prior probability and denominator information 
        
        What the function does?
        -----------------------
        Determines probability of each test document against all classes and predicts the label
        against which the class probability is maximum

        Returns:
        ---------
        Predictions of test documents - A single prediction against every test document
    '''       
   
    predictions=[] #to store prediction of each test document
    for doc in test_set: 

        #get the posterior probability of every document                                  
        post_prob=docProb(doc,classes,cats_info) #get prob of this doucment for all classes
        
        #pick the max value and map against all classes
        predictions.append(classes[np.argmax(post_prob)])
            
    return np.array(predictions)



def main():
    train_data='forumTraining.txt' #getting all training documents 
    train_file = open(train_data)
    train_docs=[]
    df= pd.DataFrame(columns=['class','document_text'])
    for line in train_file:
        train_docs.append(line)
    train_file.close()
    for line in train_docs:
        line = line.rstrip('\n')
        words_list= line.split(" ")
        df= df.append({'class': words_list[0], 'document_text':words_list[1:]}, ignore_index=True)

    print ("Total Number of Training documents: ",len(train_data))
    print ("------------------- train set Categories -------------- ") 
    classes= pd.unique(df['class'])
    pprint(classes)
    
    cats_info= train(df['document_text'],df['class'],classes)
    print ("---------------- Training In Progress --------------------")
    
    print ('----------------- Training Completed ---------------------')
    test_file='forumTest.txt'
    test_docs=[]
    test_data= open(test_file)
    for line in test_data:
       test_docs.append(line)
    test_data.close()
    test_labels=[]
    for line in test_docs:
        line = line.rstrip('\n')
        words_list= line.split(" ")
        test_labels.append(words_list[0])
    print("------------------- test set Categories -------------- ")
    print(np.unique(test_labels))
    test_docs = np.array(test_docs)
    pclasses= test(test_docs,classes,cats_info)
    test_acc= np.sum(pclasses==test_labels)/float(test_docs.shape[0])
    print ("Test Set Documents: ",test_docs.shape[0])
    print ("Test Set Accuracy: ",test_acc*100,"%")
if __name__ == "__main__":
    main()