import os
import math

#These first two functions require os operations and so are completed for you
#Completed for you
def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d.startswith('.'):
            # ignore hidden files
            continue
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d+"/"
        files = os.listdir(directory+subdir)
        for f in files:
            bow = create_bow(vocab, directory+subdir+f)
            dataset.append({'label': label, 'bow': bow})
    return dataset

#Completed for you
def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """

    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        if d.startswith('.'):
            # ignore hidden files
            continue
        subdir = d if d[-1] == '/' else d+'/'
        files = os.listdir(directory+subdir)
        for f in files:
            with open(directory+subdir+f,'r', encoding = 'utf-8') as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])

#The rest of the functions need modifications ------------------------------
#Needs modifications
def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}
    bow[None] = 0
    with open(filepath, 'r', encoding = 'utf-8') as f:
        for word in f:
            word = word.strip()
            if word not in vocab:
                bow[None] += 1
            elif word in vocab:
                if word not in bow:
                    bow[word] = 1
                else:
                    bow[word] += 1
    if bow[None] == 0:
        bow.pop(None)            
    return bow

#Needs modifications
def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """

    smooth = 1 # smoothing factor
    logprob = {}
    n_2016 = 0
    n_2020 = 0
    totalFiles = 0
    
    for indices in training_data:
        for keys in indices:
            if keys == "label":
                indices[keys] = int(indices[keys])
                if indices[keys] ==  int('2016'):
                    n_2016 += 1
                elif indices[keys] == int('2020'):
                    n_2020 += 1
    
    totalFiles = n_2016 + n_2020
    prob_2016 = math.log((n_2016 + smooth)/ (totalFiles + 2))
    prob_2020 = math.log((n_2020 + smooth)/ (totalFiles + 2))
    
    logprob = {'2020': prob_2020, '2016': prob_2016}
    
    return logprob

#Needs modifications
def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1
    word_prob = {}
    
    lw = 0
    none_count = 0
    size_vocab = len(vocab)
    
    for index in training_data:
        if index["label"] == label:
            for key in index["bow"]:
                lw += index["bow"][key]
    
    for word in vocab:
        word_prob[word] = 0
        word_count = 0
        for index in training_data:
            if index["label"] == label:
                for key in index["bow"]:
                    if key == word:
                        word_count += index["bow"][key]
        word_prob[word] = math.log((word_count + smooth) / (lw + smooth * (size_vocab + 1)))
        
    for index in training_data:
        if index["label"] == label:
            for key in index["bow"]:
                if key not in word_prob:
                    none_count += index["bow"][key]
    word_prob[None] = math.log((none_count + smooth) / (lw + smooth * (size_vocab + 1)))
    
    return word_prob
##################################################################################
#Needs modifications
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = [f for f in os.listdir(training_directory) if not f.startswith('.')] # ignore hidden files
    
    
    vocab = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(vocab, training_directory)
    log_2016 = p_word_given_label(vocab, training_data, '2016')
    log_2020 = p_word_given_label(vocab, training_data, '2020')
    
    retval['vocabulary'] = vocab
    retval['log prior'] = prior(training_data, label_list)
    retval['log p(w|y=2016)'] = log_2016
    retval['log p(w|y=2020)'] = log_2020

    return retval

#Needs modifications
def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}
    # TODO: add your code here
    bow = create_bow(model["vocabulary"], filepath)
    retval["log p(y=2020|x)"] = model["log prior"]["2020"]
    retval["log p(y=2016|x)"] = model["log prior"]["2016"]
    retval["predicted y"] = 0
    print(retval)
    print(model["log p(y=2020|x)"][i])
    
    
    for word in bow:
        retval["log p(y=2020|x)"] += model["log p(y=2020|x)"][i] * bow[i]
        retval["log p(y=2016|x)"] += model["log p(y=2016|x)"][i] * bow[i]
        
    if retval["log p(y=2020|x)"] > retval["log p(y=2016|x)"]:
        retval["predicted y"] = "2020"
    
    else:
        retval["predicted y"] = "2016"

    return retval
