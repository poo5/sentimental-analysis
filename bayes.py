import math, os, pickle, re
from math import exp
from collections import defaultdict
from collections import Counter



class Bayes_Classifier:

    dict = {
        'pos' : {},
        'neg' : {},
        'count': {
            'pos': 0,
            'neg': 0
        }
    }

    def __init__(self, trainDirectory="movie_reviews/"):
        '''This method initializes and trains the Naive Bayes Sentiment Classifier.  If a
        cache of a trained classifier has been stored, it loads this cache.  Otherwise,
        the system will proceed through training.  After running this method, the classifier
        is ready to classify input text.'''
        self.trainDir = trainDirectory
        try:
            self.dict = self.load("dict.pickle")
        except:
            self.train()


    def train(self):
        '''Trains the Naive Bayes Sentiment Classifier.'''
        iFileList = []
        poscount = 0
        negcount = 0

        for fFileObj in os.walk(self.trainDir + "/"):
            iFileList = fFileObj[2]
            break

        print '%d test reviews.' % len(iFileList)

        #print iFileList
        for filename in iFileList:
            m = re.split("[-.]",filename)[-3]
            if m == '1':

                #print ("Negative")
                store_words = self.loadFile(self.trainDir + filename)
                tokenized_value = self.tokenize(store_words)
                freqs = Counter(tokenized_value)
                for key, value in freqs.items():
                    if key in self.dict["neg"]:
                        self.dict["neg"][key] += value
                    else:
                        self.dict["neg"][key] = value

                negcount += 1


            else:

                #print ("positive")
                word = self.loadFile(self.trainDir + filename)
                tokenized_value = self.tokenize(word)
                freqs = Counter(tokenized_value)
                for key, value in freqs.items():
                    if key in self.dict["pos"]:
                        self.dict["pos"][key] += value
                    else:
                        self.dict["pos"][key] = value
                poscount +=1


        self.dict["count"]["pos"] = poscount
        self.dict["count"]["neg"] = negcount
        self.save(self.dict,"dict.pickle")


    def classify(self, sText):
        '''Given a target string sText, this function returns the most likely document
        class to which the target string belongs. This function should return one of three
        strings: "positive", "negative" or "neutral".
        '''

# 35784
        vocab_count = len(set().union(self.dict["pos"].keys(), self.dict["neg"].keys()))
        total = self.dict["count"]["pos"] + self.dict["count"]["neg"]
        #p=self.dict["count"]["pos"]
        prior_pos = self.dict["count"]["pos"]/float(total)
        prior_neg = self.dict["count"]["neg"]/float(total)
        words = self.tokenize(sText)
        sum_pos = (sum(self.dict["pos"].values()))
        neg_pos = (sum(self.dict["neg"].values()))
        total1 = 0
        total2 = 0
        for word in words:
            # positive
            if word in self.dict["pos"]:

                freqs = self.dict["pos"][word]
                cond_word = (freqs + 1)/(float(sum_pos) + vocab_count)
                total1 += math.log(cond_word)

            else:
                cond_word = 1/(float(sum_pos) + vocab_count)
                total1 += math.log(cond_word)



            #negative
            if word in self.dict["neg"]:

                freqs = self.dict["neg"][word]
                cond_word = (freqs+1) / (float(neg_pos) + vocab_count)
                total2 += math.log(cond_word)
            else:
                cond_word = 1 / (float(neg_pos) + vocab_count)
                total2 += math.log(cond_word)


        prob_neg = total2 + math.log(prior_neg)
        prob_pos = total1 + math.log(prior_pos)


        #positive
        if prob_pos > prob_neg:
            return "positive"
        #neutral
        elif abs(prob_pos - prior_neg) < 0.05:
            return "Neutral"

        else:
            return "negative"


    def loadFile(self, sFilename):
        '''Given a file name, return the contents of the file as a string.'''

        f = open(sFilename, "r")
        sTxt = f.read()
        f.close()
        return sTxt

    def save(self, dObj, sFilename):
        '''Given an object and a file name, write the object to the file using pickle.'''

        f = open(sFilename, "w")
        p = pickle.Pickler(f)
        p.dump(dObj)
        f.close()

    def load(self, sFilename):
        '''Given a file name, load and return the object stored in the file.'''

        f = open(sFilename, "r")
        u = pickle.Unpickler(f)
        dObj = u.load()
        f.close()
        return dObj

    def tokenize(self, sText):
        '''Given a string of text sText, returns a list of the individual tokens that
        occur in that string (in order).'''

        lTokens = []
        sToken = ""
        for c in sText:
            if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\'" or c == "_" or c == '-':
                sToken += c
            else:
                if sToken != "":
                    lTokens.append(sToken)
                    sToken = ""
                if c.strip() != "":
                    lTokens.append(str(c.strip()))

        if sToken != "":
            lTokens.append(sToken)

        return lTokens
