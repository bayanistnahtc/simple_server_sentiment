import zerorpc

from test_python import test_me

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pickle

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


PORT = 4242
vectorizer_file = open('sentiment_vectorizer.pickle','rb')
classifier_file = open('sentiment_classifier.pickle', 'rb')
classifier = pickle.load(classifier_file)
vectorizer = pickle.load(vectorizer_file)

class PythonServer(object):


    def listen(self):
        print(f'Python Server started listening on {PORT} ...')

    def test(self, param):
        return test_me(param)
        
    def predict(self, message):
        '''messages the list of the messages to be classified
        returns label of the single message :
        1.0 - stands for positive label
        2.0 - stands for negative label
        3.0 - stands for neutral label
        '''
        message = [message]
        vectorized_message = vectorizer.transform(message)
        predictions = classifier.predict(vectorized_message)
        return str(int(predictions[0]))

    def predict_collection(self, messages):
        '''messages the list of the messages to be classified
        returns the list of the predicted labels :
        1.0 - stands for positive label
        2.0 - stands for negative label
        3.0 - stands for neutral label
        '''
        vectorized_messages = vectorizer.transform(messages)
        predictions = classifier.predict(vectorized_messages)
        return predictions

try:
    s = zerorpc.Server(PythonServer())
    s.bind(f'tcp://0.0.0.0:{PORT}')
    s.run()
    print('PythonServer running...')




except Exception as e:
    print('unable to start PythonServer:', e)
    raise e