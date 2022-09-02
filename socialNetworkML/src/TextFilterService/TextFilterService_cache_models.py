import time
from pathlib import Path

import joblib
import numpy as np
from textblob import TextBlob
from transformers import pipeline
from transformers.pipelines import check_task

vectorizer_path = Path(__file__).resolve().parent / 'data' / 'vectorizer.joblib'
model_path = Path(__file__).resolve().parent / 'data' / 'model.joblib'
vectorizer = joblib.load(vectorizer_path)
model = joblib.load(model_path)


def predict_prob(texts):
    def _get_profane_prob(prob):
        return prob[1]
    return np.apply_along_axis(_get_profane_prob, 1, model.predict_proba(vectorizer.transform(texts)))


text = '''
The titular threat of The Blob has always struck me as the ultimate movie
monster: an insatiably hungry, amoeba-like mass able to penetrate
virtually any safeguard, capable of--as a doomed doctor chillingly
describes it--"assimilating flesh on contact.
Snide comparisons to gelatin be damned, it's a concept with the most
devastating of potential consequences, not unlike the grey goo scenario
proposed by technological theorists fearful of
artificial intelligence run rampant.
'''

classifier = pipeline('sentiment-analysis')
targeted_task = check_task('sentiment-analysis')[0]['default']['model']
print('sentiment-analysis model: {}'.format(targeted_task))

t = time.time()
print('predict_prob: {}'.format(predict_prob([text])))
blob = TextBlob(text)
print('sentiment: {}'.format(blob.sentiment))
print('time: {}'.format(time.time() - t))


t = time.time()
print('classifier: {}'.format(classifier(text)))
print('time: {}'.format(time.time() - t))
