import json
import logging
import random
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from textblob import TextBlob
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from thrift.transport import TSocket, TTransport
from transformers import pipeline

gen_py_dir = Path(__file__).resolve().parent.parent.parent / 'gen-py'
sys.path.append(str(gen_py_dir))
from social_network import TextFilterService

logging.basicConfig(level=logging.DEBUG)
warnings.filterwarnings('ignore')

CLASSIFIER = pipeline('sentiment-analysis')


class TextFilterServiceHandler:
    def __init__(self):
        vectorizer_path = Path(__file__).resolve().parent / 'data' / 'vectorizer.joblib'
        model_path = Path(__file__).resolve().parent / 'data' / 'model.joblib'

        self.vectorizer = joblib.load(vectorizer_path)
        self.model = joblib.load(model_path)

    def predict_prob(self, texts):
        def _get_profane_prob(prob):
            return prob[1]
        return np.apply_along_axis(_get_profane_prob, 1, self.model.predict_proba(self.vectorizer.transform(texts)))

    def TextFilter(self, req_id, text, carrier):
        global CLASSIFIER

        start = time.time()
        probs = self.predict_prob([text])
        if random.random() < 0.10:
            try:
                sentiment = CLASSIFIER(text)
            except Exception as e:
                blob = TextBlob(text)
                sentiment = blob.sentiment
        else:
            blob = TextBlob(text)
            sentiment = blob.sentiment
        end = time.time()
        duration = end - start
        logging.info('processing time = {0:.1f}ms'.format(duration * 1000))
        return probs[0] > 0.5


if __name__ == '__main__':
    host_addr = 'localhost'
    host_port = 9090

    service_config_path = Path(__file__).resolve().parent.parent.parent / \
        'config' / 'service-config.json'

    with Path(service_config_path).open(mode='r') as f:
        config_json_data = json.load(f)
        host_addr = config_json_data['text-filter-service']['addr']
        host_port = int(config_json_data['text-filter-service']['port'])

    print(host_addr, ' ', host_port)
    handler = TextFilterServiceHandler()
    processor = TextFilterService.Processor(handler)
    transport = TSocket.TServerSocket(host=host_addr, port=host_port)
    tfactory = TTransport.TFramedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TThreadedServer(processor, transport, tfactory, pfactory)
    # A Thrift server that forks a new process for each request
    # This is more scalable than the threaded server as it does not cause
    # GIL contention.
    # Tensorflow is not compatible with TForkingServer
    # server = TServer.TForkingServer(processor, transport, tfactory, pfactory)

    logging.info('Starting the text-filter-service server...')
    server.serve()
