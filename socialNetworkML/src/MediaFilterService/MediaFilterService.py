import base64
import io
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from tensorflow import keras
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from thrift.transport import TSocket, TTransport

from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.framework import dtypes
from tensorflow.core.protobuf import rewriter_config_pb2


gen_py_dir = Path(__file__).resolve().parent.parent.parent / 'gen-py'
sys.path.append(str(gen_py_dir))
from social_network import MediaFilterService

tf.get_logger().setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')

#tf.compat.v1.disable_eager_execution()

#SESSION = tf.compat.v1.Session()
#tf.compat.v1.keras.backend.set_session(SESSION)
IMAGE_DIM = 224
MODEL_PATH = Path(__file__).resolve().parent / 'data' / 'output_graph_inceptionv3_224_224_3.pb'
INPUTS = 'Placeholder'
OUTPUTS = 'final_result'
infer_graph = tf.Graph()

infer_config = tf.compat.v1.ConfigProto()
#uncomment to turn bfloat16 on
#infer_config.graph_options.rewrite_options.auto_mixed_precision_mkl = rewriter_config_pb2.RewriterConfig.ON

with infer_graph.as_default():
    graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.FastGFile(MODEL_PATH, 'rb') as input_file:
        input_graph_content = input_file.read()
        graph_def.ParseFromString(input_graph_content)
    output_graph = optimize_for_inference(graph_def, [INPUTS], [OUTPUTS], dtypes.float32.as_datatype_enum, False)
    tf.import_graph_def(output_graph, name='')
input_tensor = infer_graph.get_tensor_by_name('Placeholder:0')
output_tensor = infer_graph.get_tensor_by_name('final_result:0')
infer_sess = tf.compat.v1.Session(graph=infer_graph, config=infer_config)


class MediaFilterServiceHandler:
    def __init__(self):
        pass

    def _load_base64_image(self, base64_str):
        global IMAGE_DIM

        img_str = base64.b64decode(base64_str)
        temp_buff = io.BytesIO()
        temp_buff.write(img_str)
        temp_buff.flush()
        image = Image.open(temp_buff).convert('RGB')
        image = image.resize(size=(IMAGE_DIM, IMAGE_DIM),
                             resample=Image.NEAREST)
        temp_buff.close()
        image = keras.preprocessing.image.img_to_array(image)
        image /= 255
        return image

    def _classify_nd(self, nd_images):
        global infer_sess
        global output_tensor
        predictions = infer_sess.run(output_tensor, feed_dict={input_tensor: nd_images})
        return predictions
        

    def _classify_base64(self, base64_images):
        # logging.info('loading images ...')
        images = np.ndarray(shape=(len(base64_images), 224, 224, 3))
        for i,img in enumerate(base64_images):
            images[i] = (self._load_base64_image(base64_str=img))
        #images = np.asarray(images)
        # logging.info('finish loading images ...')

        filter_list = list()
        category_list = list()
        try:
            if len(images)>0:
                #start = time.time()
                probs = self._classify_nd(images)
                #end = time.time()
                #duration = end - start
                #logging.info('classify time = {0:.1f}ms'.format(duration * 1000))
                for prob in probs:
                    maxprob = np.argmax(prob, axis=0)
                    filter_list.append(maxprob in [0, 2])
        except Exception as e:
            logging.error('prediction failed: {}'.format(e))
            for _ in range(0, len(base64_images)):
                filter_list.append(False)
        return filter_list

    def MediaFilter(self, req_id, media_ids, media_types, media_data_list, carrier):
        #logging.info('Number of Images in Requests = {}'.format(len(media_data_list))
        #start = time.time()
        filter_list = self._classify_base64(base64_images=media_data_list)
        #end = time.time()
        #duration = end - start
        #logging.info('inference time = {0:.1f}ms'.format(duration * 1000))
        return filter_list


if __name__ == '__main__':
    host_addr = 'localhost'
    host_port = 9090

    service_config_path = Path(__file__).resolve().parent.parent.parent / 'config' / 'service-config.json'

    with Path(service_config_path).open(mode='r') as f:
        config_json_data = json.load(f)
        host_addr = config_json_data['media-filter-service']['addr']
        host_port = int(config_json_data['media-filter-service']['port'])

    print(host_addr, ' ', host_port)
    handler = MediaFilterServiceHandler()
    processor = MediaFilterService.Processor(handler)
    transport = TSocket.TServerSocket(host=host_addr, port=host_port)
    tfactory = TTransport.TFramedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TThreadedServer(processor, transport, tfactory, pfactory)

    # Tensorflow is not compatible with TForkingServer
    # server = TServer.TForkingServer(processor, transport, tfactory, pfactory)

    logging.info('Starting the media-filter-service server...')
    server.serve()
