import base64
import json
import logging
import os
import random
import string
import sys
import time
from pathlib import Path

import locust.stats
import numpy as np
import urllib3
from locust import HttpUser, LoadTestShape, TaskSet, constant, tag, task
from locust.contrib.fasthttp import FastHttpUser

sys.path.append('/mnt/locust/lib')
from helpers import (compose_random_text, compose_random_user, random_decimal,
                     random_string)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
locust.stats.CSV_STATS_INTERVAL_SEC = 1
rng = np.random.default_rng()

# load images
image_dir = Path('/mnt/locust/data')
image_data_dict = {}
image_names = []
for img in image_dir.glob('*'):
    image_name = img.name
    image_names.append(image_name)
    with img.open(mode='r') as f:
        image_data_dict[image_name] = f.read()


class SocialNetworkUser(FastHttpUser):
    def wait_time(self):
        global rng
        return rng.exponential(scale=0.1)

    @task(5)
    @tag('compose_post')
    def compose_post(self):
        #----------------- contents -------------------#
        user_id = compose_random_user()
        username = 'username_' + user_id
        text = compose_random_text()

        # #---- user mentions ----#
        user_mention_ids = list()
        for _ in range(0, 5):
            while True:
                user_mention_id = random.randint(1, 962)
                if (user_mention_id != user_id and
                        user_mention_id not in user_mention_ids):
                    user_mention_ids.append(user_mention_id)
                    break
        for user_mention_id in user_mention_ids:
            text = text + ' @username_' + str(user_mention_id)

        #---- urls ----#
        for _ in range(0, 5):
            if random.random() <= 0.2:
                num_urls = random.randint(1, 5)
                for _ in range(0, num_urls):
                    text = text + ' https://www.bilibili.com/av' + random_decimal(8)

        #---- media ----#
        num_media = 0
        media_ids = list()
        media_types = list()
        media_data_list = list()
        # if random.random() < 0.21:
        if random.random() < 0.25:
            num_media = random.randint(1, 3)

        for _ in range(0, num_media):
            img_name = random.choice(image_names)
            if 'jpg' in img_name:
                media_types.append('jpg')
            elif 'png' in img_name:
                media_types.append('png')
            else:
                continue
            media_ids.append(random_decimal(18))
            media_data_list.append(image_data_dict[img_name])

        url = '/wrk2-api/post/compose'
        body = {}
        if num_media > 0:
            body['username'] = username
            body['user_id'] = user_id
            body['text'] = text
            body['media_ids'] = json.dumps(media_ids)
            body['media_types'] = json.dumps(media_types)
            body['media_data_list'] = json.dumps(media_data_list)
            body['post_type'] = '0'
        else:
            body['username'] = username
            body['user_id'] = user_id
            body['text'] = text
            body['media_ids'] = ''
            body['media_types'] = ''
            body['media_data_list'] = ''
            body['post_type'] = '0'

        r = self.client.post(url, data=body, name='compose_post', timeout=10)

        if r.status_code > 202:
            logging.info('compose-post: {}'.format(body))
            logging.warning('compose_post resp.status={}, text={}'
                            .format(r.status_code, r.text))

    @task(65)
    @tag('read_home_timeline')
    def read_home_timeline(self):
        start = random.randint(0, 100)
        stop = start + random.randint(5, 10)

        url = '/wrk2-api/home-timeline/read'
        params = {}
        params['user_id'] = str(random.randint(1, 962))
        params['start'] = str(start)
        params['stop'] = str(stop)

        # FastHttpUser's client does not support params
        url = url + '?user_id=' + params['user_id'] + \
            '&start=' + params['start'] + '&stop=' + params['stop']
        r = self.client.get(url, params=params, name='read_home_timeline', timeout=10)

        if r.status_code > 202:
            logging.warning('read_home_timeline resp.status={}, text={}'
                            .format(r.status_code, r.text))

    @task(30)
    @tag('read_user_timeline')
    def read_user_timeline(self):
        start = random.randint(0, 100)
        stop = start + random.randint(5, 10)

        url = '/wrk2-api/user-timeline/read'
        params = {}
        params['user_id'] = str(random.randint(1, 962))
        params['start'] = str(start)
        params['stop'] = str(stop)

        # FastHttpUser's client does not support params
        url = url + '?user_id=' + params['user_id'] + \
            '&start=' + params['start'] + '&stop=' + params['stop']
        r = self.client.get(url, params=params, name='read_user_timeline', timeout=10)

        if r.status_code > 202:
            logging.warning('read_user_timeline resp.status={}, text={}'
                            .format(r.status_code, r.text))
