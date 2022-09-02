import random
import string

from constants import user_id_by_follower_num


def random_string(length):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def random_decimal(length):
    return ''.join(random.choices(string.digits, k=length))


def compose_random_text():
    coin = random.random() * 100
    if coin <= 30.0:
        length = random.randint(0, 50)
    elif coin <= 58.2:
        length = random.randint(51, 100)
    elif coin <= 76.5:
        length = random.randint(101, 150)
    elif coin <= 85.3:
        length = random.randint(151, 200)
    elif coin <= 92.6:
        length = random.randint(201, 250)
    else:
        length = random.randint(251, 280)
    return random_string(length)


def compose_random_user():
    global user_id_by_follower_num

    user = 0
    coin = random.random() * 100
    if coin <= 0.4:
        user = random.choice(user_id_by_follower_num[10])
    elif coin <= 6.1:
        user = random.choice(user_id_by_follower_num[30])
    elif coin <= 16.6:
        user = random.choice(user_id_by_follower_num[50])
    elif coin <= 43.8:
        user = random.choice(user_id_by_follower_num[80])
    elif coin <= 66.8:
        user = random.choice(user_id_by_follower_num[100])
    else:
        user = random.choice(user_id_by_follower_num[300])
    return str(user)
