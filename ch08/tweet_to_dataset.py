import sys
sys.path.append('..')

tweet_path = 'tweet/tweet2020-11-15嬉しいわ.txt'


def load_data(tweet_path='tweet/tweet2020-11-15嬉しいわ.txt', size=100):
    x = []
    t = []
    train_x = []
    train_t = []
    test_x = []
    test_t = []
    with open(tweet_path, 'r') as f:
        data = f.readlines()
        for i in range(size):
            if (i % 2) == 0:
                x.append(data[i][4:])
            else:
                t.append(data[i][4:])
    train_x = x[:size-10]
    train_t = t[:size-10]
    test_x = x[size-10:]
    test_t = t[size-10:]
    return train_x, train_t, test_x, test_t
