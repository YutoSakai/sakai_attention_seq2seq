import sys
sys.path.append('..')
import numpy
import MeCab
import os


m = MeCab.Tagger("-Owakati")
id_to_word = {}
word_to_id = {}


def _update_vocab(txt):
    words = m.parse(txt).split(" ")
    print(words)
    for i, word in enumerate(words):
        if word not in word_to_id:
            tmp_id = len(word_to_id)
            word_to_id[word] = tmp_id
            id_to_word[tmp_id] = word
    print(id_to_word)
    print(word_to_id)

def load_data(file_name='tweet/tweet2020-11-15嬉しいわ.txt', size=100, seed=1984):
    file_path = os.path.dirname(os.path.abspath(__file__)) + '/' + file_name

    if not os.path.exists(file_path):
        print('No file: %s' % file_name)
        return None

    x, t = [], []
    with open(file_path, 'r') as f:
        data = f.readlines()
        for i in range(size):
            if (i % 2) == 0:
                x.append(data[i][4:])
            else:
                t.append(data[i][4:])

    # create vocab dict
    for i in range(len(x)):
        _update_vocab(x[i])
        _update_vocab(t[i])

    # shuffle
    indices = numpy.arange(len(x))
    if seed is not None:
        numpy.random.seed(seed)
    numpy.random.shuffle(indices)
    x = x[indices]
    t = t[indices]

    # 10% for validation set
    split_at = len(x) - len(x) // 10
    (x_train, x_test) = x[:split_at], x[split_at:]
    (t_train, t_test) = t[:split_at], t[split_at:]
    return (x_train, t_train), (t_test, t_test)


def get_vocab():
    return word_to_id, id_to_word

# print(m.parse("わ～～嬉しいなぁありがとう。").split(" "))
_update_vocab("わ～～嬉しいなぁありがとう。")