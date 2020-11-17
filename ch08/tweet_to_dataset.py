import sys
sys.path.append('..')
import numpy
import MeCab
import os


m = MeCab.Tagger("-Owakati")
id_to_word = {}
word_to_id = {}

word_to_id['<pad>'] = 0
id_to_word[0] = '<pad>'


def _update_vocab(txt):
    words = m.parse(txt).split(" ")
    for i, word in enumerate(words):
        if word not in word_to_id:
            tmp_id = len(word_to_id)
            word_to_id[word] = tmp_id
            id_to_word[tmp_id] = word


def load_data(file_name='../../tweet/tweet/tweet2020-11-15嬉しいわ.txt', seed=1984):
    if len(sys.argv) >= 2:
        file_name = sys.argv[1]
    file_path = os.path.dirname(os.path.abspath(__file__)) + '/' + file_name

    if not os.path.exists(file_path):
        print('No file: %s' % file_name)
        return None

    req, res = [], []
    with open(file_path, 'r') as f:
        data = f.readlines()
        for i in range(len(data)):
            if (i % 2) == 0:
                req.append(data[i][4:])
            else:
                res.append(data[i][4:])

    # create vocab dict
    for i in range(len(req)):
        _update_vocab(req[i])
        _update_vocab(res[i])

    # create numpy array
    max_sentence_len = 0
    for i in range(len(req)):
        if max_sentence_len < len(req[i]):
            max_sentence_len = len(req[i])
        if max_sentence_len < len(res[i]):
            max_sentence_len = len(res[i])
    x = numpy.zeros((len(req), max_sentence_len), dtype=numpy.int)
    t = numpy.zeros((len(req), max_sentence_len), dtype=numpy.int)

    for i, sentence in enumerate(req):
        for j, word in enumerate(m.parse(sentence).split(" ")):
            x[i][j] = [word_to_id[word]][0]
    for i, sentence in enumerate(res):
        for j, word in enumerate(m.parse(sentence).split(" ")):
            t[i][j] = [word_to_id[word]][0]

    # shuffle
    if seed is not None:
        numpy.random.seed(seed)
    for l in [x, t]:
        numpy.random.seed(seed)
        numpy.random.shuffle(l)

    # 10% for validation set
    split_at = len(x) - len(x) // 10
    (x_train, x_test) = x[:split_at], x[split_at:]
    (t_train, t_test) = t[:split_at], t[split_at:]
    print("train size: " + str(len(x_train)))
    return (x_train, t_train), (t_test, t_test)


def get_vocab():
    return word_to_id, id_to_word

# print(m.parse("わ～～嬉しいなぁありがとう。").split(" "))
# _update_vocab("わ～～嬉しいなぁありがとう。")
# load_data()