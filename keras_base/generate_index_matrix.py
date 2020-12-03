# coding: utf-8

def generate_mat(directory_name) :

    # どうせ1ファイルだろうけどlistディレクトリにあるファイルのパスの配列をglobによって取得
    file_list = glob.glob('/root/sakai_attention_seq2seq/keras_base/'+directory_name+'/list_corpus/*')

    print('listディレクトリ内のlist_corpusファイル数:'+str(len(file_list)))
    mat = []
    for i in range (0,len(file_list)) :
        print('これから読み込むlist_corpusファイルのパス:'+str(file_list[i]))

        with open(file_list[i],'rb') as f :

            # list_corpus.pickleを読み込む
            generated_list = pickle.load(f)

            # 単語配列を順に処理 次元を減らしている list_corpusには二次元の配列が入っているから
            for j in range(0, len(generated_list)):                
                mat.extend(generated_list[j])
            del generated_list

    # setは重複を排除したものとなるため、一度arrayのmatをsetのwordsに変換して重複を排除
    # その後すぐにlistに変換して文字でソートする
    words = sorted(list(set(mat)))

    # 語彙数分の全ゼロ配列を作成
    cnt = np.zeros(len(words))

    print('重複排除後の語彙数:'+str(len(words)))

    # 単語をキーにインデックス検索するためのword_indicesを作成
    word_indices = dict((w, i) for i, w in enumerate(words))

    # インデックスをキーに単語を検索するためのword_indicesを作成
    indices_word = dict((i, w) for i, w in enumerate(words))

    # 単語の出現数をカウント
    # cnt[単語のインデックス] = 出現数 となるように
    for j in range (0,len(mat)):
        cnt[word_indices[mat[j]]] += 1
    
    # 未知語一覧のための変数初期化
    words_unk = []

    # 出現頻度の少ない単語を「UNK」で置き換え
    for k in range(0,len(words)):

        # 出現回数10以下の単語は未知語として処理
        if cnt[k] <= 10 :

            # 未知語リストへ単語自体をappend
            words_unk.append(words[k])

            # 未知語とする単語をUNKに変換
            words[k] = 'UNK'

    print('未知語数:'+str(len(words_unk)))

    # 未知語処理を行ったので、set->list処理を行い、重複しているUNKを1つにする
    words = list(set(words))

    # ０パディング対策。インデックス０用キャラクタを追加
    words.append('\t')

    # 単語をソート
    words = sorted(words)

    print('未知語処理後の最終総単語数:'+str(len(words)))

    # 再度単語をキーにインデックス検索するためのword_indicesを作成
    word_indices = dict((w, i) for i, w in enumerate(words))

    # 再度インデックスをキーに単語を検索するためのword_indicesを作成
    indices_word = dict((i, w) for i, w in enumerate(words))

    # 発言した順の単語配列であるmatから、その単語順でindexの配列を作成する
    # 二次元の配列にする
    mat_urtext = np.zeros((len(mat),1),dtype=int)
    for i in range(0,len(mat)):
        if mat[i] in word_indices:
            mat_urtext[i,0] = word_indices[mat[i]]
        else:
            mat_urtext[i,0] = word_indices['UNK']

    print('発言単語配列の次元:', mat_urtext.shape)

    #作成した辞書をセーブ
    with open('/root/sakai_attention_seq2seq/keras_base/'+directory_name+'/index/word_indices.pickle', 'wb') as f :
        pickle.dump(word_indices , f)

    with open('/root/sakai_attention_seq2seq/keras_base/'+directory_name+'/index/indices_word.pickle', 'wb') as g :
        pickle.dump(indices_word , g)

    #単語ファイルセーブ
    with open('/root/sakai_attention_seq2seq/keras_base/'+directory_name+'/index/words.pickle', 'wb') as h :
        pickle.dump(words , h)

    #発言順の単語index配列をセーブ
    with open('/root/sakai_attention_seq2seq/keras_base/'+directory_name+'/index/mat_urtext.pickle', 'wb') as ff :
        pickle.dump(mat_urtext , ff)    

if __name__ == '__main__':
    import numpy as np
    import pickle
    import glob
    import sys

    # ディレクトリ名を引数から取得
    directory_name = sys.argv[1]

    # generate_word_list.pyで保存した二次元の単語を順に配列にしたものをgenerate_mat関数によって処理する
    # 読み込む単語配列は発言順になっている
    generate_mat(directory_name)
