# coding: utf-8


def generate_tensors(maxlen_e,maxlen_d,directory_name) :
 
    # 単語ファイルロード
    with open('/root/sakai_attention_seq2seq/keras_base/'+directory_name+'/index/words.pickle', 'rb') as f :
        words=pickle.load(f)    

    # 作成した辞書をロード
    with open('/root/sakai_attention_seq2seq/keras_base/'+directory_name+'/index/word_indices.pickle', 'rb') as f :
        word_indices=pickle.load(f)

    with open('/root/sakai_attention_seq2seq/keras_base/'+directory_name+'/index/indices_word.pickle', 'rb') as g :
        indices_word = pickle.load(g)

    # 単語順の単語index配列をロード
    with open('/root/sakai_attention_seq2seq/keras_base/'+directory_name+'/index/mat_urtext.pickle', 'rb') as ff :
        mat_urtext = pickle.load(ff)  

    # REQREQとRESRESのインデックスを持っておく
    req = word_indices['REQREQ']
    res = word_indices['RESRES']

    # mat_urtextという二次元配列(2次元にしているのはNNへの入力のため)のなかでREQREQとRESRESのインデックスがある位置を記録するための配列を初期化
    delimiters = []
    req_delimiter = None
    res_delimiter = None

    #コーパス上のデリミタ(RESRESとREQREQ)の位置をdelimitersにappendしていく
    for i in range(0,mat_urtext.shape[0]):
        if mat_urtext[i,0] == req:
            req_delimiter = i
        elif mat_urtext[i,0] == res:
            if req_delimiter != None:
                res_delimiter = i
                delimiters.append(req_delimiter)
                delimiters.append(res_delimiter)
                req_delimiter = None
                res_delimiter = None
        if i % 1000000 == 0:
            print(str(i)+'単語処理完了')
    print('デリミタ数:'+str(len(delimiters)))

    # デリ見たは２種類あるので、対話数はその半分 n:対話数
    n = len(delimiters) // 2

    #入力、ラベルテンソルの初期値定義（０値マトリックス）
    enc_input = np.zeros((n,maxlen_e))
    dec_input = np.zeros((n,maxlen_d))
    target = np.zeros((n,maxlen_d))

    # デリミタを目印に、コーパスから文章データを切り出して入力／ラベルマトリックスにコピー
    j = 0

    # エラーのカウント
    err_cnt = 0

    for i in range(0,n) :

        # 「REQREQ」のdelimitersインデックス
        index1=2*i+err_cnt
        # 「RESRES」のインデックス
        index2=2*i+1+err_cnt
        # 次の「REQREQ」のインデックス
        index3=2*i+2+err_cnt

        if index3 >= n :
            break
        
        # 発話／応答の組が崩れていた時はスキップする
        if mat_urtext[delimiters[index1],0] != req or mat_urtext[delimiters[index2],0] != res :
            print(str(i)+'番目対話はシーケンスエラー ')
            err_cnt += 1
            continue

        #入力文の単語数
        len_e = delimiters[index2] - delimiters[index1] - 1
        #出力文の単語数
        len_d = delimiters[index3] - delimiters[index2]

        # 系列長より短い会話のみ、入力／ラベルマトリックスに書き込む
        if len_e <= maxlen_e and len_d <= maxlen_d:
            enc_input[j,0:len_e] = mat_urtext[delimiters[index1]+1:delimiters[index2],0].T
            dec_input[j,0:len_d] = mat_urtext[delimiters[index2]:delimiters[index3],0].T
            target[j,0:len_d] = mat_urtext[delimiters[index2]+1:delimiters[index3]+1,0].T
            j += 1
        if i % 100000 == 0 :
            print(str(i)+'番目対話まで処理完了')

    #会話文データを書き込んだ分だけ切り出す
    e = enc_input[0:j,:].reshape(j,maxlen_e,1)
    d = dec_input[0:j,:].reshape(j,maxlen_d,1)
    t_ = target[0:j,:].reshape(j,maxlen_d,1)

    #ラベルテンソル定義
    t = np.zeros((j,maxlen_d,2),dtype='int32')

    #各単語の出現頻度カウント
    cnt = np.zeros(len(words),dtype='int32')
    for i in range (0,mat_urtext.shape[0]) :
        cnt[mat_urtext[i,0]] += 1

    #出現頻度の降順配列
    freq_indices = np.argsort(cnt)[::-1]          #出現頻度からインデックス検索
    indices_freq = np.argsort(freq_indices)       #インデックスから出現頻度検索

    # 次元数決定(語彙数の8分の1)
    dim = math.ceil(len(words) / 8)

    #ラベルテンソル作成
    for i in range(0,j) :
        for k in range(0,maxlen_d) :
            if t_[i,k] != 0:
                freq = indices_freq[int(t_[i,k])]
                t[i,k,0] = freq // dim
                t[i,k,1] = freq % dim
            else:
                break

    # シャッフル処理
    z = list(zip(e, d, t))
    nr.seed(12345)
    nr.shuffle(z)
    e,d,t=zip(*z)
    nr.seed()

    e = np.array(e).reshape(j,maxlen_e,1)
    d = np.array(d).reshape(j,maxlen_d,1)
    t = np.array(t).reshape(j,maxlen_d,2)

    print('データセットの形式 (e.shape, d.shape, t.shape):('+str(e.shape)+','+str(d.shape)+','+str(t.shape)+')')

    #Encoder Inputデータをセーブ
    with open('/root/sakai_attention_seq2seq/keras_base/'+directory_name+'/dataset/e.pickle', 'wb') as f :
        pickle.dump(e , f)

    #Decoder Inputデータをセーブ
    with open('/root/sakai_attention_seq2seq/keras_base/'+directory_name+'/dataset/d.pickle', 'wb') as g :
        pickle.dump(d , g)

    #ラベルデータをセーブ
    with open('/root/sakai_attention_seq2seq/keras_base/'+directory_name+'/dataset/t.pickle', 'wb') as h :
        pickle.dump(t , h)

    #maxlenセーブ
    with open('/root/sakai_attention_seq2seq/keras_base/'+directory_name+'/dataset/maxlen.pickle', 'wb') as maxlen :
        pickle.dump([maxlen_e, maxlen_d] , maxlen)


    #各単語の出現頻度順位（降順）
    with open('/root/sakai_attention_seq2seq/keras_base/'+directory_name+'/dataset/freq_indices.pickle', 'wb') as f :
        pickle.dump(freq_indices , f)
    #出現頻度→インデックス変換
    with open('/root/sakai_attention_seq2seq/keras_base/'+directory_name+'/dataset/indices_freq.pickle', 'wb') as f :
        pickle.dump(indices_freq , f)

if __name__ == '__main__':  
    import numpy.random as nr
    import pickle
    import numpy as np
    import math
    import sys

    # ディレクトリ名を引数から取得
    directory_name = sys.argv[1]

    # generate_index_matrix.pyで保存した辞書や単語リスト、発言単語indexリストをgenerate_tensors関数によって処理する
    #generate_tensorの第一引数と第二引数はそれぞれ、入力の最大単語数と出力の最大単語数(これを超える対話はスキップ)
    generate_tensors(50, 50, directory_name)
