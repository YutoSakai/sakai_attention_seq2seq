# coding: utf-8

import numpy as np
import csv
import glob
import pickle

# REQREQ:をREQREQ, :に分解
def modification(word) :
    if len(word) > 6 and word[:6] == 'REQREQ':
        modified = ['REQREQ', word[6:]]
    elif len(word) > 6 and word[:6] == 'RESRES' :
        modified = ['RESRES', word[6:]]
    else :
        modified = [word]
    return modified

# 品詞分解
def decomposition(file, jumanpp):

    # fileを読み込む
    f=open(file, 'r')

    # ファイル内のテキストを行ごとの配列として読み込む
    # [
    #     '私の名前は千速です。よろしくお願いします',
    #     'こんにちは。千速さん。'
    # ]
    # のような形になる
    data = f.readlines()
    print('読み込んだ行数:'+str(len(data)))

    # 単語リストを初期化
    parts = []

    # 行ごとに回す
    for i in range(len(data)) :
        try:
            # jumanppに入力するとエラーが出る文字がないなら処理する
            if len(data[i].encode('utf-8')) <= 4096:

                # REQやRESだとこれ自体が分解されてしまうのでREQREQとRESRESに置換
                text = data[i]
                text = text.replace(' ', '　')
                text = text.replace('\n', '')
                text = text.replace('REQ:', 'REQREQ')
                text = text.replace('RES:', 'RESRES')

                # jumanで分解
                result = jumanpp.analysis(text)
            else :
                print(str(i)+'行目にエラー')
                continue

            # 単語ごとの配列初期化
            tmp_part = []

            # 分かち書きした単語ごとに回す
            for mrph in result.mrph_list():
                # modification関数によってtmp_partに単語配列をconcat
                tmp_part += modification(mrph.midasi)

            # partsにtmp_partの内容をappend
            # このappendによって二次元配列になる
            # generate_index_matrix.py内で次元を減らしている
            parts.append(tmp_part)

            if i % 5000 == 0 :
                print(str(i)+'行処理完了')
        except:
            print(str(i)+'行目に例外')

    return parts


# 分解結果を最終処理
def genarate_npy(parts_list):

    # 最終処理配列初期化
    list_corpus = []

    # dictの場合にこれによって配列に直される
    mat = [ v for v in parts_list]

    # list_corpusに実際にappendした数をカウントするための変数を初期化
    j = 0

    # 単語ごとにまわす
    for i in range(0,len(mat)):

        # 空やEOSなどでないなら実行
        if len(mat[i]) != 0 and mat[i][0] != '' and mat[i][0] != '@' and mat[i][0] != 'EOS' and mat[i][0] != ':':

            #list_corpusへi番目の単語をappend
            list_corpus.append(mat[i])

            if i % 1000000 == 0:
                print(str(i)+'単語中'+str(j)+'単語処理成功')
            j += 1

    print('処理結果'+str(len(list_corpus))+'単語有効')
    del mat
    return list_corpus


# メイン処理
if __name__ == '__main__':
    from pyknp import Juman
    import sys
    import numpy as np
    import csv
    import glob
    import re
    import pickle

    # ディレクトリ名を引数から取得
    directory_name = sys.argv[1]

    # Jumanオブジェクトを生成
    jumanpp = Juman()

    # rawディレクトリ内にある.txtファイルパスのリストをglobによって取得する
    file_list=glob.glob('/root/research/data/'+directory_name+'/raw/*.txt')
    print('rawディレクトリ内のtxtファイル数:'+str(len(file_list)))

    # ファイル名でソートする(日付順などになる)
    file_list.sort()

    # 単語の配列を初期化
    parts_list = []

    # txtファイルごとに繰り返す
    for j in range(len(file_list)):
        print('これから読み込むtxtファイルのパス:'+str(file_list[j]))

        # decompositionメソッドによってtxtファイルを読み込んでjumanで処理
        # 分かち書きした単語の配列をparts_listにconcat
        parts_list += decomposition(file_list[j], jumanpp)

    # 最終処理を行う
    generated_list = genarate_npy(parts_list)

    # 二次元の配列を　list/list_corpus.pickle　にセーブする
    # [
    #     ['REQREQ', ':', '私', 'の', '名前', 'は', '千', '速', 'です'],
    #     ['RESRES', ':', '私', 'の', '名前', 'は', '田中', 'です'],
    # ]
    # のような形である
    with open('/root/research/data/'+directory_name+'/list/list_corpus.pickle', 'wb') as g :
        pickle.dump(generated_list, g)

    print('総単語数:'+str(len(generated_list)))
    del generated_list