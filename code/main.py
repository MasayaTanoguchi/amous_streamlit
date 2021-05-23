import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import norm
import glob
import re

@st.cache
def preprocess_table(CSV_PATH):
    df = pd.read_csv(CSV_PATH, encoding='cp932')
    #df = pd.read_excel(CSV_PATH, encoding='cp932')
    df = df.rename(columns={'Unnamed: 0':'メンバー'})
    df['メンバー'] = df['メンバー'].map(lambda x : x.split('\n')[0])
    
    df.index = df['メンバー']
    df.drop('メンバー', axis=1, inplace=True)
    df = df.T.copy()
    return df

def get_agg_table(df):
    df_agg = pd.DataFrame(index=df.index)
    df_agg['ゲーム参加回数'] = df['開始したゲーム']
    df_agg['ゲーム勝率'] = (df['インポスターでの投票勝利'] + df['インポスターでの殺害勝利'] + df['インポスターでのサボタージュ勝利'] + df['クルーでの投票勝利'] + df['クルーでのタスク勝利']) / df['開始したゲーム']
    df_agg['クルーでのゲーム勝率'] = (df['クルーでの投票勝利'] + df['クルーでのタスク勝利']) / df['クルーになった回数']
    df_agg['インポスターでのゲーム勝率'] = (df['インポスターでの投票勝利'] + df['インポスターでの殺害勝利']) / df['インポスターになった回数']
    df_agg['生存率'] = (df['開始したゲーム'] - (df['殺害された回数'] + df['追放された回数'])) / df['開始したゲーム']
    df_agg['緊急会議開催率'] = df['開いた緊急会議'] / df['開始したゲーム']
    df_agg['平均サボタージュ解決数'] = df['解決したサボタージュ'] / df['開始したゲーム']
    df_agg['タスク完遂率'] = df['全タスク完了'] / df['クルーになった回数']
    df_agg['平均完了タスク数'] = df['完了したタスク'] / df['クルーになった回数']
    df_agg['平均殺害数'] = df['インポスターによる殺害'] / df['インポスターになった回数']
    return df_agg

def compute_personality_table(df_agg):
    df_personality = pd.DataFrame()
    df_personality['A'] = df_agg['ゲーム勝率'] * df_agg['生存率'] * df_agg['緊急会議開催率'] * df_agg['タスク完遂率'] * df_agg['平均殺害数']
    df_personality['B'] = df_agg['ゲーム勝率'] * df_agg['生存率'] * df_agg['タスク完遂率'] * df_agg['平均殺害数']
    df_personality['C'] = df_agg['ゲーム勝率'] * df_agg['生存率']
    df_personality['D'] = df_agg['インポスターでのゲーム勝率'] * df_agg['生存率'] * df_agg['平均殺害数']
    df_personality['E'] = df_agg['インポスターでのゲーム勝率'] * df_agg['生存率'] * df_agg['平均殺害数'] * df_agg['緊急会議開催率']
    df_personality['F'] = df_agg['インポスターでのゲーム勝率'] * df_agg['平均殺害数'] * df_agg['タスク完遂率']
    df_personality['G'] = df_agg['クルーでのゲーム勝率'] * df_agg['生存率'] * df_agg['平均殺害数']
    df_personality['H'] = df_agg['クルーでのゲーム勝率'] * df_agg['生存率'] * df_agg['平均殺害数'] * df_agg['タスク完遂率']
    df_personality['I'] = df_agg['クルーでのゲーム勝率'] * df_agg['生存率']
    df_personality['J'] = (1.0 - df_agg['ゲーム勝率']) * (1.0 - df_agg['タスク完遂率']) * (1.0 - df_agg['緊急会議開催率']) * (1.0 - df_agg['生存率'])
    df_personality['K'] = (1.0 - df_agg['ゲーム勝率']) * df_agg['平均殺害数'] * df_agg['タスク完遂率']
    df_personality['L'] = (1.0 - df_agg['ゲーム勝率']) * df_agg['緊急会議開催率']
    df_personality['M'] = (1.0 - df_agg['ゲーム勝率']) * df_agg['平均殺害数']
    df_personality['N'] = (1.0 - df_agg['ゲーム勝率']) * df_agg['生存率'] * df_agg['タスク完遂率'] * df_agg['平均殺害数']
    return df_personality

def compute_personality(df_personality, df_base):
    keys = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']
    
    mean = df_base.mean()
    std = df_base.std()
    
    df_persentile = pd.DataFrame()
    for key in keys:
        df_persentile['%s'%key] = df_personality['%s'%key].map(lambda x : norm.cdf(x, loc=mean['%s'%key], scale=std['%s'%key]))
    
    personality_result = {}
    for user in df_persentile.index:
        max_ = df_persentile.loc['%s'%user].max()
        result = df_persentile.loc['%s'%user][df_persentile.loc['%s'%user] == max_].index[0]
        personality_result[user] = result
    return personality_result

def get_personality_caption(label):
    personality = {'A':{'main':'最強タイプ','sub':'戦闘力のスーパーインフレ','metrics':''},
                   'B':{'main':'プロフェッショナル仕事の流儀タイプ','sub':'','metrics':''},
                   'C':{'main':'草葉の陰からこんにちはタイプ','sub':'骨の髄まで働かされ','metrics':''},
                   'D':{'main':'残虐なるサイコキラータイプ','sub':'最白位置で人肉食う','metrics':''},
                   'E':{'main':'華麗なる詐欺師タイプ','sub':'意のままにあなたを操る','metrics':''},
                   'F':{'main':'悩める堕天使タイプ','sub':'本当はもう殺したくない...','metrics':''},
                   'G':{'main':'最強の正直ものタイプ','sub':'このゲームは信じるゲーム','metrics':''},
                   'H':{'main':'名探偵タイプ','sub':'犯人はこの中にいる！','metrics':''},
                   'I':{'main':'勤勉兵士タイプ','sub':'','metrics':'我が身を捧げ切る！'},
                   'J':{'main':'器用貧乏タイプ','sub':'明日から多分本気出す','metrics':''},
                   'K':{'main':'全力社畜タイプ','sub':'月間残業時間天城越え','metrics':''},
                   'L':{'main':'大阪のおばちゃんタイプ','sub':'ただただ話したいだけ','metrics':''},
                   'M':{'main':'快楽殺人タイプ','sub':'後先気にせずキルかます','metrics':''},
                   'N':{'main':'A.Tフィールド全開タイプ','sub':'世界とかどうなってもよくない？','metrics':''}}
    return personality[label]['main']

if __name__ == '__main__':
    
    # 参照データ
    CSV_PATH = 'data/csv/among_us_agg_list.csv'
    IMG_FOLDER_PATH = 'data/img'
    IMG_EXPLAIN_PATH = IMG_FOLDER_PATH + '/Z_explain.JPG'
    IMG_LIST_PATH = IMG_FOLDER_PATH + '/Z_list.JPG'
    IMG_INTRO_PATH = IMG_FOLDER_PATH + '/Z_intro.jpg'

    df = preprocess_table(CSV_PATH)
    df_agg = get_agg_table(df)
    df_personality = compute_personality_table(df_agg)
    
    # UIの基本設定
    st.title('あもあす性格診断')
    #st.write('© Copyright 2021, aiko eshiro - ゆるふわAmongUs村')
    st.write('made by aiko eshiro - ゆるふわAmongUs村, 2021')
    image_intro = Image.open(IMG_INTRO_PATH)
    st.image(image_intro,use_column_width=True)


    # 性格診断者のプレイデータ
    st.header('データ入力フォーム')
    name = st.text_input('あなたのお名前', 'ダミー１')
    input01 = st.number_input('通報された死体数',0,10000,24,step=1)
    input02 = st.number_input('開いた緊急会議',0,10000,58,step=1)
    input03 = st.number_input('完了したタスク',0,10000,1006,step=1)
    input04 = st.number_input('全タスク完了',0,10000,80,step=1)
    input05 = st.number_input('解決したサボタージュ',0,10000,220,step=1)
    input06 = st.number_input('インポスターによる殺害',0,10000,71,step=1)
    input07 = st.number_input('殺害された回数',0,10000,70,step=1)
    input08 = st.number_input('追放された回数',0,10000,34,step=1)
    input09 = st.number_input('連続でクルーになった回数',0,10000,9,step=1)
    input10 = st.number_input('インポスターになった回数',0,10000,36,step=1)
    input11 = st.number_input('クルーになった回数',0,10000,190,step=1)
    input12 = st.number_input('開始したゲーム',0,10000,226,step=1)
    input13 = st.number_input('終了したゲーム',0,10000,222,step=1)
    input14 = st.number_input('インポスターでの投票勝利',0,10000,0,step=1)
    input15 = st.number_input('インポスターでの殺害勝利',0,10000,7,step=1)
    input16 = st.number_input('インポスターでのサボタージュ勝利',0,10000,8,step=1)
    input17 = st.number_input('クルーでの投票勝利',0,10000,99,step=1)
    input18 = st.number_input('クルーでのタスク勝利',0,10000,22,step=1)
    
    inputs = [input01, input02, input03, input04, input05, input06, input07, input08, input09, input10, input11, input12, input13, input14, input15, input16, input17, input18]
    st.header('')
    start = st.button('診断スタート')

    if start:
        df_player = pd.DataFrame(dict(zip(df.columns, [[i] for i in inputs])), index=[name])
        df_player_agg = get_agg_table(df_player)
        df_player_personality = compute_personality_table(df_player_agg)
        df_player_result = compute_personality(df_player_personality, df_base=df_personality)
        personality_label = df_player_result[name]
        personality_caption = get_personality_caption(personality_label)
        
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        IMG_PATHS = [IMG_FOLDER_PATH + '/' + l + '.jpg' for l in labels]
        st.header('診断結果')
        st.subheader('%sさん！あなたの性格は....'%name)
        st.subheader('「'+personality_caption+'」')
        img_path_dict = dict(zip(labels, IMG_PATHS))
        img_path = img_path_dict[personality_label]
        image = Image.open(img_path)
        st.image(image,use_column_width=True)

        st.header('データ集計結果')
        win_r = int(round(df_player_agg.loc[name, 'ゲーム勝率'] * 100))
        win_r_imposter = int(round(df_player_agg.loc[name, 'インポスターでのゲーム勝率'] * 100))
        win_r_crew = int(round(df_player_agg.loc[name, 'クルーでのゲーム勝率'] * 100))
        alive_r = int(round(df_player_agg.loc[name, '生存率'] * 100))
        mtg_r = int(round(df_player_agg.loc[name, '緊急会議開催率'] * 100))
        kill_n = int(round(df_player_agg.loc[name, '平均殺害数']))
        task_r = int(round(df_player_agg.loc[name, 'タスク完遂率'] * 100))

        agg01 = st.number_input('ゲーム勝率：全体（％）',-100,100,win_r ,step=1)
        agg02 = st.number_input('ゲーム勝率：インポスター（％）',-100,100,win_r_imposter,step=1)
        agg03 = st.number_input('ゲーム勝率：クルー（％）',-100,100,win_r_crew,step=1)
        agg04 = st.number_input('生存率（％）',-100,100,alive_r,step=1)
        agg05 = st.number_input('緊急会議開催率（％）',-100,100,mtg_r,step=1)
        agg06 = st.number_input('平均殺害数（人）',-100,100,kill_n,step=1)
        agg07 = st.number_input('タスク完遂率（％）',-100,100,task_r,step=1)

        st.header('性格一覧')
        image_list = Image.open(IMG_LIST_PATH)
        st.image(image_list, use_column_width=True)

        st.header('性格の説明')
        image_explain = Image.open(IMG_EXPLAIN_PATH)
        st.image(image_explain, use_column_width=True)


