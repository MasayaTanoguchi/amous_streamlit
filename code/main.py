import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import norm
import glob
import re

def preprocess_table(CSV_PATH):
    df = pd.read_csv(CSV_PATH, encoding='cp932')
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
    df_agg['生存率'] = (df['開始したゲーム'] - (df['殺害された回数'] - df['追放された回数'])) / df['開始したゲーム']
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

def compute_personality(df_personality):
    keys = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']
    
    mean = df_personality.mean()
    std = df_personality.std()
    
    df_persentile = pd.DataFrame()
    for key in keys:
        df_persentile['%s'%key] = df_personality['%s'%key].map(lambda x : norm.cdf(x, loc=mean['%s'%key], scale=std['%s'%key]))
    #df_persentile.columns = mains
    
    personality_result = {}
    for user in df_persentile.index:
        max_ = df_persentile.loc['%s'%user].max()
        result = df_persentile.loc['%s'%user][df_persentile.loc['%s'%user] == max_].index[0]
        #print(user,':',df_persentile.loc['%s'%user][df_persentile.loc['%s'%user] == max_].index[0])
        personality_result[user] = result
    return personality_result

def get_personality_caption(label):
    personality = {'A':{'main':'フリーザータイプ','sub':'戦闘力のスーパーインフレ','metrics':''},
                   'B':{'main':'プロフェッショナル仕事の流儀タイプ','sub':'','metrics':''},
                   'C':{'main':'草葉の陰からこんにちはタイプ','sub':'骨の髄まで働かされ','metrics':''},
                   'D':{'main':'残虐なるサイコキラータイプ','sub':'最白位置で人肉食う','metrics':''},
                   'E':{'main':'華麗なる詐欺師タイプ','sub':'意のままにあなたを操る','metrics':''},
                   'F':{'main':'悩める堕天使タイプ','sub':'本当はもう殺したくない...','metrics':''},
                   'G':{'main':'最強の正直ものタイプ','sub':'このゲームは信じるゲーム','metrics':''},
                   'H':{'main':'名探偵コナンガス','sub':'犯人はこの中にいる！','metrics':''},
                   'I':{'main':'調査兵団タイプ','sub':'心臓を捧げ終えた','metrics':''},
                   'J':{'main':'器用貧乏タイプ','sub':'明日から多分本気出す','metrics':''},
                   'K':{'main':'全力社畜タイプ','sub':'月間残業時間天城越え','metrics':''},
                   'L':{'main':'大阪のおばちゃんタイプ','sub':'ただただ話したいだけ','metrics':''},
                   'M':{'main':'快楽殺人タイプ','sub':'後先気にせずキルかます','metrics':''},
                   'N':{'main':'A.Tフィールド全開タイプ','sub':'世界とかどうなってもよくない？','metrics':''}}
    return personality[label]['main']
    

if __name__ == '__main__':
    
    CSV_PATH = 'data/csv/among_us_agg_list.csv'
    IMG_FOLDER_PATH = 'data/img'
    
    df = preprocess_table(CSV_PATH)
    df_agg = get_agg_table(df)
    df_personality = compute_personality_table(df_agg)
    personality_result = compute_personality(df_personality)
    
    st.title('あもあす性格診断')
    player = st.selectbox('プレイヤー選択',(tuple(df.index)))
    personality_label = personality_result[player]
    personality_caption = get_personality_caption(personality_label)
    
    IMG_PATHS = glob.glob(IMG_FOLDER_PATH + '/*')
    labels = [re.sub(IMG_FOLDER_PATH+'|'+'\\\\|.jpg','',PATH) for PATH in IMG_PATHS]
    
    st.subheader('%sさん！あなたの性格は....'%player)
    
    st.header(personality_caption)
    
    img_path_dict = dict(zip(labels, IMG_PATHS))
    img_path = img_path_dict[personality_label]
    image = Image.open(img_path)
    st.image(image,use_column_width=True)
