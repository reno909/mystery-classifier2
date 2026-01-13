import streamlit as st
import pandas as pd
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# --- ページ設定 ---
st.set_page_config(page_title="ミステリー・トリック分類器", layout="centered")

# --- 1. データの読み込みと学習 ---
@st.cache_resource
def load_and_train():
    # データ読み込み
    df = pd.read_csv('mystery_dataset.csv', encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    
    t = Tokenizer()
    def tokenize(text):
        tokens = t.tokenize(str(text))
        return " ".join([token.base_form for token in tokens if token.part_of_speech.split(',')[0] in ['名詞', '動詞', '形容詞']])
    
    df['tokenized_text'] = df['Trick_Summary'].apply(tokenize)
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['tokenized_text'])
    y = df['Label']
    
    model = SVC(kernel='linear', probability=True)
    model.fit(X, y)
    return model, vectorizer, t, tokenize

model, vectorizer, t, tokenize = load_and_train()

# --- 2. セッション状態の管理 ---
if "user_text" not in st.session_state:
    st.session_state.user_text = ""

def clear_text():
    st.session_state.user_text = ""

# --- 3. UI ---
st.title("🕵️‍♂️ ミステリー・トリック分類器")
st.sidebar.info("卒業論文用プロトタイプ: 江戸川乱歩『類別トリック集成』に基づく分類")

st.subheader("あらすじ（サマリー）を入力")

user_input = st.text_area(
    "あらすじを詳しく入力してください：", 
    value=st.session_state.user_text,
    key="user_text_area",
    placeholder="例：犯人は被害者の親友になりすまし、変装して現場を立ち去った...", 
    height=150
)
st.session_state.user_text = user_input

col1, col2 = st.columns([1, 1])
with col1:
    predict_btn = st.button("AI判定を開始", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("入力欄をクリア", on_click=clear_text, use_container_width=True)

# --- 4. 判定と可視化 ---
if predict_btn:
    if st.session_state.user_text:
        tokenized_input = tokenize(st.session_state.user_text)
        vec_input = vectorizer.transform([tokenized_input])
        prob = model.predict_proba(vec_input)[0][1]
        
        st.divider()
        st.subheader("判定結果")
        
        if prob > 0.5:
            st.error(f"判定： **一人二役トリックの可能性が高い**（確率: {prob*100:.1f}%）")
        else:
            st.success(f"判定： **他のトリックの可能性が高い**（確率: {prob*100:.1f}%）")
        
        # --- ここを書き換え：文字化けしないStreamlit標準グラフ ---
        st.write("#### 💡 AIが注目した単語")
        
        feature_names = vectorizer.get_feature_names_out()
        coefs = model.coef_.toarray()[0]
        importance_df = pd.DataFrame({'単語': feature_names, '重要度': coefs})
        
        # 上位10単語を抽出
        top_words = importance_df.sort_values(by='重要度', ascending=False).head(10)
        
        # Streamlitのネイティブグラフを表示（Matplotlibを使わない）
        st.bar_chart(data=top_words, x='単語', y='重要度', horizontal=True)
        
        st.caption("※グラフの数値が高いほど、AIが『一人二役』だと判断する強い材料になっています。")
    else:
        st.warning("あらすじを入力してください。")