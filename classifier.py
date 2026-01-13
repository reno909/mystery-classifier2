import pandas as pd
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 1. データの読み込み
try:
    # 210件のデータを読み込みます
    df = pd.read_csv('mystery_dataset.csv')
except FileNotFoundError:
    print("エラー：'mystery_dataset.csv' が見つかりません。同じフォルダに置いてください。")
    exit()

# 2. 日本語の解析（単語に分ける）
t = Tokenizer()
def tokenize(text):
    tokens = t.tokenize(str(text))
    words = [token.base_form for token in tokens if token.part_of_speech.split(',')[0] in ['名詞', '動詞', '形容詞']]
    return " ".join(words)

print("AIがミステリー小説のサマリーを読み込んでいます...")
df['tokenized_text'] = df['Trick_Summary'].apply(tokenize)

# 3. 学習用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(
    df['tokenized_text'], df['Label'], test_size=0.2, random_state=42
)

# 4. 単語を数値に変換（TF-IDF）
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. SVMで学習
model = SVC(kernel='linear', probability=True)
model.fit(X_train_vec, y_train)

# 6. 結果の表示
y_pred = model.predict(X_test_vec)
print("\n===== 判定精度（テスト結果） =====")
print(f"正解率: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

# 7. 判定に影響した単語ランキング（卒論用）
feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_.toarray()[0]
features_df = pd.DataFrame({'word': feature_names, 'importance': coefs})
top15 = features_df.sort_values(by='importance', ascending=False).head(15)

print("\n===== AIが『一人二役』と見抜く際に重視した単語トップ15 =====")
print(top15[['word', 'importance']])

# 8. 現代ミステリーの判定シミュレーション
def predict_new(text):
    vec = vectorizer.transform([tokenize(text)])
    prob = model.predict_proba(vec)[0][1]
    return prob

print("\n===== 未知のトリック判定テスト =====")
test_sample = "犯人は被害者の親友になりすまし、カツラと眼鏡で外見を変え、鏡の反射を使って目撃者を騙した。"
score = predict_new(test_sample)
print(f"テスト文章: {test_sample}")
print(f"判定結果: このトリックが『一人二役』である確率は {score*100:.2f}% です。")

# 誤分類されたデータの特定
# y_test のインデックスを使って、元のデータフレームから該当する行を抽出
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=X_test.index)
misclassified = results_df[results_df['Actual'] != results_df['Predicted']]

print("\n===== AIが判定を間違えた作品の特定 =====")
for idx in misclassified.index:
    print(f"ID: {df.loc[idx, 'ID']}")
    print(f"タイトル: {df.loc[idx, 'Title']}")
    print(f"正解: {df.loc[idx, 'Label']} / AIの予測: {misclassified.loc[idx, 'Predicted']}")
    print(f"理由の考察用サマリー: {df.loc[idx, 'Trick_Summary']}\n")
