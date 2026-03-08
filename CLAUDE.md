# kabu-advisor — CLAUDE.md

## プロジェクト概要

日本株の個人投資家（So）向け、AIマルチエージェント株式アドバイザー。
毎朝、複数のAIエージェントが独立して分析・議論し、最終的な打ち手を提案する。
朝に確認して手動発注、翌朝に結果を踏まえて再分析するサイクルで運用する。

---

## ディレクトリ構成

```
kabu-advisor/
├── CLAUDE.md
├── main.py              # FastAPIエントリーポイント
├── advisor/
│   ├── agents.py        # マルチエージェント議論エンジン
│   ├── data.py          # データ収集（yfinance・J-Quants・RSS）
│   └── prompts.py       # 各エージェントのプロンプト定義
├── database/
│   └── db.py            # SQLite操作
├── templates/
│   └── index.html       # スマホ対応フロントエンド
├── requirements.txt
└── .env                 # ANTHROPIC_API_KEY等
```

---

## 技術スタック

- **バックエンド**: FastAPI + Python 3.11+
- **データ取得**: yfinance, J-Quants API, Yahoo!ファイナンスRSS, feedparser
- **AI**: Anthropic Claude API (claude-sonnet-4-20250514) + web_search tool
- **DB**: SQLite (SQLAlchemy)
- **フロントエンド**: Vanilla HTML/CSS/JS（スマホ対応）
- **デプロイ**: Render

---

## コアロジック：マルチエージェント議論システム

### 思想

「一人の思考は固執する」問題を解決するため、役割バイアスを持つ複数エージェントに
同じ情報を渡して独立分析させ、モデレーターが統合する。

人間が容易にできる判断（ニュースが良いから買い等）は価値がない。
AIが提供すべき価値は以下に限定する：

1. 複数情報の矛盾検出（表面的な好材料と機関の動きの乖離等）
2. 感情排除の数値ベース判断（根拠を必ず定量化）
3. 人間が見落とす関連情報の網羅（サプライチェーン・競合・セクター全体）
4. 目標から逆算したリスク許容度計算
5. 過去の自分の判断の検証と学習

### エージェント構成

```python
AGENTS = {
    "bull": {
        "name": "強気アナリスト",
        "role": "買うべき・保有すべき根拠を徹底的に探す。反論は一切しない。",
        "bias": "強制的に強気の視点のみで分析する"
    },
    "bear": {
        "name": "弱気アナリスト",
        "role": "売るべき・避けるべき根拠を徹底的に探す。反論は一切しない。",
        "bias": "強制的に弱気の視点のみで分析する"
    },
    "risk": {
        "name": "リスク管理官",
        "role": "目標・期日・元手・現在資産から数学的にリスク許容度を計算する。",
        "bias": "感情なし。数字のみで語る。"
    },
    "moderator": {
        "name": "モデレーター",
        "role": "3人の議論を統合し、矛盾を指摘し、最終打ち手を決定する。",
        "bias": "中立。矛盾点に対して追加質問を行い、最も論理的な結論を導く。"
    }
}
```

### 議論フロー

```
Step 1: データ収集（data.py）
  └─ yfinance: 対象銘柄の株価・出来高・テクニカル
  └─ J-Quants: 決算・業績データ
  └─ RSS: Yahoo!ファイナンス最新ニュース
  └─ Claude web_search: 以下を網羅的に検索（各銘柄3〜5クエリ）
       - 前日の米国市場・為替・日経先物
       - 銘柄個別の最新ニュース（複数角度から）
       - セクター・テーマ動向
       - 競合他社・サプライチェーン動向
       - アナリスト評価変化

Step 2: 強気アナリスト分析（独立）
  └─ 上記データを全て渡す
  └─ 買い・保有の根拠のみを徹底的に列挙

Step 3: 弱気アナリスト分析（独立）
  └─ 同じデータを渡す
  └─ 売り・回避の根拠のみを徹底的に列挙

Step 4: リスク管理官分析（独立）
  └─ 目標金額・期日・現在資産を渡す
  └─ 残り日数で必要なリターン率を計算
  └─ 許容できる最大損失額を計算
  └─ 目標が数学的に実現可能かを判定

Step 5: モデレーター統合
  └─ 3人の分析を全て渡す
  └─ 矛盾点・見解の相違を指摘
  └─ 最終打ち手を決定（根拠必須）
  └─ 自信度を%で明示
  └─ 「わからない」場合はわからないと明言
```

---

## データ収集仕様（data.py）

### yfinance取得項目

```python
# 銘柄コードは「7203.T」形式（TSE銘柄）
def get_stock_data(ticker: str) -> dict:
    """
    取得項目:
    - 直近20日の終値・出来高
    - 52週高値・安値
    - 移動平均（5日・25日・75日）
    - RSI（14日）
    - MACD
    - 出来高移動平均比
    """
```

### J-Quants API取得項目

```python
def get_fundamental_data(ticker: str) -> dict:
    """
    取得項目:
    - 直近決算（売上・営業利益・純利益）
    - 業績予想との乖離率
    - PER・PBR・配当利回り
    """
```

### Claude web_search検索クエリ設計

```python
SEARCH_QUERIES = [
    "{company}株 最新ニュース",
    "{company} 決算 業績 {year}",
    "{sector}セクター 市場動向 今日",
    "NYダウ ナスダック 前日終値",
    "ドル円 為替 今朝",
    "日経平均 先物 今日",
    "{company} 競合 {competitor} 比較",
    "{company} アナリスト 目標株価"
]
# 検索結果は全文をコンテキストに含める（要約しない）
# AIが自ら情報の重要度を判断する
```

---

## プロンプト設計原則（prompts.py）

### 共通ルール（全エージェント）

```
- 根拠のない推測を禁止する
- 「〜と思われる」「〜かもしれない」は使わない
- 数値・ソースを必ず明示する
- 「わからない」場合は明示的にそう述べる
- 人間が容易に判断できることを繰り返さない
```

### モデレーター最終出力フォーマット

```
## 本日の打ち手
[具体的なアクション: 買う/売る/持ち越す/銘柄変更]
銘柄・数量・価格帯まで明示

## 根拠（数値ベース）
[定量的な根拠のみ。感覚的表現禁止]

## 強気・弱気の主な対立点
[議論の核心だった論点を明示]

## リスクシナリオ
[このアクションが外れる場合の具体的条件]

## 目標達成の実現可能性
[数学的計算に基づく評価。無謀なら正直に指摘]

## 自信度
[X%] 理由: [なぜその数値か]

## 明日の確認事項
[翌朝最初に確認すべき具体的な指標・ニュース]
```

---

## データベース設計（db.py）

```sql
-- 設定テーブル
CREATE TABLE settings (
    id INTEGER PRIMARY KEY,
    capital INTEGER,          -- 元手
    current_amount INTEGER,   -- 現在資産
    target_amount INTEGER,    -- 目標金額
    deadline DATE,            -- 期日
    stocks TEXT,              -- 対象銘柄（JSON配列）
    created_at TIMESTAMP
);

-- 提案テーブル
CREATE TABLE proposals (
    id INTEGER PRIMARY KEY,
    date DATE,
    raw_data TEXT,            -- 収集した全データ（JSON）
    bull_analysis TEXT,       -- 強気アナリスト分析
    bear_analysis TEXT,       -- 弱気アナリスト分析
    risk_analysis TEXT,       -- リスク管理官分析
    final_proposal TEXT,      -- モデレーター最終提案
    confidence INTEGER,       -- 自信度（%）
    created_at TIMESTAMP
);

-- 結果テーブル
CREATE TABLE results (
    id INTEGER PRIMARY KEY,
    date DATE,
    proposal_id INTEGER,
    pnl INTEGER,              -- 損益（円）
    memo TEXT,
    amount_after INTEGER,     -- 取引後資産
    FOREIGN KEY (proposal_id) REFERENCES proposals(id)
);
```

---

## API設計（main.py）

```
GET  /                    # メインUI（HTML）
GET  /api/status          # 現在の設定・進捗
POST /api/setup           # 初期設定
POST /api/generate        # 今日の提案を生成
GET  /api/proposal/today  # 今日の提案を取得
POST /api/result          # 結果を記録
GET  /api/history         # 履歴一覧
```

---

## フロントエンド要件（index.html）

- **スマホファースト**（max-width: 480px）
- ダークテーマ
- 単一HTMLファイル（CDN使用可）
- 画面構成:
  1. ヘッダー: 現在資産・目標・進捗バー・残り日数
  2. タブ: 今日 / 議論の詳細 / 履歴 / 設定
  3. 「今日の打ち手を生成」ボタン
  4. 結果記録フォーム
  5. 「議論の詳細」タブで各エージェントの分析を折りたたみ表示

---

## 環境変数（.env）

```
ANTHROPIC_API_KEY=sk-...
JQUANTS_REFRESH_TOKEN=...   # J-Quants APIトークン
```

---

## Render デプロイ設定

```yaml
# render.yaml
services:
  - type: web
    name: kabu-advisor
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## 開発優先順位

1. データ収集基盤（yfinance + RSS）が動くことを確認
2. シングルエージェントで提案生成の動作確認
3. マルチエージェント化（bull/bear/risk/moderator）
4. DBへの保存・履歴参照
5. フロントエンド（スマホUI）
6. J-Quants API連携追加
7. Renderデプロイ

---

## 注意事項

- **投資判断はSoさん自身が行う**。ツールはあくまで情報整理と提案のみ。
- API呼び出しコストを意識し、1回の提案生成で4〜5回のClaude API呼び出しになる設計とする。
- yfinanceは15分遅延のため、朝の取引前（8:00〜8:30）に生成することを推奨。
- J-Quants APIは無料枠の制限に注意（1日のリクエスト数上限あり）。
