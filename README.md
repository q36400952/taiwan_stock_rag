# 📈 台股 RAG 智慧分析系統

> 結合即時新聞全文抓取、語意向量搜尋與大型語言模型（LLM），打造針對台灣股市的智慧分析工具。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/你的帳號/你的repo名稱/blob/main/taiwan_stock_rag.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🗂️ 目錄

- [系統簡介](#-系統簡介)
- [系統架構](#-系統架構)
- [什麼是 RAG？](#-什麼是-rag)
- [功能特色](#-功能特色)
- [快速開始](#-快速開始)
- [申請 Groq API Key](#-申請-groq-api-key)
- [使用說明](#-使用說明)
- [技術細節](#-技術細節)
- [常見問題](#-常見問題)
- [注意事項與免責聲明](#-注意事項與免責聲明)

---

## 🔍 系統簡介

本專案是一個基於 **RAG（檢索增強生成）** 架構的台股分析系統，能夠：

1. 自動從 **Google News** 抓取最新台股新聞全文
2. 利用 **語意向量搜尋** 找出與你問題最相關的新聞段落
3. 結合 **yfinance 即時股價資料**（含技術指標）
4. 透過 **Groq 免費 LLM（llama-3.3-70b）** 生成專業的繁體中文分析報告

只要輸入股票代號（如 `0050`、`2330`）和你的問題，系統就會自動完成所有工作。

---

## 🏗️ 系統架構

```
使用者輸入（股票代號 + 問題）
         │
         ▼
┌────────────────────────────────────┐
│          【知識庫建構】              │
│  Google News RSS → 全文抓取         │
│  （trafilatura，平行爬取，速度 5x）  │
│  → Chunking（300字/段，50字重疊）   │
└────────────────┬───────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│        【向量化 & 索引建立】          │
│  paraphrase-multilingual-MiniLM    │
│  → FAISS（Cosine Similarity）      │
└────────────────┬───────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│      【兩階段語意檢索 Retrieval】     │
│  粗排：Bi-encoder Top-10           │
│  精排：Cross-encoder Reranking     │
│        → 取最相關 Top-5 段落       │
└────────────────┬───────────────────┘
                 │         ┌──────────────────────┐
                 │         │    【股價直接注入】     │
                 │         │  yfinance 歷史股價     │
                 │         │  MA5/MA20/布林通道     │
                 │         └────────────┬─────────┘
                 ▼                      ▼
┌────────────────────────────────────────────────┐
│              【Prompt 組裝】                     │
│  股價摘要 + 相關新聞段落 + 使用者問題             │
└──────────────────────┬─────────────────────────┘
                       ▼
┌────────────────────────────────────────────────┐
│        【LLM 生成分析報告】                       │
│  Groq API → llama-3.3-70b-versatile            │
│  → 繁體中文專業分析（技術面 + 基本面 + 建議）     │
└────────────────────────────────────────────────┘
```

---

## 🤔 什麼是 RAG？

**RAG（Retrieval-Augmented Generation，檢索增強生成）** 是一種 AI 架構，解決了 LLM 的兩個核心問題：

| 純 LLM 的問題 | RAG 的解決方式 |
|---|---|
| 知識有截止日期，不知道最新股市資訊 | 即時從網路抓取最新新聞作為知識來源 |
| 可能憑空捏造（幻覺問題） | 強制 LLM 根據真實文件來回答 |
| 無法分析特定個股的即時數據 | 即時注入股價資料和相關新聞 |

簡單來說：**RAG = 先查資料，再讓 AI 根據查到的資料回答**，而不是讓 AI 靠記憶亂猜。

### 本系統的 RAG 流程

```
問題：「台積電近期值得買入嗎？」
  │
  ├─ 查股價：過去6個月收盤價、均線、波動率
  ├─ 查新聞：抓取15篇最新台積電相關新聞全文
  ├─ 切片：每篇新聞切成300字小段（共約100+段落）
  ├─ 向量化：將每段文字轉換成數學向量
  ├─ 語意搜尋：找出與「值得買入」最相關的5段新聞
  └─ 生成：LLM 根據股價 + 5段新聞，撰寫完整分析報告
```

---

## ✨ 功能特色

### 🔎 新聞全文語意搜尋（Two-stage RAG）

一般 RAG 系統只做一次搜尋，本系統使用**兩階段精排**，大幅提升準確度：

| 階段 | 方法 | 說明 |
|---|---|---|
| **粗排** | Bi-encoder + FAISS | 快速從100+段落中取出 Top-10 候選 |
| **精排** | Cross-encoder Reranking | 對每個 (問題, 段落) 配對重新打分，取 Top-5 |

> **為什麼需要兩階段？**
> Bi-encoder 很快但不夠精準（分開編碼問題和文件）；
> Cross-encoder 很精準但很慢（同時看問題和文件的關係）。
> 兩者結合，兼顧速度與精度。

### 📊 豐富的股價技術指標

系統會自動計算並注入以下指標：

- **MA5 / MA20**：5日、20日移動平均線
- **均線排列訊號**：多頭（MA5 > MA20）或空頭
- **日波動率**：收益率標準差，衡量風險
- **量能分析**：近5日量能與均量對比
- **布林通道**（圖表展示）：MA20 ± 2σ

### 🎨 深色主題股價走勢圖

- 收盤價線 + 布林通道
- MA5（黃線）、MA20（紅線）
- 依漲跌著色的成交量柱（紅漲綠跌）

---

## 🚀 快速開始

### 方法一：使用 Google Colab（推薦，零安裝）

1. 點擊上方 **Open in Colab** 按鈕
2. 申請並設定 Groq API Key（見下方說明）
3. 依序執行每個 Cell

### 方法二：本地端執行

```bash
# 1. 安裝相依套件（建議在虛擬環境中執行）
pip install "numpy>=1.24,<2.0"
pip install yfinance pandas feedparser trafilatura \
            "sentence-transformers>=2.7" faiss-cpu \
            gradio groq matplotlib requests

# 2. 設定環境變數
export GROQ_API_KEY="gsk_你的api金鑰"

# 3. 執行（將 .ipynb 轉成 .py 後執行，或用 Jupyter）
jupyter notebook taiwan_stock_rag.ipynb
```

---

## 🔑 申請 Groq API Key

本系統使用 **Groq** 提供的免費 LLM API。

### 申請步驟

1. 前往 [https://console.groq.com](https://console.groq.com)
2. 點右上角 **Sign Up** 註冊（支援 Google / GitHub 登入）
3. 登入後，左側選單點 **「API Keys」**
4. 點 **「Create API Key」**，輸入任意名稱後點 Submit
5. 複製顯示的 Key（格式：`gsk_xxxxxxxxxx`）

> ⚠️ Key 只會顯示一次，請立即複製保存！

### 在 Colab 中設定 Key

**方法一（推薦）：Colab Secrets**

```
左側工具列 → 🔑 鑰匙圖示（Secrets）
→ 點「+ Add new secret」
→ Name: GROQ_API_KEY
→ Value: 貼上你的 Key
→ 開啟「Notebook access」開關
```

**方法二：直接寫入程式碼（僅測試用，不安全）**

```python
os.environ["GROQ_API_KEY"] = "gsk_你的api金鑰"
```

### Groq 免費方案限制

| 項目 | 限制 |
|---|---|
| 費用 | 完全免費 |
| 每分鐘請求數 | 30 次 |
| 每日 Token 上限 | 約 14,400 Token |
| 每次分析消耗 | 約 2,000~3,000 Token |

> 免費額度每日約可分析 **50～100 次**，個人使用完全足夠。

---

## 📖 使用說明

### Notebook 各 Cell 說明

| Cell | 功能 | 執行順序 |
|---|---|---|
| Cell 1 | 安裝套件（含 NumPy 版本修正） | 第一次使用必須執行，執行後需**重啟 Runtime** |
| Cell 2 | 設定 Groq API Key | 每次重啟後需執行 |
| Cell 3 | 載入 AI 模型（Bi-encoder + Cross-encoder） | 每次重啟後需執行，約需 1-2 分鐘 |
| Cell 4 | 股價模組 | 每次重啟後需執行 |
| Cell 5 | 新聞抓取模組 | 每次重啟後需執行 |
| Cell 6 | RAG 核心（Chunking + FAISS + Reranking） | 每次重啟後需執行 |
| Cell 7 | Prompt 組裝 + LLM 呼叫 | 每次重啟後需執行 |
| Cell 8 | 股價走勢圖 | 每次重啟後需執行 |
| Cell 9 | 主流程整合 | 每次重啟後需執行 |
| Cell 10 | 快速測試（不開 UI） | 可選，直接在 Notebook 看結果 |
| Cell 11 | 啟動 Gradio UI | 開啟互動介面 |

### 執行流程（標準）

```
Cell 1 → 重啟 Runtime → Cell 2 → Cell 3 → Cell 4 → Cell 5
→ Cell 6 → Cell 7 → Cell 8 → Cell 9 → Cell 11
```

> ⚠️ Cell 1 安裝完後**必須重啟 Runtime**（Runtime → Restart session），然後從 Cell 2 繼續，**不需要重新執行 Cell 1**。

### 支援的股票代號

| 類型 | 範例 |
|---|---|
| 台灣 ETF | `0050`、`0056`、`00878`、`00915` |
| 台灣個股 | `2330`（台積電）、`2454`（聯發科）、`2317`（鴻海）|
| 金融股 | `2882`（國泰金）、`2881`（富邦金）|

> 輸入時**不需要加 `.TW`**，系統會自動處理。

### 分析報告結構

AI 生成的報告固定包含以下五個部分：

1. **📊 股價技術面解讀** — 均線、波動率、量能訊號解析
2. **📰 新聞基本面影響** — 相關新聞的正面/負面影響評估
3. **⚠️ 風險提示** — 2-3 個主要風險因素
4. **💡 綜合投資建議** — 短期（1-4週）操作方向建議
5. **📝 免責聲明** — 本分析僅供參考

---

## 🔬 技術細節

### 使用的模型

| 模型 | 用途 | 來源 |
|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | 向量化（支援中文） | HuggingFace |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking 精排 | HuggingFace |
| `llama-3.3-70b-versatile` | 生成分析報告 | Groq API |

### 為什麼選 Cosine Similarity 而非 L2 距離？

```
L2 距離：量的是兩個向量「距離有多遠」
Cosine：量的是兩個向量「方向有多接近」

語意搜尋關心的是「意思像不像」而非「數值差多少」
→ Cosine Similarity 更符合語意搜尋的本質
```

### Chunking 策略

```python
chunk_size = 300  # 每段最多 300 個字元
overlap    = 50   # 相鄰段落重疊 50 個字元

# 重疊的作用：
# 避免重要句子剛好被切在兩段的邊界，
# 確保每段都有前後文的連貫性。
```

### 主要相依套件版本要求

```
numpy >= 1.24, < 2.0   ← 必須，2.0 與 sentence-transformers 不相容
sentence-transformers >= 2.7
gradio >= 4.0
groq（最新版）
```

---

## ❓ 常見問題

### Q1：安裝完執行 Cell 3 出現 `ImportError: cannot import name '_center'`

**原因**：NumPy 2.x 版本與 sentence-transformers 不相容。

**解法**：
1. 將 Cell 1 的安裝指令替換為：
   ```bash
   !pip install -q "numpy>=1.24,<2.0"
   !pip install -q yfinance feedparser trafilatura \
       "sentence-transformers>=2.7" faiss-cpu gradio groq matplotlib
   ```
2. 執行後點 **Runtime → Restart session**
3. 從 Cell 2 重新開始執行

---

### Q2：Gradio 出現 `TypeError: Textbox.__init__() got an unexpected keyword argument 'show_copy_button'`

**原因**：Gradio 6.0 移除了 `show_copy_button` 參數。

**解法**：在 Cell 11 的 `gr.Textbox()` 中刪除 `show_copy_button=True,` 這一行。

---

### Q3：Gradio 出現 `UserWarning: The parameters have been moved from the Blocks constructor...`

**原因**：Gradio 6.0 將 `theme`、`css` 參數從 `gr.Blocks()` 移到 `launch()`。

**解法**：將 `theme` 和 `css` 從 `gr.Blocks(...)` 移到 `iface.launch(...)` 中。

---

### Q4：新聞抓取成功但大多數篇數的 body 是空的

**原因**：部分新聞網站有反爬蟲保護，trafilatura 無法取得全文。

**影響**：無全文的新聞會退化為只存標題，仍可用於分析，但精度稍低。

**這是正常現象**，一般 15 篇中會有 5~10 篇成功取得全文。

---

### Q5：上傳到 GitHub 顯示 "Invalid Notebook"

**原因**：在 Colab 執行過後，notebook 被自動存入格式不完整的 `metadata.widgets`。

**解法**：
- 在 Colab 中點 **編輯 → 清除所有輸出** 後再下載上傳
- 或手動在 JSON 中刪除 `"widgets": {...}` 區塊

---

### Q6：LLM 回傳 `❌ API 請求次數超限`

**原因**：Groq 免費版每分鐘限制 30 次請求。

**解法**：等待 1 分鐘後重試。若頻繁發生，可考慮使用較小的模型（如 `llama-3.1-8b-instant`）減少 Token 消耗。

---

## ⚠️ 注意事項與免責聲明

1. **本系統的分析結果僅供學習與參考之用，不構成任何投資建議。**
2. 股票投資有風險，過去績效不代表未來表現，請在做出投資決策前諮詢專業財務顧問。
3. 新聞資料來源為 Google News RSS，資訊的準確性取決於原始新聞來源。
4. 系統使用的 LLM 模型（llama-3.3-70b）可能產生不準確或過時的資訊，請批判性地閱讀分析結果。
5. 請遵守 Groq、yfinance、Google News 等服務的使用條款。

---

## 📄 授權

本專案採用 [MIT License](LICENSE) 授權，歡迎自由使用與修改。

---

<div align="center">

如果本專案對你有幫助，歡迎給個 ⭐ Star！

</div>
