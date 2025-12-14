# 專案名稱：AI 金融交易系統

## 專案描述
本專案旨在開發一個基於機器學習的金融交易系統，專注於台灣股票市場（2020-2024）。該系統通過技術指標和機器學習模型預測股票趨勢，並與 0050 基準進行比較。

## 功能模組

### 1. `data_pipeline.py`
- **功能描述**：
  - 提供數據處理功能，包括下載股票數據、計算技術指標（如 RSI、SMA、ATR）以及生成特徵和標籤。
- **主要函數**：
  - `download_stock_data`：下載指定股票的歷史數據。
  - `engineer_features`：生成技術指標作為模型的輸入特徵。
  - `create_labels`：生成交易標籤（買入、持有、賣出）。
  - `build_dataset`：構建訓練和測試數據集。

### 2. `baseline.py`
- **功能描述**：
  - 實現基線模型，使用隨機森林分類器進行股票趨勢預測。
- **主要類別**：
  - `BaselineModel`：基於 `RandomForestClassifier` 的基線模型，提供訓練和預測功能。

### 3. `final_report.py`
- **功能描述**：
  - 實現 Prototypical Network 模型，用於少樣本學習場景。
- **主要類別**：
  - `ProtoEncoder`：嵌入層，用於將輸入特徵轉換為嵌入向量。
  - `ProtoNet`：少樣本學習模型，通過計算樣本與原型的距離進行分類。
  - `ProtoNetModel`：封裝 ProtoNet 的高層接口，提供訓練和預測功能。

### 4. `run_experiments.py`
- **功能描述**：
  - 運行實驗以評估基線模型和 ProtoNet 模型的性能。
- **實驗設計**：
  - **E0**：基線模型（隨機森林分類器）。
  - **E1**：ProtoNet 模型（無少樣本訓練）。
  - **E2**：ProtoNet 模型（少樣本訓練）。
- **評估指標**：
  - 分類指標：準確率、宏平均 F1 分數、每類召回率、混淆矩陣。
  - 回測指標：總收益率、最大回撤、交易次數。
- **輸出**：
  - `results.csv`：保存實驗結果。
  - `results_summary.md`：生成實驗結果摘要報告。

## 使用說明

### 環境配置
1. 安裝必要的 Python 套件：
   ```bash
   pip install -r requirements.txt
   ```

2. 確保網絡連接正常，能夠訪問 Yahoo Finance API。

### 運行實驗
1. 執行以下命令運行實驗：
   ```bash
   python run_experiments.py
   ```
2. 結果將保存到 `results.csv` 和 `results_summary.md`。

## 文件結構
```
final_project/
├── data_pipeline.py       # 數據處理模組
├── baseline.py            # 基線模型
├── final_report.py        # Prototypical Network 模型
├── run_experiments.py     # 實驗運行腳本
├── requirements.txt       # 依賴包列表
├── results.csv            # 實驗結果（運行後生成）
├── results_summary.md     # 實驗摘要（運行後生成）
```
