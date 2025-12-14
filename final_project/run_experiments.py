import pandas as pd
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from data_pipeline import build_dataset
from baseline import BaselineModel
from final_report import ProtoNetModel

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Simplified backtesting function
def backtest(predictions, test_meta_df, commission_rate=0.002):
    """
    Perform a simplified backtest based on predictions.

    Args:
        predictions: Array of predicted actions (0 = sell, 1 = hold, 2 = buy).
        test_meta_df: DataFrame containing metadata (Open and Close prices).
        commission_rate: Transaction cost rate (default 0.2%).

    Returns:
        tuple: (total return, maximum drawdown, number of trades).
    """
    capital = 1.0  # Initial capital
    max_drawdown = 0.0
    peak = capital
    num_trades = 0

    for i in range(len(predictions)):
        if i == len(predictions) - 1:
            break

        action = predictions[i]  # 0 = sell, 1 = hold, 2 = buy
        open_price = test_meta_df.iloc[i + 1]['Open']
        close_price = test_meta_df.iloc[i + 1]['Close']

        if action == 2:  # Buy
            capital *= (1 - commission_rate) * (close_price / open_price)
            num_trades += 1
        elif action == 0:  # Sell
            capital *= (1 - commission_rate) * (open_price / close_price)
            num_trades += 1

        peak = max(peak, capital)
        drawdown = (peak - capital) / peak
        max_drawdown = max(max_drawdown, drawdown)

    return capital - 1.0, max_drawdown, num_trades



# Experiment runner
def run_experiments():
    """
    Run experiments to evaluate Baseline and ProtoNet models.

    Experiments:
        E0: Baseline (RandomForestClassifier).
        E1: ProtoNet without few-shot training.
        E2: ProtoNet with few-shot training.

    Metrics:
        - Classification metrics: accuracy, macro F1, per-class recall, confusion matrix.
        - Backtesting metrics: total return, maximum drawdown, number of trades.

    Results are saved to results.csv and summarized in results_summary.md.
    """
    config = {
        'start_date': '2020-01-01',
        'end_date': '2024-12-31',
        'train_end': '2023-12-31'
    }

    # Load dataset
    X_train, y_train, X_test, y_test, test_meta_df = build_dataset('2330.TW', config)

    results = []

    # Experiment E0: Baseline
    baseline = BaselineModel()
    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)
    y_proba = baseline.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    recalls = recall_score(y_test, y_pred, average=None)
    total_return, max_drawdown, num_trades = backtest(y_pred, test_meta_df)

    results.append({
        'model_name': 'E0 Baseline',
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'recall_sell': recalls[0],
        'recall_buy': recalls[2],
        'total_return': total_return,
        'mdd': max_drawdown,
        'num_trades': num_trades
    })

    # Experiment E1: ProtoNet (no few-shot)
    protonet_no_fs = ProtoNetModel(input_dim=X_train.shape[1], n_classes=3, use_few_shot=False)
    protonet_no_fs.fit(X_train, y_train)
    y_pred = protonet_no_fs.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    recalls = recall_score(y_test, y_pred, average=None)
    total_return, max_drawdown, num_trades = backtest(y_pred, test_meta_df)

    results.append({
        'model_name': 'E1 ProtoNet (no few-shot)',
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'recall_sell': recalls[0],
        'recall_buy': recalls[2],
        'total_return': total_return,
        'mdd': max_drawdown,
        'num_trades': num_trades
    })

    # Experiment E2: ProtoNet (few-shot)
    protonet_fs = ProtoNetModel(input_dim=X_train.shape[1], n_classes=3, use_few_shot=True)
    protonet_fs.fit(X_train, y_train)
    y_pred = protonet_fs.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    recalls = recall_score(y_test, y_pred, average=None)
    total_return, max_drawdown, num_trades = backtest(y_pred, test_meta_df)

    results.append({
        'model_name': 'E2 ProtoNet (few-shot)',
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'recall_sell': recalls[0],
        'recall_buy': recalls[2],
        'total_return': total_return,
        'mdd': max_drawdown,
        'num_trades': num_trades
    })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('results.csv', index=False)

    # Generate summary report
    with open('results_summary.md', 'w') as f:
        f.write("# Experiment Results Summary\n\n")
        for result in results:
            f.write(f"## {result['model_name']}\n")
            f.write(f"- Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"- Macro F1: {result['macro_f1']:.4f}\n")
            f.write(f"- Recall (Sell): {result['recall_sell']:.4f}\n")
            f.write(f"- Recall (Buy): {result['recall_buy']:.4f}\n")
            f.write(f"- Total Return: {result['total_return']:.4f}\n")
            f.write(f"- Max Drawdown: {result['mdd']:.4f}\n")
            f.write(f"- Number of Trades: {result['num_trades']}\n\n")

    print("Results saved to results.csv and results_summary.md.")

if __name__ == '__main__':
    run_experiments()