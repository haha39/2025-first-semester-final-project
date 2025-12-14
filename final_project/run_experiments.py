import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from data_pipeline import build_dataset
from baseline import BaselineModel
from final_report import ProtoNetModel

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

    Results are saved to results.csv.
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
    cm = confusion_matrix(y_test, y_pred)
    total_return, max_drawdown, num_trades = backtest(y_pred, test_meta_df)

    results.append({
        'Experiment': 'E0 Baseline',
        'Accuracy': accuracy,
        'Macro F1': macro_f1,
        'Recall_0': recalls[0],
        'Recall_1': recalls[1],
        'Recall_2': recalls[2],
        'Confusion Matrix': cm.tolist(),
        'Total Return': total_return,
        'Max Drawdown': max_drawdown,
        'Num Trades': num_trades
    })

    # Experiment E1: ProtoNet (no few-shot)
    protonet_no_fs = ProtoNetModel(input_dim=X_train.shape[1], n_classes=3, use_few_shot=False)
    protonet_no_fs.fit(X_train, y_train)
    y_pred = protonet_no_fs.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    recalls = recall_score(y_test, y_pred, average=None)
    cm = confusion_matrix(y_test, y_pred)
    total_return, max_drawdown, num_trades = backtest(y_pred, test_meta_df)

    results.append({
        'Experiment': 'E1 ProtoNet (no few-shot)',
        'Accuracy': accuracy,
        'Macro F1': macro_f1,
        'Recall_0': recalls[0],
        'Recall_1': recalls[1],
        'Recall_2': recalls[2],
        'Confusion Matrix': cm.tolist(),
        'Total Return': total_return,
        'Max Drawdown': max_drawdown,
        'Num Trades': num_trades
    })

    # Experiment E2: ProtoNet (few-shot)
    protonet_fs = ProtoNetModel(input_dim=X_train.shape[1], n_classes=3, use_few_shot=True)
    protonet_fs.fit(X_train, y_train)
    y_pred = protonet_fs.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    recalls = recall_score(y_test, y_pred, average=None)
    cm = confusion_matrix(y_test, y_pred)
    total_return, max_drawdown, num_trades = backtest(y_pred, test_meta_df)

    results.append({
        'Experiment': 'E2 ProtoNet (few-shot)',
        'Accuracy': accuracy,
        'Macro F1': macro_f1,
        'Recall_0': recalls[0],
        'Recall_1': recalls[1],
        'Recall_2': recalls[2],
        'Confusion Matrix': cm.tolist(),
        'Total Return': total_return,
        'Max Drawdown': max_drawdown,
        'Num Trades': num_trades
    })

    # Save results
    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv('results.csv', index=False)

if __name__ == '__main__':
    run_experiments()