import os.path
import time
import pandas as pd

from IDV_CS_Model import *
import sys
import json

# Get absolute path to data directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'Evaluation_CoTs', 'Final_results')
# DF_NAME = 'GSM8K'
# DIFFICULTY = 'easy'
# NUM_OF_SAMPLES = 500
# NUM_OF_COT = 40
# MODEL = 'gpt-3.5-turbo-0125'


def normalize_cs(cs_li, threshold):
    cs_arr = np.array(cs_li)
    normalized_cs = [(cs-threshold)/(1-threshold) if cs > threshold else (cs-threshold)/(threshold) for cs in cs_arr]
    # normalized_cs = cs_arr - threshold
    # normalized_cs = [0.5 if cs > threshold else -0.5 for cs in cs_arr]
    return np.array(normalized_cs)


def stop_con1(individual_cs):
    cumulative_difference = (individual_cs).cumsum()
    stop_idx = np.argmax(cumulative_difference > 0.5)
    return stop_idx


def stop_con2(individual_cs, buffer_size=5):
    buffer = []
    for idx, cs in enumerate(individual_cs):
        if cs > 0:
            buffer.append(idx)
        if len(buffer) == buffer_size:
            return buffer[-1]


def consecutive_scores_above_threshold(scores, answers, threshold, n):
    for i in range(len(scores) - n + 1):
        # Check if all the next 'n' scores are above the threshold
        if all(score > threshold for score in scores[i:i + n]):
            # Check if all answers in this range are the same
            if len(set(answers[i:i + n])) == 1:
                return True, i + n, answers[i]
    return False, len(scores), None  # If no consecutive scores found or answers differ


def CS_early_stopping(df, threshold, N=5, stop_mechanism='PositiveN'):
    CS_Answer = []
    CS_correctness = []
    CS_steps = []
    
    # Timing tracking
    inference_time = 0  # Would track LLM calls if we were making them
    non_inference_start = time.time()
    
    if stop_mechanism == 'PositiveN':
        for row_idx in range(len(df)):
            test_row = df.iloc[row_idx]
            individual_cs = normalize_cs(test_row['confidence_score'], threshold)
            # individual_cs = test_row['confidence_score'][warm_up_steps:] - threshold
            stop_idx = stop_con2(individual_cs, buffer_size=N)

            if stop_idx:
                num_of_steps = stop_idx + 1  # +1 to account for the 0-indexing
            else:
                num_of_steps = 40
            answers = test_row['CoT answers'][:num_of_steps]
            scores = individual_cs[:num_of_steps]
            weighted_votes = Counter()
            for answer, score in zip(answers, scores):
                if score > 0:
                    weighted_votes[answer] += score
            # Find the answer with the highest total score
            if (len(weighted_votes)) == 0:
                for answer, score in zip(answers, scores):
                    weighted_votes[answer] += score
            result = max(weighted_votes, key=weighted_votes.get)
            CS_Answer.append(result)
            CS_correctness.append(1 if result == str(test_row['correct answer']) else 0)
            CS_steps.append(num_of_steps)
    elif stop_mechanism == 'ConsistencyN':
        found_count = 0
        for idx, row in df.iterrows():
            confidence_scores = row['confidence_score']
            answers = row['CoT answers']
            found, num_of_steps, answer = consecutive_scores_above_threshold(confidence_scores, answers, threshold, N)
            if found:
                found_count += 1
                CS_Answer.append(answer)
                CS_correctness.append(1 if answer == str(row['correct answer']) else 0)
                CS_steps.append(num_of_steps)  # +1 to account for the 0-indexing
            else:
                CS_Answer.append(None)
                CS_correctness.append(row['SC_correctness'])
                CS_steps.append(num_of_steps)
    df['CS_Answer'] = CS_Answer
    df['CS_correctness'] = CS_correctness
    df['CS_steps'] = CS_steps
    
    # Calculate timing metrics
    non_inference_time = time.time() - non_inference_start
    total_time = inference_time + non_inference_time

    df_model_comp_dict = {
        'SC_ACC': df.SC_correctness.sum() / len(df),
        'ES_ACC': df.ES_correctness.sum() / len(df),
        'CS_ACC': df.CS_correctness.sum() / len(df),
        'SC_Avg_Steps': 40,
        'ES_Avg_Steps': df.ES_steps.mean(),
        'CS_Avg_Steps': df.CS_steps.mean(),
        'ASC_Avg_Steps': df.asc_steps.mean(),
        'ASC_ACC': df.asc_correctness.sum() / len(df),
        'Non_Inference_Time': non_inference_time,
        'Total_Time': total_time,
        'Avg_Time_Per_Sample': total_time / len(df)
    }
    # Print each metric

    # for key, val in df_model_comp_dict.items():
    #     print(f"{key} : {val}")

    return df, df_model_comp_dict


if __name__ == '__main__':
    # file_path = os.path.join(DATA_DIR, 'final_ASC.csv')
    # df_raw = pd.read_csv(file_path)
    # df_with_features = pd.DataFrame(extract_feature(df_raw))
    # df_with_features.to_json(os.path.join(DATA_DIR,'final_ASC_with_feature.json'))
    
    # Start total timing
    script_start_time = time.time()
    
    df_with_features = pd.read_json(os.path.join(DATA_DIR,'final_with_feature_subset.json'))
    
    # Multi-model support: filter by model if specified via 5th argument
    model_filter = sys.argv[5] if len(sys.argv) > 5 else None
    if model_filter and 'model' in df_with_features.columns:
        df_with_features = df_with_features[df_with_features['model'] == model_filter]
        print(f"Filtered to model: {model_filter} ({len(df_with_features)} samples)")
    
    # Feature ablation: allow specifying feature subset via 4th argument
    # Default: all features (combined)
    # Options: 'answer', 'reasoning', 'combined'
    feature_type = sys.argv[4] if len(sys.argv) > 4 else 'combined'
    
    if feature_type == 'answer':
        # Answer-level features only
        feature_li = ['SIM_AC_BIGRAM', 'SIM_AC_AGG', 'SIM_AC_PW']
        coe = [2, 1, 3]  # Corresponding coefficients
        intercept = -2.5
    elif feature_type == 'reasoning':
        # Reasoning-level features only
        feature_li = ['QUA_IM', 'DIF_IV', 'SIM_COT_AGG']
        coe = [-5, -5, 3]  # Corresponding coefficients
        intercept = -2.5
    else:  # combined (default)
        # All features
        feature_li = [
            'QUA_IM',
            'DIF_IV',
            'SIM_COT_AGG',
            'SIM_AC_BIGRAM',
            'SIM_AC_AGG',
            'SIM_AC_PW',
        ]
        coe = [-5, -5, 3, 2, 1, 3]
        intercept = -2.5
    # Logistic regression coefficients from paper's trained model
    # Trained on full GSM8K dataset with 6 features:
    # [QUA_IM, DIF_IV, SIM_COT_AGG, SIM_AC_BIGRAM, SIM_AC_AGG, SIM_AC_PW]
    df_cs, _ = customized_LR_model(df=df_with_features, feature_li=feature_li, coe=coe, intercept=intercept)
    N = int(sys.argv[2])
    # N = 3
    threshold = float(sys.argv[1])
    # df_cs, _ = trained_LR_model(df=df_with_features, feature_li=feature_li)
    stop_mechanism = str(sys.argv[3])
    df, metrics = CS_early_stopping(df=df_cs, threshold=threshold, N=N, stop_mechanism=stop_mechanism)
    
    # Calculate total script time
    total_script_time = time.time() - script_start_time
    
    # Include feature type and model in filename if specified
    feature_suffix = f"_features_{feature_type}" if feature_type != 'combined' else ""
    model_suffix = f"_model_{model_filter}" if model_filter else ""
    file_name = f"df_threshold_{threshold}_N_{N}_stop_{stop_mechanism}{feature_suffix}{model_suffix}.csv"
    storage_dir = os.path.join(PROJECT_ROOT, 'result', 'experiments_output')
    os.makedirs(storage_dir, exist_ok=True)
    df.to_csv(os.path.join(storage_dir, file_name), index=False)
    
    # Save timing metrics
    metrics['Total_Script_Time'] = total_script_time
    metrics_file = file_name.replace('.csv', '_metrics.json')
    with open(os.path.join(storage_dir, metrics_file), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nTiming Metrics:")
    print(f"  Non-inference time: {metrics['Non_Inference_Time']:.4f}s")
    print(f"  Total time: {metrics['Total_Script_Time']:.4f}s")
    print(f"  Avg time per sample: {metrics['Avg_Time_Per_Sample']:.4f}s")
