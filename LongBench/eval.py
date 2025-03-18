import os
import json
import argparse
import numpy as np

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="gov_report")
    parser.add_argument('--model_name', type=str, default="Qwen1.5-MoE-A2.7B-Chat")
    parser.add_argument('--method', type=str, default="baseline")
    parser.add_argument('--topk', type=int, default="Qwen1.5-MoE-A2.7B-Chat")
    return parser.parse_args(args)



def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

if __name__ == '__main__':
    args = parse_args()
    
    scores = dict()
    scores_e = dict()

    pred_file = f"pred_e_v1/{args.model_name}_topk_{args.topk}_method_{args.method}/{args.dataset}.jsonl"

    predictions, answers, lengths = [], [], []
    
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            predictions.append(data["pred"])
            answers.append(data["answers"])
            all_classes = data["all_classes"]
            if "length" in data:
                lengths.append(data["length"])
    
    # score_e = scorer_e(args.dataset, predictions, answers, lengths, all_classes)
    score = scorer(args.dataset, predictions, answers, all_classes)
    
    list_1 = ['dataset',    'model_name',    'topk',    'method',   'score']
    list_2 = [args.dataset, args.model_name, args.topk, args.method, score]

    if not os.path.exists("all_results"):
        os.makedirs("all_results")

    csv_file = f"all_results/{args.model_name}_results.csv" 

    assert len(list_1) == len(list_2)
    file_exists = os.path.exists(csv_file)
    
    import csv
    with open(csv_file, 'a', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        
        if not file_exists:
            writer.writerow(list_1)
        
        writer.writerow(list_2) 
        
    csvfile.close()

