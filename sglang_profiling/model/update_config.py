
import argparse
import os
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--desired_topk', type=int, default=None)
    parser.add_argument('--desired_topk_key', type=str, default=None)
    parser.add_argument('--num_layer_key', type=str, default=None)
    args = parser.parse_args()

    with open(os.path.join(args.model_dir, "config.json"), 'r') as f:
        config = json.load(f)

    topk_dict = {}
    for layer_id in range(config[args.num_layer_key]):
        topk_dict[layer_id] = args.desired_topk

    config[args.desired_topk_key] = topk_dict

    with open(os.path.join(args.model_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=False)


    
    