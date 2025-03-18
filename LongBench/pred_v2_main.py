
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--model",    type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument("--rag",      type=int, default=0)
    parser.add_argument("--n_proc",   type=int, default=1)
    parser.add_argument("--topk",     type=int, default=4)

    parser.add_argument("--cot",        action='store_true')
    parser.add_argument("--no_context", action='store_true')

    args = parser.parse_args()

    if args.model == "deepseek-ai/DeepSeek-V2-Lite-Chat":
        from pred_models.deepseek.deepseek2_longbench_v2 import main_deepseekv2
        main_deepseekv2(args)

    elif args.model == "ai21labs/AI21-Jamba-Mini-1.6":
        from pred_models.jamba.jamba_longbench_v2 import main_jamba
        main_jamba(args)

    elif args.model == "mistralai/Mixtral-8x7B-Instruct-v0.1":
        from pred_models.mixtral.mixtral_longbench_v2 import main_mixtral
        main_mixtral(args)

    elif args.model == "microsoft/Phi-3.5-MoE-instruct":
        from pred_models.phi.phi3_longbench_v2 import main_phi3
        main_phi3(args)

    elif args.model == "Qwen/Qwen1.5-MoE-A2.7B-Chat":
        from pred_models.qwen.qwen_longbench_v2 import main_qwen2
        main_qwen2(args)

    


