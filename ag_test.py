import os
import argparse
import datamol as dm
from eletrolyte_filter import check_sol_molecule, check_add_molecule
# Ensure TOKENIZERS_PARALLELISM off to avoid warnings from HF tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Local import (assumes safe/augment_memory.py 已在仓库中)
from safe.augment_memory import SafeAugmentedOptimizer

# RewardCalculator 在仓库根目录的 reward.py 中
try:
    from reward import RewardCalculator
except Exception:
    # 如果你的 reward.py 在另一路径，请调整 import
    from .reward import RewardCalculator  # attempt relative

def parse_args():
    p = argparse.ArgumentParser(description="Run Safe Augmented Memory test")
    p.add_argument("--model-path", type=str, default="/AI4S/Users/jwli/safe_gpt/models/model_sol_withoutB/checkpoint-1924776", help="路径到 SAFE 模型目录")
    p.add_argument("--out", type=str, default="/AI4S/Users/jwli/safe_gpt/models/ag-model", help="保存路径")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--train-batch-size", type=int, default=32)
    p.add_argument("--memory-size", type=int, default=200)
    p.add_argument("--augmentation-rounds", type=int, default=2)
    p.add_argument("--exploration-prob", type=float, default=0.2)
    p.add_argument("--entropy-weight", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--target-score", type=float, default=2.0)
    p.add_argument("--save-freq", type=int, default=10)
    p.add_argument("--ft-epochs-per-step", type=int, default=1)
    p.add_argument("--mode", choices=["random","scaffold","motif","linker","substructure"], default="random")
    p.add_argument("--mol-type", choices=["sol","add"], default="sol", help="生成分子类型，决定使用哪个检查函数 (check_sol_molecule / check_add_molecule)")
    p.add_argument("--invalid-score-penalty", type=float, default=0.75, help="当 checker 返回 False 时对 score 做惩罚")
    p.add_argument("--memory-prune-strategy", choices=["keep_top_score","age_based","diversity","novelty_replace"], default="novelty_replace")
    p.add_argument("--tanimoto-threshold", type=float, default=0.8, help="for diversity pruning: threshold to consider two molecules similar")
    p.add_argument("--novelty-threshold", type=float, default=0.6, help="for novelty_replace: max sim allowed for candidate to be considered novel")
    p.add_argument("--min-score-delta", type=float, default=-0.1, help="minimal score improvement required to evict lowest-scoring memory item")
    return p.parse_args()

def main():
    args = parse_args()
    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    calc = RewardCalculator({"homo": (-20, -8.5), "ox": (9, 20)}, match_mode="all")

    # 尝试获取 checker 函数
    checker = None
    if args.mol_type == "sol":
        checker = check_sol_molecule
        if checker is None:
            print("[ag_test] Warning: RewardCalculator has no method check_sol_molecule; no checking will be applied for 'sol'.")
    else:
        checker = check_add_molecule
        if checker is None:
            print("[ag_test] Warning: RewardCalculator has no method check_add_molecule; no checking will be applied for 'add'.")

    optimizer = SafeAugmentedOptimizer(
        model_path=args.model_path,
        lr=args.lr,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        entropy_weight=args.entropy_weight,
        exploration_prob=args.exploration_prob,
        augmentation_rounds=args.augmentation_rounds,
    )

    optimizer.run(
        mode=args.mode,
        score_fn=calc.calculate,
        save_path=args.out,
        input_data=None,
        target_score=args.target_score,
        epochs=args.epochs,
        save_freq=args.save_freq,
        ft_epochs_per_step=args.ft_epochs_per_step,
        train_batch_size=args.train_batch_size,
        gen_n_trials=1,
        gen_n_samples=None,
        mol_type=args.mol_type,
        checker=checker,
        invalid_score_penalty=args.invalid_score_penalty,
        memory_prune_strategy=args.memory_prune_strategy,
        tanimoto_threshold=args.tanimoto_threshold,
        novelty_threshold=args.novelty_threshold,
        min_score_delta=args.min_score_delta
    )

if __name__ == "__main__":
    main()
