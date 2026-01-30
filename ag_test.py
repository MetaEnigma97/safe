#!/usr/bin/env python3
import os
import argparse
import datamol as dm

# electrolyte checker functions (用户仓库可能已放在工程根或别处)
try:
    from eletrolyte_filter import check_sol_molecule, check_add_molecule
except Exception:
    # 如果模块不在 PYTHONPATH 中，先定义占位函数避免崩溃（会在运行时打印 warning）
    def check_sol_molecule(smi: str) -> bool:
        return True

    def check_add_molecule(smi: str) -> bool:
        return True

# Ensure TOKENIZERS_PARALLELISM off to avoid warnings from HF tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Local import (assumes safe/augment_memory.py 已在仓库中)
from safe.augment_memory import SafeAugmentedOptimizer

# RewardCalculator 在仓库根目录的 reward.py 中（允许相对导入）
try:
    from reward import RewardCalculator  # top-level
except Exception:
    try:
        from .reward import RewardCalculator  # relative
    except Exception:
        # placeholder: 若没有 reward.py，定义一个最简单的打分函数，便于做 smoke-test
        class RewardCalculator:
            def __init__(self, *args, **kwargs):
                pass

            def calculate(self, smiles_list):
                # return a dict-like mapping of individual components (match original expectation)
                # 这里返回单个 component "score" -> list of zeros
                return {"score": [0.0 for _ in smiles_list]}


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
    # new options
    p.add_argument("--unique-sampling", type=bool, default=True, help="是否在生成后进行去重采样，默认 False")
    p.add_argument("--double-loop-augment",type=bool, default=True, help="是否启用 double-loop augmentation（训练阶段对随机化序列再训练），默认 False")
    p.add_argument("--export-memory-path", type=str, default=None, help="如果指定，训练结束后导出 memory csv 到该路径（文件夹或带文件名）")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    # example RewardCalculator instantiation; adapt to你的具体打分逻辑和参数
    calc = RewardCalculator({"homo": (-20, -8.5), "ox": (9, 20)}, match_mode="all") if hasattr(RewardCalculator, "__init__") else RewardCalculator()

    # pick checker 函数
    checker = check_sol_molecule if args.mol_type == "sol" else check_add_molecule
    if checker is None:
        print("[ag_test] Warning: no checker function available; proceeding without extra validation.")

    optimizer = SafeAugmentedOptimizer(
        model_path=args.model_path,
        lr=args.lr,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        entropy_weight=args.entropy_weight,
        exploration_prob=args.exploration_prob,
        augmentation_rounds=args.augmentation_rounds,
        unique_sampling=args.unique_sampling,
        double_loop_augment=args.double_loop_augment,
    )

    optimizer.run(
        mode=args.mode,
        score_fn=calc.calculate if hasattr(calc, "calculate") else calc,
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

    if args.export_memory_path:
        # export memory snapshot
        out_path = args.export_memory_path
        try:
            optimizer.export_memory(out_path)
            print(f"[ag_test] memory exported to {out_path}")
        except Exception as e:
            print(f"[ag_test] export memory failed: {e}")


if __name__ == "__main__":
    main()
