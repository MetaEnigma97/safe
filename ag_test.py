from safe.augment_memory import SafeAugmentedOptimizer
import datamol as dm
from reward import RewardCalculator
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
calc = RewardCalculator(
    {"homo": (-20, -8.5), "ox": (9, 20) }, 
    # substructure=['C(F)(F)F'], 
    match_mode="all")

optimizer = SafeAugmentedOptimizer(
    model_path="/AI4S/Users/jwli/safe_gpt/models/model_sol_withoutB/checkpoint-1924776", 
    # tokenizer_path="/AI4S/Users/jwli/safe_gpt/models/model_sol_withoutB/checkpoint-1924776/tokenizer.json",
    lr=5e-5,
    memory_size=100, # 记忆库大小
    entropy_weight=0.05,
    exploration_prob=0.2,
    batch_size=1024
)

optimizer.run(
    mode="random",           # 模式: scaffold decoration
    input_data=None, # 输入骨架
    score_fn=calc.calculate, # 两个评分函数相加
    save_path="/AI4S/Users/jwli/safe_gpt/models/ag-model",
    target_score=2,
    epochs=50,                 # 总共跑 20 轮
    save_freq=10,                # 每 5 轮保存一次
    ft_epochs_per_step=2,
    train_batch_size=512
)
