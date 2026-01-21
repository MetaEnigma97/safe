import datamol as dm
from safe.dap import SafeReinventOptimizer
from reward import RewardCalculator
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
calc = RewardCalculator(
    {"homo": (-20, -8.5), "ox": (9, 20) }, 
    # substructure=['C(F)(F)F'], 
    match_mode="all")

# 2. 设置参数
model_path = "datamol-io/safe-gpt" # 替换为实际路径
target_scaffold = "c1ccccc1"

# 3. 初始化优化器
optimizer = SafeReinventOptimizer(
    model_path="/AI4S/Users/jwli/safe_gpt/models/model_sol_withoutB/checkpoint-1924776",
    lr=1e-5,               # 用户要求的学习率
    batch_size=256,       # 用户要求的 Batch Size
    max_length=80,         # 用户要求的 Max Length
    sigma=50.0,            # 强化学习系数
    bucket_size=100         # 多样性过滤阈值
)

# 4. 运行优化
optimizer.run(
    mode="random",           # 模式: scaffold decoration
    score_fn=calc.calculate,   # 评分函数
    save_path="/AI4S/Users/jwli/safe_gpt/models/dap-model",
    input_data=None,
    epochs=50,
    save_freq=5
)
