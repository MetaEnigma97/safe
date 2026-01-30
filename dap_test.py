import datamol as dm
from safe.dap import SafeReinventOptimizer
from reward import RewardCalculator
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

calc = RewardCalculator(
    {"homo": (-20, -8.5), "ox": (9, 20) }, 
    match_mode="all"
)

# Initialize Optimizer
optimizer = SafeReinventOptimizer(
    model_path="/AI4S/Users/jwli/safe_gpt/models/model_add_with1B/checkpoint-7309428",
    lr=1e-4,
    batch_size=1024,
    max_length=80,
    sigma=5.0,
    bucket_size=5000,
    memory_size=200,
    penalty_factor=0.3 
)

# Run Optimization
optimizer.run(
    mode="random",
    score_fn=calc.calculate,
    save_path="/AI4S/Users/jwli/safe_gpt/models/dap-model",
    input_data=None,
    epochs=50,
    save_freq=5
)
