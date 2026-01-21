from safe.augment_memory import SafeAugmentedOptimizer
import datamol as dm
from reward import RewardCalculator
import os

# Disable parallel tokenization if it causes issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup Reward
calc = RewardCalculator(
    {"homo": (-20, -8.5), "ox": (9, 20) }, 
    # substructure=['C(F)(F)F'], 
    match_mode="all"
)

# Initialize Optimizer
optimizer = SafeAugmentedOptimizer(
    model_path="/AI4S/Users/jwli/safe_gpt/models/model_sol_withoutB/checkpoint-1924776", 
    # tokenizer_path="/AI4S/Users/jwli/safe_gpt/models/model_sol_withoutB/checkpoint-1924776/tokenizer.json",
    lr=5e-5,
    memory_size=100,      # Size of the memory buffer
    entropy_weight=0.05,  # Regularization to prevent collapse
    exploration_prob=0.2, # Probability of using the frozen original model
    augmentation_rounds=1, # NEW: Number of randomized variants per memory item to generate (Data Augmentation)
    batch_size=1024
)

# Run Optimization
optimizer.run(
    mode="random",           # Mode: random, scaffold, motif, etc.
    input_data=None,         # Input structure if required by mode
    score_fn=calc.calculate, # Reward function
    save_path="/AI4S/Users/jwli/safe_gpt/models/ag-model",
    target_score=2,          # Score threshold for "perfect" logging
    epochs=50,               # Total iterations
    save_freq=10,            # Checkpoint frequency
    ft_epochs_per_step=2,    # Fine-tuning epochs per step
    train_batch_size=512     # Batch size for fine-tuning
)
