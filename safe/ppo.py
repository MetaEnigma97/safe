"""
Reinforcement Learning PPO Trainer for SAFE models.

This module implements a Proximal Policy Optimization (PPO) trainer to guide
different sampling methods defined in safe.sample.

Example:
    >>> import safe as sf
    >>> from safe.ppo import SafePPOTrainer
    >>>
    >>> # Define a simple reward function (e.g., molecular weight penalty)
    >>> def mw_reward(smiles):
    ...     mol = dm.to_mol(smiles)
    ...     if mol is None:
    ...         return 0.0
    ...     mw = dm.descriptors.mw(mol)
    ...     # Reward molecules with MW between 300-500
    ...     if 300 <= mw <= 500:
    ...         return 1.0
    ...     else:
    ...         return -abs(mw - 400) / 400
    >>>
    >>> # Initialize trainer
    >>> trainer = SafePPOTrainer(
    ...     model="path/to/safe/model",
    ...     tokenizer="path/to/tokenizer",
    ... )
    >>>
    >>> # Define training data (scaffolds for scaffold decoration)
    >>> train_data = ["c1ccccc1", "C1CCCCC1", "c1ccncc1"]
    >>>
    >>> # Train the model
    >>> trainer.train(
    ...     train_data=train_data,
    ...     generation_method="scaffold_decoration",
    ...     score_functions=[mw_reward],
    ...     model_save_path="./ppo_trained_model",
    ...     epochs=50,
    ...     batch_size=16,
    ...     ppo_config={"lr": 1e-5, "kl_coef": 0.1},
    ... )
"""

import copy
import os
import random
from typing import List, Optional, Union, Dict, Callable, Any
from pathlib import Path

import torch
import torch.nn.functional as F
import datamol as dm
from loguru import logger
from tqdm.auto import tqdm

import safe as sf
from safe.tokenizer import SAFETokenizer
from safe.trainer.model import SAFEDoubleHeadsModel


class SafePPOTrainer:
    """
    PPO Trainer for SAFE models to guide molecular generation.

    This trainer uses Proximal Policy Optimization to fine-tune a SAFE model
    based on reward functions, supporting various generation strategies.

    The trainer maintains two models:
    - A trainable policy model that is updated during training
    - A frozen reference model used to compute KL divergence penalty

    Supported generation methods:
    - de_novo_generation: Generate molecules from scratch
    - motif_extension: Extend molecular motifs
    - linker_generation: Generate linkers between two fragments
    - scaffold_decoration: Decorate molecular scaffolds
    - super_structure: Generate super structures from molecular cores

    Attributes:
        model: The trainable SAFE model
        tokenizer: The SAFE tokenizer
        ref_model: The frozen reference model for KL penalty
        device: Device for computation (cuda/cpu)
        designer: SAFEDesign instance for molecule generation
    """

    def __init__(
        self,
        model: Union[SAFEDoubleHeadsModel, str, os.PathLike],
        tokenizer: Union[SAFETokenizer, str, os.PathLike],
        ref_model: Optional[Union[SAFEDoubleHeadsModel, str, os.PathLike]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the SafePPOTrainer.

        Args:
            model: The SAFE model to train (or path to it)
            tokenizer: The tokenizer (or path to it)
            ref_model: Reference model for KL penalty (or path to it). If None, uses a copy of the model.
            device: Device to use for training ('cuda' or 'cpu')
        """
        # Load model and store the original path if provided
        model_path = model if isinstance(model, (str, os.PathLike)) else None
        if isinstance(model, (str, os.PathLike)):
            model = SAFEDoubleHeadsModel.from_pretrained(model)
        self.model = model

        # Load tokenizer
        if isinstance(tokenizer, (str, os.PathLike)):
            tokenizer = SAFETokenizer.load(tokenizer)
        self.tokenizer = tokenizer

        # Setup device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

        # Load or create reference model for KL penalty
        if ref_model is None:
            # Create a copy of the model as reference
            if model_path is not None:
                ref_model = SAFEDoubleHeadsModel.from_pretrained(model_path)
            else:
                # Deep copy the model
                ref_model = copy.deepcopy(self.model)
        elif isinstance(ref_model, (str, os.PathLike)):
            ref_model = SAFEDoubleHeadsModel.from_pretrained(ref_model)

        self.ref_model = ref_model
        self.ref_model.to(self.device)
        self.ref_model.eval()

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Create SAFEDesign instance for generation
        self.designer = sf.SAFEDesign(
            model=self.model,
            tokenizer=self.tokenizer,
            verbose=False,
        )

    def train(
        self,
        train_data: List[Any],
        generation_method: str,
        score_functions: List[Callable[[str], float]],
        model_save_path: Union[str, os.PathLike],
        save_freq_epoch: int = 10,
        epochs: int = 100,
        batch_size: int = 32,
        ppo_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Train the model using PPO.

        Args:
            train_data: List of input data appropriate for the generation method
                - For de_novo_generation: empty list or None
                - For motif_extension/scaffold_decoration: list of scaffolds (SMILES or Mol)
                - For linker_generation: list of tuples of (fragment1, fragment2)
                - For super_structure: list of cores (SMILES or Mol)
            generation_method: The generation method to use. One of:
                - 'de_novo_generation': Random generation
                - 'motif_extension': Scaffold decoration
                - 'linker_generation': Linker generation
                - 'scaffold_decoration': Scaffold decoration
                - 'super_structure': Substructure generation
            score_functions: List of scoring functions that take a SMILES string and return a scalar reward
            model_save_path: Path to save the fine-tuned model
            save_freq_epoch: Number of epochs between model saves
            epochs: Total number of training epochs
            batch_size: Batch size for training
            ppo_config: Dictionary of PPO hyperparameters:
                - lr: Learning rate (default: 1e-5)
                - clip_range: PPO clip range (default: 0.2)
                - kl_coef: KL penalty coefficient (default: 0.1)
                - gamma: Discount factor (default: 1.0)
                - n_samples: Number of samples to generate per input (default: 4)
        """
        # Set default PPO config
        default_config = {
            'lr': 1e-5,
            'clip_range': 0.2,
            'kl_coef': 0.1,
            'gamma': 1.0,
            'n_samples': 4,
        }
        if ppo_config is not None:
            default_config.update(ppo_config)
        ppo_config = default_config

        # Setup optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=ppo_config['lr'])

        # Validate generation method
        supported_methods = [
            'de_novo_generation',
            'motif_extension',
            'linker_generation',
            'scaffold_decoration',
            'super_structure',
        ]
        if generation_method not in supported_methods:
            raise ValueError(
                f"Unsupported generation method: {generation_method}. "
                f"Supported methods: {supported_methods}"
            )

        # Prepare train_data based on generation method
        if generation_method == 'de_novo_generation':
            train_data = [None] * batch_size  # Placeholder for de novo
        elif not train_data:
            raise ValueError(f"train_data cannot be empty for {generation_method}")

        model_save_path = Path(model_save_path)
        model_save_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting PPO training for {epochs} epochs")
        logger.info(f"Generation method: {generation_method}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"PPO config: {ppo_config}")

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            epoch_losses = []
            epoch_rewards = []

            # Sample batches from train_data
            # Use random.choices with replacement if batch_size > len(train_data)
            if batch_size >= len(train_data):
                batch_indices = random.choices(range(len(train_data)), k=batch_size)
            else:
                batch_indices = random.sample(range(len(train_data)), batch_size)
            batch_data = [train_data[i] for i in batch_indices]

            for data_item in tqdm(batch_data, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                # Generate molecules using the current policy
                generated_mols, log_probs, entropies = self._generate_with_log_probs(
                    data_item,
                    generation_method,
                    n_samples=ppo_config['n_samples'],
                )

                if not generated_mols:
                    continue

                # Calculate rewards
                rewards = self._calculate_rewards(generated_mols, score_functions)

                # Calculate KL divergence with reference model
                kl_div = self._calculate_kl_divergence(
                    data_item,
                    generation_method,
                    generated_mols,
                )

                # Compute PPO loss
                loss = self._compute_ppo_loss(
                    log_probs,
                    rewards,
                    kl_div,
                    ppo_config,
                )

                # Update model
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_losses.append(loss.item())
                epoch_rewards.append(torch.mean(rewards).item())

            # Log epoch statistics
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Avg Loss: {avg_loss:.4f}, "
                f"Avg Reward: {avg_reward:.4f}"
            )

            # Save model
            if (epoch + 1) % save_freq_epoch == 0:
                save_path = model_save_path / f"checkpoint_epoch_{epoch+1}"
                self._save_model(save_path)
                logger.info(f"Model saved to {save_path}")

        # Save final model
        final_save_path = model_save_path / "final_model"
        self._save_model(final_save_path)
        logger.info(f"Training complete. Final model saved to {final_save_path}")

    def _generate_with_log_probs(
        self,
        data_item: Any,
        generation_method: str,
        n_samples: int = 4,
    ):
        """
        Generate molecules and compute log probabilities.

        Args:
            data_item: Input data for generation
            generation_method: Generation method to use
            n_samples: Number of samples to generate

        Returns:
            Tuple of (generated_molecules, log_probs, entropies)
        """
        self.model.eval()

        with torch.no_grad():
            # Generate molecules using the designer
            if generation_method == 'de_novo_generation':
                generated = self.designer.de_novo_generation(
                    n_samples_per_trial=n_samples,
                    sanitize=True,
                )
            elif generation_method == 'motif_extension':
                generated = self.designer.motif_extension(
                    motif=data_item,
                    n_samples_per_trial=n_samples,
                    sanitize=True,
                )
            elif generation_method == 'scaffold_decoration':
                generated = self.designer.scaffold_decoration(
                    scaffold=data_item,
                    n_samples_per_trial=n_samples,
                    sanitize=True,
                )
            elif generation_method == 'linker_generation':
                if not isinstance(data_item, (list, tuple)) or len(data_item) != 2:
                    raise ValueError("linker_generation requires a tuple/list of 2 fragments")
                generated = self.designer.linker_generation(
                    *data_item,
                    n_samples_per_trial=n_samples,
                    sanitize=True,
                )
            elif generation_method == 'super_structure':
                generated = self.designer.super_structure(
                    core=data_item,
                    n_samples_per_trial=n_samples,
                    sanitize=True,
                )
            else:
                raise ValueError(f"Unsupported generation method: {generation_method}")

        # Filter out None values
        generated = [mol for mol in generated if mol is not None]

        if not generated:
            return [], torch.tensor([]), torch.tensor([])

        # Compute log probabilities for generated molecules
        log_probs_list = []
        entropies_list = []

        # Set model to training mode once before the loop
        self.model.train()

        for mol_smiles in generated:
            try:
                # Encode the molecule
                safe_str = sf.encode(mol_smiles, canonical=True)

                # Tokenize
                inputs = self.tokenizer.get_pretrained()(
                    safe_str,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                ).to(self.device)

                # Get model outputs
                with torch.set_grad_enabled(True):
                    outputs = self.model(**inputs, labels=inputs['input_ids'])
                    logits = outputs.logits

                    # Calculate log probabilities
                    log_probs = F.log_softmax(logits, dim=-1)

                    # Get log probs for actual tokens
                    token_log_probs = log_probs[:, :-1, :].gather(
                        2, inputs['input_ids'][:, 1:].unsqueeze(-1)
                    ).squeeze(-1)

                    # Sum log probs
                    total_log_prob = token_log_probs.sum()
                    log_probs_list.append(total_log_prob)

                    # Calculate entropy (measure of uncertainty)
                    probs = F.softmax(logits, dim=-1)
                    entropy = -(probs * log_probs).sum(dim=-1).mean()
                    entropies_list.append(entropy)

            except Exception as e:
                logger.warning(f"Failed to compute log probs for {mol_smiles}: {e}")
                continue

        if not log_probs_list:
            return generated, torch.tensor([]), torch.tensor([])

        log_probs = torch.stack(log_probs_list)
        entropies = torch.stack(entropies_list)

        return generated, log_probs, entropies

    def _calculate_rewards(
        self,
        molecules: List[str],
        score_functions: List[Callable[[str], float]],
    ) -> torch.Tensor:
        """
        Calculate rewards for generated molecules.

        Args:
            molecules: List of SMILES strings
            score_functions: List of scoring functions

        Returns:
            Tensor of rewards
        """
        rewards = []

        for mol_smiles in molecules:
            mol_reward = 0.0

            for score_fn in score_functions:
                try:
                    score = score_fn(mol_smiles)
                    mol_reward += score
                except Exception as e:
                    logger.warning(f"Scoring function failed for {mol_smiles}: {e}")
                    mol_reward += 0.0

            rewards.append(mol_reward)

        return torch.tensor(rewards, dtype=torch.float32, device=self.device)

    def _calculate_kl_divergence(
        self,
        data_item: Any,
        generation_method: str,
        molecules: List[str],
    ) -> torch.Tensor:
        """
        Calculate KL divergence between current policy and reference policy.

        Args:
            data_item: Input data for generation
            generation_method: Generation method used
            molecules: Generated molecules

        Returns:
            KL divergence tensor
        """
        kl_divs = []

        for mol_smiles in molecules:
            try:
                # Encode the molecule
                safe_str = sf.encode(mol_smiles, canonical=True)

                # Tokenize
                inputs = self.tokenizer.get_pretrained()(
                    safe_str,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                ).to(self.device)

                # Get current model logits (with gradients for backprop)
                current_outputs = self.model(**inputs)
                current_logits = current_outputs.logits

                # Get reference model logits (no gradients needed)
                with torch.no_grad():
                    ref_outputs = self.ref_model(**inputs)
                    ref_logits = ref_outputs.logits

                # Calculate KL divergence
                current_log_probs = F.log_softmax(current_logits, dim=-1)
                ref_probs = F.softmax(ref_logits, dim=-1)

                kl_div = F.kl_div(
                    current_log_probs,
                    ref_probs,
                    reduction='batchmean',
                    log_target=False,
                )
                kl_divs.append(kl_div)

            except Exception as e:
                logger.warning(f"Failed to compute KL divergence for {mol_smiles}: {e}")
                kl_divs.append(torch.tensor(0.0, device=self.device))

        if not kl_divs:
            return torch.tensor(0.0, device=self.device)

        return torch.stack(kl_divs).mean()

    def _compute_ppo_loss(
        self,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        kl_div: torch.Tensor,
        ppo_config: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Compute PPO loss.

        Args:
            log_probs: Log probabilities of actions
            rewards: Rewards for generated molecules
            kl_div: KL divergence with reference policy
            ppo_config: PPO configuration

        Returns:
            PPO loss tensor
        """
        if log_probs.numel() == 0:
            # Return a scalar tensor that participates in the computational graph
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Normalize rewards
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Policy gradient loss (negative because we want to maximize reward)
        pg_loss = -(log_probs * rewards).mean()

        # KL penalty
        kl_penalty = ppo_config['kl_coef'] * kl_div

        # Total loss
        return pg_loss + kl_penalty


    def _save_model(self, save_path: Union[str, os.PathLike]):
        """
        Save the model and tokenizer.

        Args:
            save_path: Path to save the model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(save_path)

        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)

        logger.info(f"Model and tokenizer saved to {save_path}")


# Alias for backward compatibility
SafePPO = SafePPOTrainer
