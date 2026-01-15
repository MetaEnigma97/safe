"""
Reinforcement Learning PPO Trainer for SAFE molecule generation.

This module implements a PPO (Proximal Policy Optimization) trainer that guides
different sampling methods defined in safe.sample to generate molecules with
desired properties.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tqdm.auto import tqdm

import safe as sf
from safe.tokenizer import SAFETokenizer
from safe.trainer.model import SAFEDoubleHeadsModel
from safe.sample import SAFEDesign


class SafePPOTrainer:
    """
    PPO Trainer for guiding SAFE molecule generation using reinforcement learning.
    
    This trainer uses Proximal Policy Optimization to fine-tune a SAFE model to generate
    molecules with desired properties, as measured by score functions.
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
            model: The SAFE model to train, or path to load from
            tokenizer: The tokenizer to use, or path to load from
            ref_model: Reference model for KL penalty (frozen), or path to load from.
                      If None, uses a copy of the initial model.
            device: Device to use ('cpu' or 'cuda'). If None, auto-detect.
        """
        # Load model
        if isinstance(model, (str, os.PathLike)):
            self.model = SAFEDoubleHeadsModel.from_pretrained(model)
        else:
            self.model = model
            
        # Load tokenizer
        if isinstance(tokenizer, (str, os.PathLike)):
            self.tokenizer = SAFETokenizer.load(tokenizer)
        else:
            self.tokenizer = tokenizer
            
        # Setup device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        
        # Load or create reference model (frozen)
        if ref_model is None:
            # Create a copy of the initial model as reference
            if isinstance(model, (str, os.PathLike)):
                self.ref_model = SAFEDoubleHeadsModel.from_pretrained(model)
            else:
                # Deep copy the model
                self.ref_model = SAFEDoubleHeadsModel.from_pretrained(
                    model.config._name_or_path if hasattr(model.config, '_name_or_path') else 'datamol-io/safe-gpt'
                )
        elif isinstance(ref_model, (str, os.PathLike)):
            self.ref_model = SAFEDoubleHeadsModel.from_pretrained(ref_model)
        else:
            self.ref_model = ref_model
            
        self.ref_model.to(self.device)
        # Freeze reference model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
        # Create designer for generation
        self.designer = SAFEDesign(
            model=self.model,
            tokenizer=self.tokenizer,
            verbose=False
        )
        
    def train(
        self,
        train_data: List[Any],
        generation_method: str,
        score_functions: Union[Callable, List[Callable]],
        model_save_path: Union[str, os.PathLike],
        save_freq_epoch: int = 1,
        epochs: int = 10,
        batch_size: int = 8,
        ppo_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Train the model using PPO.
        
        Args:
            train_data: Training data (context for generation methods)
            generation_method: One of: 'de_novo_generation', 'motif_extension',
                              'linker_generation', 'scaffold_decoration', 'super_structure'
            score_functions: Function(s) to score generated molecules. Should accept
                           a list of SMILES and return a list of scores.
            model_save_path: Path to save model checkpoints
            save_freq_epoch: Save model every N epochs
            epochs: Number of training epochs
            batch_size: Batch size for generation and training
            ppo_config: Configuration dict with keys:
                - lr (float): Learning rate, default 1e-5
                - kl_coef (float): KL divergence coefficient, default 0.1
                - gamma (float): Discount factor for rewards, default 1.0
                - clip_eps (float): PPO clipping epsilon, default 0.2
                - n_samples_per_trial (int): Samples per generation, default 10
                - n_trials (int): Number of trials per batch item, default 1
        """
        # Setup config
        config = {
            'lr': 1e-5,
            'kl_coef': 0.1,
            'gamma': 1.0,
            'clip_eps': 0.2,
            'n_samples_per_trial': 10,
            'n_trials': 1,
        }
        if ppo_config is not None:
            config.update(ppo_config)
            
        # Setup optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        
        # Ensure score_functions is a list
        if not isinstance(score_functions, list):
            score_functions = [score_functions]
            
        # Validate generation method
        valid_methods = [
            'de_novo_generation',
            'motif_extension', 
            'linker_generation',
            'scaffold_decoration',
            'super_structure'
        ]
        if generation_method not in valid_methods:
            raise ValueError(
                f"generation_method must be one of {valid_methods}, got {generation_method}"
            )
            
        logger.info(f"Starting PPO training for {epochs} epochs")
        logger.info(f"Generation method: {generation_method}")
        logger.info(f"Config: {config}")
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_reward = 0.0
            epoch_kl = 0.0
            n_batches = 0
            
            # Sample batches from train_data
            for batch_idx in tqdm(
                range(0, len(train_data), batch_size),
                desc=f"Epoch {epoch+1}/{epochs}",
                leave=False
            ):
                batch = train_data[batch_idx:batch_idx + batch_size]
                
                # Generate molecules and compute log probabilities
                generated_mols, log_probs, entropies = self._generate_with_log_probs(
                    batch=batch,
                    generation_method=generation_method,
                    n_samples_per_trial=config['n_samples_per_trial'],
                    n_trials=config['n_trials'],
                )
                
                if len(generated_mols) == 0:
                    logger.warning(f"No valid molecules generated in batch {batch_idx}")
                    continue
                    
                # Compute rewards
                rewards = self._compute_rewards(generated_mols, score_functions)
                
                # Normalize rewards
                normalized_rewards = self._normalize_rewards(rewards)
                
                # Compute KL divergence with reference model
                kl_div = self._compute_kl_divergence(generated_mols, log_probs)
                
                # Compute PPO loss
                loss = self._compute_ppo_loss(
                    log_probs=log_probs,
                    rewards=normalized_rewards,
                    kl_div=kl_div,
                    kl_coef=config['kl_coef'],
                )
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                epoch_reward += rewards.mean().item()
                epoch_kl += kl_div.item()
                n_batches += 1
                
            # Log epoch metrics
            if n_batches > 0:
                avg_loss = epoch_loss / n_batches
                avg_reward = epoch_reward / n_batches
                avg_kl = epoch_kl / n_batches
                
                logger.info(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"Loss={avg_loss:.4f}, Reward={avg_reward:.4f}, KL={avg_kl:.4f}"
                )
            
            # Save model
            if (epoch + 1) % save_freq_epoch == 0:
                self._save_model(model_save_path, epoch + 1)
                
        logger.info("Training complete!")
        
    def _generate_with_log_probs(
        self,
        batch: List[Any],
        generation_method: str,
        n_samples_per_trial: int = 10,
        n_trials: int = 1,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """
        Generate molecules and compute their log probabilities.
        
        This method ensures synchronization between generated molecules and their
        log probabilities by excluding molecules that fail encoding/probability calculation.
        
        Args:
            batch: Batch of contexts for generation
            generation_method: Generation method to use
            n_samples_per_trial: Number of samples per trial
            n_trials: Number of trials
            
        Returns:
            Tuple of (valid_molecules, log_probs, entropies) where all have the same length
        """
        all_valid_mols = []
        all_log_probs = []
        all_entropies = []
        
        for item in batch:
            # Generate molecules using the appropriate method
            if generation_method == 'de_novo_generation':
                generated = self.designer.de_novo_generation(
                    n_samples_per_trial=n_samples_per_trial,
                    n_trials=n_trials,
                    sanitize=True,
                )
            elif generation_method == 'motif_extension':
                generated = self.designer.motif_extension(
                    motif=item,
                    n_samples_per_trial=n_samples_per_trial,
                    n_trials=n_trials,
                    sanitize=True,
                )
            elif generation_method == 'linker_generation':
                generated = self.designer.linker_generation(
                    groups=item,
                    n_samples_per_trial=n_samples_per_trial,
                    n_trials=n_trials,
                    sanitize=True,
                )
            elif generation_method == 'scaffold_decoration':
                generated = self.designer.scaffold_decoration(
                    scaffold=item,
                    n_samples_per_trial=n_samples_per_trial,
                    n_trials=n_trials,
                    sanitize=True,
                )
            elif generation_method == 'super_structure':
                generated = self.designer.super_structure(
                    core=item,
                    n_samples_per_trial=n_samples_per_trial,
                    n_trials=n_trials,
                    sanitize=True,
                )
            else:
                raise ValueError(f"Unknown generation method: {generation_method}")
                
            # Compute log probabilities for each generated molecule
            # CRITICAL: Only append molecules for which we can compute log probs
            for mol in generated:
                try:
                    log_prob, entropy = self._compute_log_prob(mol)
                    all_log_probs.append(log_prob)
                    all_entropies.append(entropy)
                    all_valid_mols.append(mol)  # Only append if computation succeeded
                except Exception as e:
                    # Skip molecules that fail encoding/probability calculation
                    logger.debug(f"Failed to compute log prob for molecule: {e}")
                    continue
                    
        # Return empty tensors if no valid molecules
        if not all_valid_mols:
            return [], torch.tensor([], device=self.device), torch.tensor([], device=self.device)
            
        # Stack tensors
        log_probs = torch.stack(all_log_probs)
        entropies = torch.stack(all_entropies)
        
        return all_valid_mols, log_probs, entropies
        
    def _compute_log_prob(self, molecule: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probability and entropy for a generated molecule.
        
        Args:
            molecule: SMILES string of the molecule
            
        Returns:
            Tuple of (log_prob, entropy)
        """
        # Encode molecule to SAFE string
        safe_str = sf.encode(molecule, canonical=True)
        
        # Tokenize
        pretrained_tk = self.tokenizer.get_pretrained()
        inputs = pretrained_tk(
            safe_str,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        
        # Move to device
        input_ids = inputs['input_ids'].to(self.device)
        
        # Remove EOS token from input for computing probabilities
        input_ids = input_ids[:, :-1]
        
        # Get model outputs
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits
            
        # Compute log probabilities for the actual tokens
        # We need to shift: predict token i+1 from tokens 0..i
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Get the log probs of the actual next tokens
        target_ids = inputs['input_ids'][:, 1:].to(self.device)  # Shift targets
        
        # Gather the log probs for actual tokens
        token_log_probs = log_probs.gather(
            dim=-1,
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        # Sum log probs (log of product = sum of logs)
        total_log_prob = token_log_probs.sum()
        
        # Compute entropy
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        return total_log_prob, entropy
        
    def _compute_rewards(
        self,
        molecules: List[str],
        score_functions: List[Callable]
    ) -> torch.Tensor:
        """
        Compute rewards for generated molecules using score functions.
        
        Args:
            molecules: List of SMILES strings
            score_functions: List of scoring functions
            
        Returns:
            Tensor of rewards
        """
        rewards = torch.zeros(len(molecules), device=self.device)
        
        for score_fn in score_functions:
            scores = score_fn(molecules)
            if not isinstance(scores, torch.Tensor):
                scores = torch.tensor(scores, device=self.device, dtype=torch.float32)
            else:
                scores = scores.to(self.device)
            rewards += scores
            
        return rewards
        
    def _normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Normalize rewards to have mean 0 and std 1.
        
        Args:
            rewards: Tensor of rewards
            
        Returns:
            Normalized rewards
        """
        if len(rewards) <= 1:
            return rewards
            
        mean = rewards.mean()
        std = rewards.std()
        
        if std > 0:
            return (rewards - mean) / (std + 1e-8)
        else:
            return rewards - mean
            
    def _compute_kl_divergence(
        self,
        molecules: List[str],
        log_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between current policy and reference policy.
        
        Args:
            molecules: List of generated molecules
            log_probs: Log probabilities from current model
            
        Returns:
            KL divergence
        """
        ref_log_probs = []
        
        for mol in molecules:
            try:
                # Encode and tokenize
                safe_str = sf.encode(mol, canonical=True)
                pretrained_tk = self.tokenizer.get_pretrained()
                inputs = pretrained_tk(
                    safe_str,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                )
                
                input_ids = inputs['input_ids'].to(self.device)
                input_ids = input_ids[:, :-1]
                
                # Get reference model outputs
                with torch.no_grad():
                    ref_outputs = self.ref_model(input_ids=input_ids)
                    ref_logits = ref_outputs.logits
                    
                ref_log_probs_mol = F.log_softmax(ref_logits, dim=-1)
                
                # Get log probs for actual tokens
                target_ids = inputs['input_ids'][:, 1:].to(self.device)
                ref_token_log_probs = ref_log_probs_mol.gather(
                    dim=-1,
                    index=target_ids.unsqueeze(-1)
                ).squeeze(-1)
                
                ref_log_probs.append(ref_token_log_probs.sum())
            except Exception as e:
                logger.debug(f"Failed to compute ref log prob: {e}")
                # Use current log prob as fallback
                ref_log_probs.append(log_probs[len(ref_log_probs)].detach())
                
        ref_log_probs = torch.stack(ref_log_probs)
        
        # KL(current || ref) = log_probs - ref_log_probs
        kl_div = (log_probs - ref_log_probs).mean()
        
        return kl_div
        
    def _compute_ppo_loss(
        self,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        kl_div: torch.Tensor,
        kl_coef: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute PPO loss.
        
        Args:
            log_probs: Log probabilities of generated molecules
            rewards: Normalized rewards
            kl_div: KL divergence
            kl_coef: KL coefficient
            
        Returns:
            PPO loss
        """
        # Policy gradient loss: maximize log_prob * reward
        # We use negative for minimization
        pg_loss = -(log_probs * rewards).mean()
        
        # Total loss with KL penalty
        loss = pg_loss + kl_coef * kl_div
        
        return loss
        
    def _save_model(self, save_path: Union[str, os.PathLike], epoch: int):
        """
        Save model and tokenizer.
        
        Args:
            save_path: Base path for saving
            epoch: Current epoch number
        """
        save_path = Path(save_path)
        epoch_path = save_path / f"epoch_{epoch}"
        epoch_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(epoch_path)
        
        # Save tokenizer
        self.tokenizer.save(epoch_path)
        
        logger.info(f"Model saved to {epoch_path}")


# Alias for backward compatibility
SafePPO = SafePPOTrainer
