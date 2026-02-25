import json
import logging
import os
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.common_utils import check_nan_inf_tensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop

class ModelMetrics:
    """Class to compute and store model metrics"""
    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        self.class_names = ['Benign cases', 'Malignant cases', 'Normal cases']

    @staticmethod
    def _class_key(class_name: str) -> str:
        return class_name.lower().replace(' ', '_')

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                      y_probs: np.ndarray | None = None) -> dict:
        """Calculate accuracy, precision, recall, F1 score, and AUC"""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted
        }
        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            ck = self._class_key(class_name)
            metrics[f'{ck}_precision'] = precision[i] if i < len(precision) else 0.0
            metrics[f'{ck}_recall'] = recall[i] if i < len(recall) else 0.0
            metrics[f'{ck}_f1'] = f1[i] if i < len(f1) else 0.0

        # Medical-specific metrics
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
        for i, class_name in enumerate(self.class_names):
            if i < cm.shape[0]:
                tp = cm[i, i]
                fn = np.sum(cm[i, :]) - tp
                fp = np.sum(cm[:, i]) - tp
                tn = np.sum(cm) - (tp + fn + fp)

                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

                ck = self._class_key(class_name)
                metrics[f'{ck}_sensitivity'] = sensitivity
                metrics[f'{ck}_specificity'] = specificity

        # AUC-ROC if probabilities are provided
        if y_probs is not None:
            try:
                if self.num_classes == 2:
                    auc = roc_auc_score(y_true, y_probs[:, 1])
                    metrics['auc_roc'] = auc
                else:
                    auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
                    metrics['auc_roc_macro'] = auc
            except ValueError:
                logger.warning("Could not calculate AUC-ROC score")

        return metrics


class ModelTrainer:
    """Main Training Engine for the model"""

    def __init__(self, model: nn.Module, device: torch.device, save_dir: str = 'checkpoints', log_dir: str = 'logs'):  # Fixed: was nn.modules
        self.model = model
        self.device = device
        self.save_dir = save_dir
        self.log_dir = log_dir

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.metrics_calculator = ModelMetrics()
        self.writer = SummaryWriter(log_dir=log_dir)
        self.history = defaultdict(list)
        # Enable AMP (mixed precision) on CUDA to reduce VRAM usage
        # Use conservative init_scale to prevent fp16 overflow on first batch
        self.scaler = amp.GradScaler(init_scale=1024) if self.device.type == 'cuda' else None
        if self.scaler:
            logger.info("AMP GradScaler ENABLED (init_scale=1024) — mixed precision training")
        self.current_round = 0

    def get_model_weights(self):
        return self.model.state_dict()

    def set_model_weights(self, weights):
        self.model.load_state_dict(weights)

    def export_weights(self):
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def import_weights(self, weights):
        self.model.load_state_dict(weights)

    def set_round(self, round_num: int):
        self.current_round = round_num
        logger.info(f"Federated Round {round_num}")

    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer,
                   criterion: nn.Module, epoch: int,
                   global_model_weights: dict[str, torch.Tensor] | None = None,
                   mu: float = 0, accumulation_steps: int = 1) -> dict:
        """Train for one epoch with FedProx, AMP, and gradient accumulation"""

        self.model.train()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1} - Training')

        optimizer.zero_grad() # Initialize gradients to zero for accumulation
        total_grad_norm = 0.0
        max_activation = 0.0
        
        # Proactively free cached GPU memory at the start of each epoch
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        for batch_idx, (data, target) in enumerate(progress_bar):
            try:
                data, target = data.to(self.device), target.to(self.device)
            except RuntimeError as cuda_err:
                if "CUDA" in str(cuda_err):
                    logger.error(f"CUDA error during data transfer at epoch {epoch+1}, batch {batch_idx+1}: {cuda_err}")
                    torch.cuda.empty_cache()
                    import gc; gc.collect()
                    try:
                        data, target = data.to(self.device), target.to(self.device)
                    except RuntimeError:
                        logger.critical("CUDA error persists after retry. Returning partial results.")
                        return {
                            'loss': running_loss / max(batch_idx, 1),
                            'accuracy': 0.0, 'grad_norm': 0.0,
                            'f1_macro': 0.0, 'max_activation': 0.0,
                            'cuda_error': True
                        }
                else:
                    raise
            # Ensure data is float32
            data = data.float()

            # Forward pass with AMP autocast and CUDA error recovery
            amp_enabled = self.scaler is not None
            try:
                with torch.amp.autocast(device_type='cuda', enabled=amp_enabled):
                    outputs = self.model(data)
                    # Cast outputs to float32 for numerically stable loss
                    outputs = outputs.float()
                    loss = criterion(outputs, target)
            except RuntimeError as fwd_err:
                if "CUDA" in str(fwd_err) or "out of memory" in str(fwd_err).lower():
                    logger.error(f"CUDA error during forward pass at epoch {epoch+1}, batch {batch_idx+1}: {fwd_err}")
                    torch.cuda.empty_cache()
                    import gc; gc.collect()
                    try:
                        with torch.amp.autocast(device_type='cuda', enabled=amp_enabled):
                            outputs = self.model(data)
                            outputs = outputs.float()
                            loss = criterion(outputs, target)
                    except RuntimeError:
                        logger.critical("CUDA forward pass error persists after retry. Returning partial results.")
                        return {
                            'loss': running_loss / max(batch_idx, 1),
                            'accuracy': 0.0, 'grad_norm': 0.0,
                            'f1_macro': 0.0, 'max_activation': 0.0,
                            'cuda_error': True
                        }
                else:
                    raise
            
            # SECTION 11 – NUMERICAL STABILITY: Validate model outputs
            if not torch.isfinite(outputs).all():
                logger.critical(f"Model outputs contain NaN/Inf at epoch {epoch+1}, batch {batch_idx+1}")
                raise RuntimeError("Model produced non-finite outputs")
            
            # Track max activation magnitude
            batch_max = float(outputs.abs().max().item())
            if batch_max > max_activation:
                max_activation = batch_max

            # Check for NaNs early
            if not torch.isfinite(loss).all():
                msg = f"Training Loss became NaN or Inf at epoch {epoch+1}, batch {batch_idx+1}. Halting training for this round."
                logger.critical(msg)
                raise RuntimeError(msg)

            # FedProx: Add proximal term
            if global_model_weights is not None and mu > 0:
                prox_term = 0.0
                for (name, param), (global_name, global_param) in zip(
                    self.model.named_parameters(), global_model_weights.items()
                ):
                    if param.requires_grad:
                        prox_term += ((param - global_param.to(self.device))**2).sum()
                
                # Log proximal term for debugging
                if batch_idx == 0:  # Log only first batch
                    logger.info(f"Epoch {epoch+1}, Batch 1: Proximal term = {prox_term.item():.6f}, mu = {mu}")
                
                loss += (mu / 2) * prox_term

            loss = loss / accumulation_steps 

            if self.scaler: # AMP backward pass
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Accumulate gradients and update model parameters
            if (batch_idx + 1) % accumulation_steps == 0:
                if self.scaler: # AMP optimizer step
                    self.scaler.unscale_(optimizer) 
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # VERIFICATION: Recompute actual norm after clipping to ensure it worked
                    try:
                        actual_norm_after = torch.norm(torch.stack([
                            torch.norm(p.grad.detach(), 2)
                            for p in self.model.parameters()
                            if p.grad is not None
                        ])).item()
                    except RuntimeError as e:
                        if "CUDA" in str(e):
                            logger.warning(f"CUDA error in grad norm verification (AMP), skipping check: {e}")
                            torch.cuda.empty_cache()
                            actual_norm_after = 0.0
                        else:
                            raise
                    
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else: # Standard optimizer step
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # VERIFICATION: Recompute actual norm after clipping
                    try:
                        actual_norm_after = torch.norm(torch.stack([
                            torch.norm(p.grad.detach(), 2)
                            for p in self.model.parameters()
                            if p.grad is not None
                        ])).item()
                    except RuntimeError as e:
                        if "CUDA" in str(e):
                            logger.warning(f"CUDA error in grad norm verification, skipping check: {e}")
                            torch.cuda.empty_cache()
                            actual_norm_after = 0.0
                        else:
                            raise
                    
                    optimizer.step()

                # Verify gradient clipping worked correctly
                if actual_norm_after > 1.1:  # Allow small epsilon for numerical precision
                    logger.error(f"GRADIENT CLIPPING FAILED: Pre-clip={total_grad_norm:.4f}, Post-clip={actual_norm_after:.4f}")
                    logger.error(f"Epoch {epoch+1}, Batch {batch_idx+1}: Clipping did not constrain gradients!")
                
                # Log for debugging (every 10 batches)
                if batch_idx % 10 == 0:
                    logger.debug(f"Grad norm: pre={total_grad_norm:.4f}, post={actual_norm_after:.4f}")
                
                # Abort training if norm is NaN
                if not np.isfinite(actual_norm_after):
                    logger.critical(f"Gradient Norm became NaN at epoch {epoch+1}, batch {batch_idx+1}. Aborting.")
                    return {
                        'loss': float('nan'),
                        'accuracy': float('nan'),
                        'grad_norm': float('nan'),
                        'f1_macro': 0.0 
                    }

                optimizer.zero_grad() 

            # Statistics
            running_loss += loss.item() * accumulation_steps 
            with torch.no_grad():
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().detach().numpy())

            # Update progress bar
            progress_bar.set_postfix({'Loss': running_loss / ((batch_idx + 1) * accumulation_steps)}) 

            # Log to tensorboard
            global_step = epoch * len(train_loader) + batch_idx
            self.writer.add_scalar('Train/BatchLoss', loss.item() * accumulation_steps, global_step) # Log actual batch loss

        # Handle remaining gradients if batches are not perfectly divisible by accumulation_steps
        if (len(train_loader) % accumulation_steps != 0) and (accumulation_steps > 1):
            if self.scaler:
                self.scaler.unscale_(optimizer)
                total_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Verify gradient clipping worked
            if total_grad_norm > 1.1:
                logger.warning(f"Final batch gradient norm {total_grad_norm:.4f} exceeds max_norm=1.0 after clipping")

            optimizer.zero_grad()

        # Calculate epoch metrics
        avg_loss = running_loss / len(train_loader)
        metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_labels),
            np.array(all_predictions),
            np.array(all_probabilities)
        )
        metrics['loss'] = avg_loss
        metrics['accuracy'] = accuracy_score(np.array(all_labels), np.array(all_predictions))
        metrics['grad_norm'] = total_grad_norm
        metrics['max_activation'] = max_activation
        return metrics

    def validate_epoch(self, val_loader: DataLoader, criterion: nn.Module, epoch: int) -> dict:
        """Validate for one epoch - This method was missing in your original code"""
        self.model.eval()

        running_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1} - Validation')

            for data, target in progress_bar:
                try:
                    data, target = data.to(self.device), target.to(self.device)
                except RuntimeError as cuda_err:
                    if "CUDA" in str(cuda_err):
                        logger.error(f"CUDA error during val data transfer: {cuda_err}")
                        torch.cuda.empty_cache()
                        data, target = data.to(self.device), target.to(self.device)
                    else:
                        raise
                # Ensure data is float32
                data = data.float()

                try:
                    with torch.amp.autocast(device_type='cuda', enabled=self.scaler is not None):
                        outputs = self.model(data)
                        outputs = outputs.float()  # Ensure float32 for loss computation
                except RuntimeError as fwd_err:
                    if "CUDA" in str(fwd_err) or "out of memory" in str(fwd_err).lower():
                        logger.error(f"CUDA error during val forward pass: {fwd_err}")
                        torch.cuda.empty_cache()
                        try:
                            with torch.amp.autocast(device_type='cuda', enabled=self.scaler is not None):
                                outputs = self.model(data)
                                outputs = outputs.float()
                        except RuntimeError:
                            logger.critical("CUDA val forward error persists. Skipping batch.")
                            continue
                    else:
                        raise

                loss = criterion(outputs, target)

                running_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().detach().numpy())

                progress_bar.set_postfix({'Loss': loss.item()})

        # Calculate epoch metrics
        avg_loss = running_loss / len(val_loader)
        metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_labels),
            np.array(all_predictions),
            np.array(all_probabilities)
        )
        metrics['loss'] = avg_loss

        return metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
          num_epochs: int = 50, learning_rate: float = 0.001,
          weight_decay: float = 1e-4, class_weights: torch.Tensor | None = None,
          use_scheduler: bool = True, patience: int = 10,
          criterion: nn.Module | None = None,
          optimizer_name: str = 'adamw',
          scheduler_name: str = 'plateau',
          global_model_weights: dict[str, torch.Tensor] | None = None,
          mu: float = 0, accumulation_steps: int = 1) -> dict:
        """
        Complete training loop with detailed logging, scheduler, early stopping,
        and support for external criterion and optimizer config.
        """

        #Optimizer and Scheduler
        optimizer = get_optimizer(self.model, optimizer_name, learning_rate, weight_decay)

        if use_scheduler:
            scheduler = get_scheduler(optimizer, scheduler_name)
        else:
            scheduler = None # Ensure scheduler is None if not used

        #Loss Function
        if criterion is None:
            if class_weights is not None:
                criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
            else:
                criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        best_model_state = None
        early_stopping = EarlyStopping(patience=patience, mode='min')

        logger.info(f"Starting training loop with optimizer={optimizer_name}, scheduler={scheduler_name if use_scheduler else 'None'}")

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_loader, optimizer, criterion, epoch,
                                             global_model_weights, mu, accumulation_steps)
            avg_epoch_loss = train_metrics['loss']
            epoch_accuracy = train_metrics['accuracy']

            #Validation
            val_metrics = self.validate_epoch(val_loader, criterion, epoch)
            val_loss = val_metrics['loss']

            #Scheduler Step
            if use_scheduler:
                if scheduler_name.lower() == 'plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {avg_epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

            self._log_epoch_metrics(epoch, {
                'loss': avg_epoch_loss,
                'accuracy': epoch_accuracy
            }, val_metrics)

            self.history['train_loss'].append(avg_epoch_loss)
            self.history['train_accuracy'].append(epoch_accuracy)
            self.history['train_f1_macro'].append(train_metrics.get('f1_macro', 0.0))
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_f1_macro'].append(val_metrics['f1_macro'])

            # Bug fix: Store max_activation and grad_norm in history
            self.history['max_activation'].append(train_metrics.get('max_activation', 0.0))
            self.history['grad_norm'].append(train_metrics.get('grad_norm', 0.0))

            #Best Model Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()
                self.save_checkpoint(epoch, train_metrics, val_metrics, optimizer, scheduler, is_best=True)
            elif early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break



        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info("Best model weights loaded.")

        logger.info("Training loop completed.")
        self.writer.close()

        self.plot_training_history(metrics=['loss'], save_path=os.path.join(self.save_dir, 'loss_curve.png'))
        self.plot_training_history(metrics=['accuracy'], save_path=os.path.join(self.save_dir, 'accuracy_curve.png'))

        return self.history



    def evaluate(self, test_loader: DataLoader, save_name: str = "confusion_matrix.png") -> dict:
        """Comprehensive evaluation on test set"""
        self.model.eval()
        
        # Verify model is in eval mode
        if self.model.training:
            logger.error("Model is still in training mode! Forcing eval mode.")
            self.model.eval()
        
        # Additional check: ensure BatchNorm layers are in eval mode
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                if module.training:
                    logger.warning(f"BatchNorm layer {name} was in training mode, switching to eval")
                    module.eval()

        all_predictions = []
        all_labels = []
        all_probabilities = []
        running_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in tqdm(test_loader, desc='Evaluating'):
                try:
                    data, target = data.to(self.device), target.to(self.device)
                except RuntimeError as cuda_err:
                    if "CUDA" in str(cuda_err):
                        logger.error(f"CUDA error during eval data transfer: {cuda_err}")
                        torch.cuda.empty_cache()
                        data, target = data.to(self.device), target.to(self.device)
                    else:
                        raise
                # Ensure data is float32
                data = data.float()
                
                # For single-sample batches, ensure model is in eval mode
                # This prevents BatchNorm errors
                if data.size(0) == 1:
                    # Already in eval mode from above, but explicitly verify
                    if self.model.training:
                        logger.warning("Model was in training mode during single-sample evaluation, switching to eval")
                        self.model.eval()
                
                try:
                    with torch.amp.autocast(device_type='cuda', enabled=self.scaler is not None):
                        outputs = self.model(data)
                        outputs = outputs.float()  # Ensure float32
                except RuntimeError as fwd_err:
                    if "CUDA" in str(fwd_err) or "out of memory" in str(fwd_err).lower():
                        logger.error(f"CUDA error during eval forward pass: {fwd_err}")
                        torch.cuda.empty_cache()
                        try:
                            with torch.amp.autocast(device_type='cuda', enabled=self.scaler is not None):
                                outputs = self.model(data)
                                outputs = outputs.float()
                        except RuntimeError:
                            logger.critical("CUDA eval forward error persists. Skipping batch.")
                            continue
                    else:
                        raise
                loss = criterion(outputs, target)
                running_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # Log prediction distribution to debug class collapse
        pred_counter = Counter(all_predictions)
        label_counter = Counter(all_labels)
        logger.info(f"Prediction distribution: {dict(pred_counter)}")
        logger.info(f"True label distribution: {dict(label_counter)}")
        
        # Check for severe class imbalance in predictions (possible model collapse)
        if len(pred_counter) < len(label_counter):
            missing_classes = set(label_counter.keys()) - set(pred_counter.keys())
            logger.warning(f"Model is NOT predicting classes: {missing_classes} - possible model collapse!")
        
        # Check if model is predicting only one class
        if len(pred_counter) == 1:
            logger.error(f"MODEL COLLAPSE DETECTED: Only predicting class {list(pred_counter.keys())[0]}!")

        # Calculate comprehensive metrics
        test_metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_labels),
            np.array(all_predictions),
            np.array(all_probabilities)
        )
        test_metrics['loss'] = running_loss / len(test_loader)

        # Bug fix: Include predictions and labels in returned metrics for saving
        # Convert to native Python int to avoid numpy.int64 JSON/Flower serialization errors
        test_metrics['predictions'] = [int(p) for p in all_predictions]
        test_metrics['labels'] = [int(l) for l in all_labels]

        # Generate and save confusion matrix
        self.plot_confusion_matrix(all_labels, all_predictions, save_path=os.path.join(self.save_dir, save_name))

        # Generate classification report
        self._generate_classification_report(test_metrics)

        return test_metrics


    def plot_confusion_matrix(self, y_true: list, y_pred: list, save_path: str = None):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(self.metrics_calculator.num_classes)),
        )

        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.metrics_calculator.class_names,
                   yticklabels=self.metrics_calculator.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        if save_path is None:
            save_path = os.path.join(self.save_dir, 'confusion_matrix.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        # plt.show()

    def plot_training_history(self, metrics: list[str] = ['loss', 'accuracy'], save_path: str = None):
        """Plot training history"""
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
        if len(metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            train_key = f'train_{metric}'
            val_key = f'val_{metric}'

            if train_key in self.history and val_key in self.history:
                axes[i].plot(self.history[train_key], label=f'Train {metric}')
                axes[i].plot(self.history[val_key], label=f'Val {metric}')
                axes[i].set_title(f'{metric.capitalize()} History')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric.capitalize())
                axes[i].legend()
                axes[i].grid(True)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.save_dir, 'training_history.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        # plt.show()


    def save_checkpoint(self, epoch: int, train_metrics: dict, val_metrics: dict, optimizer: optim.Optimizer, scheduler, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }

        if is_best:
            checkpoint_path = os.path.join(self.save_dir, 'best_model.pth')
        else:
            checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth')

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> dict:
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint

    def _log_epoch_metrics(self, epoch: int, train_metrics: dict, val_metrics: dict):
        """Log metrics to tensorboard and console"""

        # Log to tensorboard
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'Train/{key}', value, epoch)
        for key, value in val_metrics.items():
            self.writer.add_scalar(f'Val/{key}', value, epoch)

        # Log to console
        logger.info(f"Epoch {epoch+1}:")
        logger.info(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        logger.info(f"  Val F1: {val_metrics['f1_macro']:.4f}")

    def _generate_classification_report(self, metrics: dict):
        """Generate and save detailed classification report"""
        report = {
            'Overall Metrics': {
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Macro F1': f"{metrics['f1_macro']:.4f}",
                'Weighted F1': f"{metrics['f1_weighted']:.4f}",
            },
            'Per-Class Metrics': {}
        }

        for class_name in self.metrics_calculator.class_names:
            class_key = self.metrics_calculator._class_key(class_name)
            report['Per-Class Metrics'][class_name] = {
                'Precision': f"{metrics.get(f'{class_key}_precision', 0):.4f}",
                'Recall': f"{metrics.get(f'{class_key}_recall', 0):.4f}",
                'F1-Score': f"{metrics.get(f'{class_key}_f1', 0):.4f}",
                'Sensitivity': f"{metrics.get(f'{class_key}_sensitivity', 0):.4f}",
                'Specificity': f"{metrics.get(f'{class_key}_specificity', 0):.4f}",
            }

        # Save report
        report_path = os.path.join(self.save_dir, 'classification_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Classification report saved: {report_path}")

        # Print summary
        logger.info("=== CLASSIFICATION REPORT ===")
        logger.info(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Macro F1-Score: {metrics['f1_macro']:.4f}")

        for class_name in self.metrics_calculator.class_names:
            class_key = self.metrics_calculator._class_key(class_name)
            logger.info(f"{class_name}: F1={metrics.get(f'{class_key}_f1', 0):.4f}, "
                       f"Precision={metrics.get(f'{class_key}_precision', 0):.4f}, "
                       f"Recall={metrics.get(f'{class_key}_recall', 0):.4f}")


def get_optimizer(model: nn.Module, optimizer_name: str = 'adamw',
                 learning_rate: float = 0.001, weight_decay: float = 1e-4) -> optim.Optimizer:
    """Get optimizer for training"""
    if optimizer_name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def get_scheduler(optimizer: optim.Optimizer, scheduler_name: str = 'plateau'):
    """Get learning rate scheduler"""
    if scheduler_name.lower() == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
    elif scheduler_name.lower() == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    elif scheduler_name.lower() == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_name.lower() == 'warmup_cosine': # New: For Transformer models
        # This assumes T_max for CosineAnnealingLR might need to be passed
        # A default of 50 is used, as in the user's snippet
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5), # Warmup for 5 epochs
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50) # Main annealing part
            ],
            milestones=[5] # Transition from LinearLR to CosineAnnealingLR after 5 epochs
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


























































































































































































































































































































































































































































































































