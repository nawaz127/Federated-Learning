import argparse
import hashlib
import json
import logging
import math
import os
import sys
import tempfile
import threading
import time
import warnings
from collections import OrderedDict, defaultdict
from datetime import datetime

import flwr as fl
import matplotlib.pyplot as plt
import numpy as np
import torch
from flwr.server.client_manager import ClientProxy, SimpleClientManager
from torch.utils.data import DataLoader

from models.model_factory import get_model
from utils.dataloder import CTScanDataset, get_medical_transforms
from utils.train_eval import ModelMetrics, ModelTrainer
from utils.xai.xai_metrics import decode_cam_stack, compute_cam_similarity
from utils.xai.xai_plot import plot_xai_consistency

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

RESULTS_BASE_DIR = os.path.abspath("Result/FLResult")
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

logger = logging.getLogger("FL-Server")


def _sanitize_for_json(obj):
    """Recursively replace NaN/Inf with None for valid JSON serialization."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _atomic_json_dump(data, filepath):
    """Write JSON atomically: write to temp file, flush, then rename.
    Prevents null-byte corruption if the process is killed mid-write."""
    data = _sanitize_for_json(data)
    dirpath = os.path.dirname(filepath)
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=dirpath)
    try:
        with os.fdopen(fd, "w") as tmp:
            json.dump(data, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, filepath)
    except BaseException:
        # Clean up temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

# (Optional) large message allowance for big models/metrics
GRPC_MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024  # 1GB

def get_init_parameters(
    model_name: str,
    num_classes: int,
    lsetnet_num_transformer_blocks: int | None = None,
    lsetnet_num_heads: int | None = None,
    lsetnet_ff_dim_multiplier: int | None = None,
) -> fl.common.Parameters:
    """Build the initial model and convert its state_dict to Flower Parameters.
    
    CRITICAL: Excludes BatchNorm running statistics to match FedBN behavior.
    Only trainable parameters (weights, biases, BN gamma/beta) are sent to clients.
    """
    try:
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        if models_dir not in sys.path:
            sys.path.insert(0, models_dir)

        model = get_model(
            model_name,
            num_classes,
            pretrained=True,
            lsetnet_num_transformer_blocks=lsetnet_num_transformer_blocks,
            lsetnet_num_heads=lsetnet_num_heads,
            lsetnet_ff_dim_multiplier=lsetnet_ff_dim_multiplier,
        )

        with torch.no_grad():
            # CRITICAL FIX: Exclude BatchNorm running statistics
            # This must match what clients send in get_parameters()
            parameters = []
            for name, param in model.state_dict().items():
                # Skip BatchNorm running statistics
                if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                    continue
                # Include all other parameters
                parameters.append(param.cpu().numpy())

        logger.info(f"Initial model parameters loaded for: {model_name}")
        logger.info("Server Model parameters shapes (trainable only, excluding BN stats):")
        total_params = 0
        for name, param in model.state_dict().items():
            if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                continue
            param_size = param.numel()
            logger.info(f"  {name}: shape={tuple(param.shape)}, size={param_size}")
            total_params += param_size
        logger.info(f"Total number of trainable parameters: {total_params:,}")
        logger.info(f"Note: BatchNorm running stats excluded (FedBN behavior)")

        return fl.common.ndarrays_to_parameters(parameters)
    except Exception as e:
        logger.error(f"Failed to get initial parameters: {e}", exc_info=True)
        return None

def hash_model(parameters: list[np.ndarray]) -> str:
    """Compute SHA256 hash of model parameters."""
    m = hashlib.sha256()
    for p in parameters:
        m.update(p.tobytes())
    return m.hexdigest()





def evaluate_config(server_round: int) -> dict[str, fl.common.Scalar]:
    return {"server_round": server_round}


def weighted_average(metrics: list[tuple[int, dict]]) -> dict:
    """Weighted average across client metrics dictionaries."""
    logger.info(f"Aggregating metrics from {len(metrics)} clients")
    if not metrics:
        return {}
    total_samples = sum(num_samples for num_samples, _ in metrics)
    if total_samples == 0:
        return {}
    aggregated_metrics: dict[str, float] = {}
    for num_samples, client_metrics in metrics:
        weight = num_samples / total_samples
        for key, value in client_metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                aggregated_metrics[key] = aggregated_metrics.get(key, 0.0) + weight * float(value)
    return aggregated_metrics

def aggregate_weighted_average(weights_results):
    """
    Manually aggregate model parameters using weighted average (FedAvg).
    
    Args:
        weights_results: List of tuples (parameters_list, num_examples)
    
    Returns:
        List of aggregated parameter arrays
    """
    if not weights_results:
        return None
    
    # Calculate total examples
    total_examples = sum(num_examples for _, num_examples in weights_results)
    
    if total_examples == 0:
        logger.error("Total examples is zero, cannot aggregate")
        return None
    
    # Get number of layers from first client
    num_layers = len(weights_results[0][0])
    
    # Aggregate each layer separately
    aggregated_ndarrays = []
    
    for layer_idx in range(num_layers):
        # Weighted sum for this layer
        weighted_sum = None
        
        for parameters, num_examples in weights_results:
            layer_param = parameters[layer_idx].astype(np.float64)  # Use float64 for accumulation
            weight = num_examples / total_examples
            
            if weighted_sum is None:
                weighted_sum = layer_param * weight
            else:
                weighted_sum += layer_param * weight
        
        # Check for numerical stability
        if not np.isfinite(weighted_sum).all():
            logger.critical(f"NaN/Inf detected in aggregated layer {layer_idx}")
            raise RuntimeError(f"Aggregation failed: non-finite values in layer {layer_idx}")
        
        # Convert back to float32 for storage
        aggregated_ndarrays.append(weighted_sum.astype(np.float32))
    
    return aggregated_ndarrays

class MedicalFLStrategy(fl.server.strategy.FedAvg):
    """
    FedAvg strategy extended with:
      - history tracking
      - best/last global checkpoint saving
      - detailed round logging & plots
      - optional XAI metrics aggregation
      - FedProx support
      
    """
    def __init__(
        self,
        *,
        model_name: str,
        num_classes: int,
        num_rounds: int | None = None,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures: bool = True,
        initial_parameters: fl.common.Parameters | None = None,
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
        results_base_dir: str = RESULTS_BASE_DIR,
        aggregation: str = "fedavg",
        mu: float = 0.01,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        global_val_data_dir: str | None = None,
        global_final_test_data_dir: str | None = None,
        lsetnet_num_transformer_blocks: int | None = None,
        lsetnet_num_heads: int | None = None,
        lsetnet_ff_dim_multiplier: int | None = None,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.model_name = model_name
        self.num_classes = num_classes
        self.num_rounds = num_rounds
        self.aggregation = aggregation
        self.mu = mu
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lsetnet_num_transformer_blocks = lsetnet_num_transformer_blocks
        self.lsetnet_num_heads = lsetnet_num_heads
        self.lsetnet_ff_dim_multiplier = lsetnet_ff_dim_multiplier

        logger.info(
            f"   → aggregation={self.aggregation}, mu={self.mu}, "
            f"lr={self.learning_rate}, wd={self.weight_decay}"
        )

        self.history = {
            "round": [],
            "train_loss": [],
            "train_accuracy": [],
            "train_f1": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "test_loss": [],
            "test_accuracy": [],
            "test_f1": [],
            "num_clients": [],
            "client_data_sizes": [],
            "aggregation_time": [],
            # ---- XAI history ----
            "xai_del_auc_mean": [], "xai_del_auc_std": [],
            "xai_heat_in_mask_mean": [], "xai_heat_in_mask_std": [],
            "xai_ins_auc_mean": [], "xai_ins_auc_std": [],
            "xai_cam_consistency_mean": [], "xai_cam_consistency_std": [],
            "xai_temporal_stability_mean": [], "xai_temporal_stability_std": [],
            "xai_temporal_pearson_mean": [], "xai_temporal_pearson_std": [],
            "xai_client_stability_mean": [], "xai_client_stability_std": [],
            "xai_client_pearson_mean": [], "xai_client_pearson_std": [],
            # ---- Global test metrics ----
            "global_test_accuracy": [],
            "global_val_accuracy": [],
            # ---- FedProx Convergence Monitoring ----
            "client_drift_avg": [],
            # ---- Model Divergence Tracking ----
            "cnn_drift_avg": [],
            "transformer_drift_avg": [],
            # ---- Mandatory Publication Tracking ----
            "global_weight_norm": [],
            "max_activation_magnitude": [],
            "client_drift_max": [],
            "xai_cross_method_agreement_mean": [],
        }
        self.round_history = []
        self.communication_history = []
        self.best_accuracy = 0.0
        self.best_f1 = 0.0
        self.best_round = 0
        self.best_parameters: fl.common.Parameters | None = None
        self.last_parameters: fl.common.Parameters | None = None

        self.connected_clients = set()
        self.client_metrics_history = {}
        self.client_accuracy_history = defaultdict(list)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_base_dir = os.path.join(results_base_dir, f"fl_results_{ts}")
        os.makedirs(self.results_base_dir, exist_ok=True)
        self._save_strategy_config()

        # Server-side global model evaluation setup
        self.global_val_loader = None
        self.global_test_loader = None # This one already exists, but re-initialize for clarity
        self.global_model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model_instance = get_model(
            self.model_name,
            self.num_classes,
            pretrained=True,
            lsetnet_num_transformer_blocks=self.lsetnet_num_transformer_blocks,
            lsetnet_num_heads=self.lsetnet_num_heads,
            lsetnet_ff_dim_multiplier=self.lsetnet_ff_dim_multiplier
        ).to(self.global_model_device)
        self.global_model_trainer = ModelTrainer(self.global_model_instance, self.global_model_device,
                                                os.path.join(self.results_base_dir, "server_checkpoints"),
                                                os.path.join(self.results_base_dir, "server_logs"))
        self.global_model_metrics_calculator = ModelMetrics(num_classes=self.num_classes)


        if global_val_data_dir:
            global_val_dataset = CTScanDataset(
                global_val_data_dir,
                transform=get_medical_transforms(subset='val')
            )
            self.global_val_loader = DataLoader(
                global_val_dataset, batch_size=32, shuffle=False, num_workers=0
            )
            logger.info(f"   → Global validation dataset loaded from {global_val_data_dir} "
                        f"with {len(self.global_val_loader.dataset)} samples.")

            # Calculate global class distribution for heterogeneity metrics (from validation set for monitoring)
            global_labels = [sample[1] for sample in self.global_val_loader.dataset.samples]
            global_class_counts = np.bincount(global_labels, minlength=self.num_classes)
            self.global_class_distribution = (global_class_counts / global_class_counts.sum()).tolist()
            logger.info(f"   → Global class distribution (from val set): {self.global_class_distribution}")
        else:
            logger.warning("No global validation data directory provided. Server-side validation will be skipped.")
            self.global_class_distribution = None # Ensure it's None if no val data for distribution


        if global_final_test_data_dir:
            global_test_dataset = CTScanDataset(
                global_final_test_data_dir,
                transform=get_medical_transforms(subset='test')
            )
            self.global_test_loader = DataLoader(
                global_test_dataset, batch_size=32, shuffle=False, num_workers=0
            )
            logger.info(f"   → Global final test dataset loaded from {global_final_test_data_dir} "
                        f"with {len(self.global_test_loader.dataset)} samples.")
        else:
            logger.warning("No global final test data directory provided. Final server-side testing will be skipped.")

        logger.info("FL Strategy initialized")
        logger.info(f"   → results_base_dir: {self.results_base_dir}")
        logger.info(f"   → model={self.model_name}, num_classes={self.num_classes}")
    def _save_strategy_config(self):
        config = {
            "strategy": "MedicalFLStrategy",
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "fraction_fit": self.fraction_fit,
            "fraction_evaluate": self.fraction_evaluate,
            "min_fit_clients": self.min_fit_clients,
            "min_evaluate_clients": self.min_evaluate_clients,
            "min_available_clients": self.min_available_clients,
            "accept_failures": self.accept_failures,
            "timestamp": datetime.now().isoformat(),
        }
        with open(os.path.join(self.results_base_dir, "strategy_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def _fit_config(self, server_round: int, local_epochs: int) -> dict[str, fl.common.Scalar]:
        """Per-round training config broadcast to clients."""
        # Use user-configurable base learning rate from CLI.
        base_lr = float(self.learning_rate)
        cnn_models = ["resnet50", "densenet121", "mobilenetv3", "customcnn"]
        transformer_models = ["vit", "swin_tiny"]
        hybrid_models = ["hybridmodel", "hybridswin", "LSeTNet"]
        
        # Keep model-specific adjustments but scale from configured base LR.
        if self.model_name in cnn_models:
            base_lr = float(self.learning_rate)
        elif self.model_name in hybrid_models:
            base_lr = float(self.learning_rate) * 0.5  # Slightly lower for hybrids
        elif self.model_name in transformer_models:
            base_lr = float(self.learning_rate) * 0.1  # Lower for transformers

        config = {
            "server_round": server_round,
            "local_epochs": local_epochs,
            "learning_rate": base_lr,
            "weight_decay": float(self.weight_decay),
            "loss_function": "cross_entropy",
            "optimizer": "adamw",
            "scheduler": "plateau", # Default scheduler
            "use_scheduler": True,
            "batch_size": 16,
            "xai_probe": True,
            "xai_samples": 3,  # 1 sample per class for XAI (3 classes)
            "xai_save_k": 0,
            "xai_shared_probe_dir": r"C:/Users/shahn/source/repos/Federated-Learning/Federated_Dataset/val",
            "xai_cam_downsample": 32,
            "xai_run_shap": False,   # SHAP disabled
            "xai_run_lime": True,
            "xai_run_attention": False, # Attention Rollout disabled
            "num_rounds": self.num_rounds if self.num_rounds is not None else 0,
        }

        # Convergence stabilization policy for ResNet50 in FL:
        # - cap LR to 3e-4 for batch_size=16
        # - reduce local epochs after round 1 to lower client drift
        if self.model_name == "resnet50":
            config["learning_rate"] = min(float(config["learning_rate"]), 3e-4)
            if server_round >= 2:
                config["local_epochs"] = min(int(config["local_epochs"]), 2)

        if self.num_rounds is not None:
            config["xai_run_heavy"] = (server_round == self.num_rounds)
        else:
            config["xai_run_heavy"] = False

        # Dynamic scheduler based on model name
        transformer_models = ["vit", "swin_tiny", "hybridmodel", "hybridswin", "LSeTNet"] # List of models considered "Transformer" (includes hybrids with transformer blocks)
        if self.model_name in transformer_models:
            config["scheduler"] = "warmup_cosine"
            # Adjust local_epochs or T_max for CosineAnnealingLR if necessary, e.g.,
            # if T_max in client is 50, then local_epochs should be >= 50 for full annealing.
            # For now, keep local_epochs as provided by server, assuming T_max on client is default 50.
            if config["local_epochs"] < 50:
                 logger.warning(f"Model '{self.model_name}' detected. Using 'warmup_cosine' scheduler. "
                                f"Consider increasing 'local_epochs' (current: {config['local_epochs']}) "
                                f"to at least 50 for full CosineAnnealingLR effect.")


        if server_round > 20:
            config["loss_function"] = "focal"
        if 32 <= server_round <= 60:
            config["learning_rate"] = base_lr * 0.5  # Decay relative to model-specific base LR
            config["local_epochs"] = 4
        elif 61 <= server_round <= 80:
            config["learning_rate"] = base_lr * 0.2
            config["local_epochs"] = 3
        elif server_round > 80:
            config["learning_rate"] = base_lr * 0.1
            config["local_epochs"] = 2

        logger.info(
            f"Round {server_round} training config: "
            f"epochs={config['local_epochs']}, lr={config['learning_rate']}, "
            f"loss={config['loss_function']}, scheduler={config['scheduler']}" # Log new scheduler
        )
        return config

    def configure_fit(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        logger.info(f"Round {server_round}: configuring clients for training...")
        config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}

        # Add FedProx parameters to config
        config["aggregation"] = self.aggregation
        config["mu"] = self.mu
        if self.global_class_distribution is not None:
            config["global_class_distribution"] = json.dumps(self.global_class_distribution)

        sample_size, min_num = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num)
        self.connected_clients.update({c.cid for c in clients})

        logger.info(f"Selected {len(clients)} clients: {sorted([c.cid for c in clients])}")
        logger.info(
            f"   → aggregation={self.aggregation}, mu={self.mu}, "
            f"lr={config.get('learning_rate')}, wd={config.get('weight_decay')}"
        )
        fit_ins = fl.common.FitIns(parameters, config)
        return [(c, fit_ins) for c in clients]

    def configure_evaluate(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> list[tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateIns]]:
        if self.fraction_evaluate == 0.0:
            return []
        logger.info(f"Round {server_round}: configuring clients for evaluation...")
        config = self.on_evaluate_config_fn(server_round) if self.on_evaluate_config_fn else {}
        sample_size, min_num = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num)
        logger.info(f"Selected {len(clients)} clients for evaluation")

        eval_ins = fl.common.EvaluateIns(parameters, config)
        return [(c, eval_ins) for c in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes] | BaseException],
    ) -> tuple[fl.common.Parameters | None, dict[str, fl.common.Scalar]]:
        """Aggregate fit results with custom logic for FedProx and Trimmed Mean"""
        t0 = time.time()
        logger.info(
            f"Round {server_round}: aggregating fit results (success={len(results)}, failures={len(failures)})"
        )

        if len(results) < self.min_fit_clients:
            logger.warning(
                f"Not enough results to aggregate. Expected {self.min_fit_clients}, got {len(results)}"
            )
            return None, {}

        # Aggregate parameters for FedProx explicitly if needed, otherwise use super's method
        if self.aggregation == "fedprox":
            # Extract weights from results
            weights_results = [
                (
                    fl.common.parameters_to_ndarrays(fit_res.parameters),
                    fit_res.num_examples,
                )
                for _, fit_res in results
            ]

            # Aggregate weights using manual FedAvg aggregation logic
            aggregated_ndarrays = aggregate_weighted_average(weights_results)
            
            if aggregated_ndarrays is None:
                logger.error("FedProx aggregation failed")
                return None, {}
            
            aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_ndarrays)

            # Use super's method to aggregate metrics and handle failures
            _, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        elif self.aggregation == "trimmed_mean":
            # Extract weights from results
            weights_results = [
                (
                    fl.common.parameters_to_ndarrays(fit_res.parameters),
                    fit_res.num_examples,
                )
                for _, fit_res in results
            ]
            
            num_clients = len(weights_results)
            trim_ratio = 0.1 # Standard trim ratio
            n_trim = int(num_clients * trim_ratio)
            
            if n_trim * 2 >= num_clients:
                logger.info(f"Round {server_round}: Too few clients ({num_clients}) for trimming, falling back to FedAvg")
                aggregated_ndarrays = aggregate_weighted_average(weights_results)
            else:
                logger.info(f"Round {server_round}: Applying Trimmed Mean aggregation (trimming {n_trim} clients from each end)")
                num_layers = len(weights_results[0][0])
                aggregated_ndarrays = []
                for i in range(num_layers):
                    layer_updates = np.stack([w[0][i] for w in weights_results])
                    sorted_updates = np.sort(layer_updates, axis=0)
                    trimmed = sorted_updates[n_trim : num_clients - n_trim]
                    aggregated_ndarrays.append(np.mean(trimmed, axis=0))
            
            if aggregated_ndarrays is None:
                logger.error("Trimmed mean aggregation failed")
                return None, {}
                
            aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_ndarrays)
            _, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        else:
            # For other aggregations (e.g., FedAvg), use the default behavior
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is None:
            return None, aggregated_metrics

        # SECTION 3 — FEDERATED LEARNING NUMERICAL HARDENING
        # after aggregation: for param in global_model.parameters(): assert torch.isfinite(param).all()
        aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
        
        # CRITICAL: Validate aggregated parameters are finite
        for idx, param_array in enumerate(aggregated_ndarrays):
            if not np.isfinite(param_array).all():
                logger.critical(f"Server aggregation produced NaN/Inf in layer {idx} at round {server_round}")
                logger.critical(f"  Layer shape: {param_array.shape}, dtype: {param_array.dtype}")
                logger.critical(f"  NaN count: {np.isnan(param_array).sum()}, Inf count: {np.isinf(param_array).sum()}")
                raise RuntimeError(f"Aggregation numerical instability detected in layer {idx}")
        
        logger.info(f"Round {server_round}: Aggregation stability check PASSED - all parameters finite")

        # MANDATORY PUBLICATION LOGGING: Weight Norm
        global_weight_norm = float(np.sqrt(sum(np.sum(np.square(arr)) for arr in aggregated_ndarrays)))
        self.history["global_weight_norm"].append(global_weight_norm)
        logger.info(f"Round {server_round}: Global model weight norm = {global_weight_norm:.4f}")

        # MANDATORY PUBLICATION LOGGING: Client Drift
        drifts = []
        for _, fit_res in results:
            client_ndarrays = fl.common.parameters_to_ndarrays(fit_res.parameters)
            drift = float(np.sqrt(sum(np.sum(np.square(c - g)) for c, g in zip(client_ndarrays, aggregated_ndarrays))))
            drifts.append(drift)
        
        if drifts:
            self.history["client_drift_avg"].append(float(np.mean(drifts)))
            self.history["client_drift_max"].append(float(np.max(drifts)))
            logger.info(f"Round {server_round}: Client drift (avg={np.mean(drifts):.4f}, max={np.max(drifts):.4f})")
        else:
            self.history["client_drift_avg"].append(np.nan)
            self.history["client_drift_max"].append(np.nan)

        # Log global model hash
        model_hash = hash_model(aggregated_ndarrays)
        logger.info(f"Global model hash (round {server_round}): {model_hash}")

        # Save global model checkpoint every round
        try:
            models_dir = os.path.join(os.path.dirname(__file__), "models")
            if models_dir not in sys.path:
                sys.path.insert(0, models_dir)
            model_to_save = get_model(self.model_name, num_classes=self.num_classes, pretrained=True)
            current_state_dict = OrderedDict()
            
            # CRITICAL FIX: Only iterate over trainable parameters (excluding BN running stats)
            # This must match what's in aggregated_parameters
            aggregated_ndarrays_iter = iter(fl.common.parameters_to_ndarrays(aggregated_parameters))
            
            for name, param in model_to_save.state_dict().items():
                # Skip BN running statistics - they're not in aggregated_parameters
                if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                    # Keep original BN stats from pretrained model (they won't be used in FedBN)
                    current_state_dict[name] = param
                    continue
                
                # Load aggregated trainable parameters
                try:
                    arr = next(aggregated_ndarrays_iter)
                    current_state_dict[name] = torch.as_tensor(arr, dtype=param.dtype)
                except StopIteration:
                    logger.error(f"Ran out of aggregated parameters at {name}")
                    raise RuntimeError(f"Parameter count mismatch when saving checkpoint at round {server_round}")
            
            torch.save(current_state_dict, os.path.join(self.results_base_dir, f"global_model_round_{server_round}.pth"))
            logger.info(f"Global model for round {server_round} saved.")
        except Exception as e:
            logger.error(f"Failed to save global model for round {server_round}: {e}", exc_info=True)

        # Summaries from client metrics
        summary = self._calculate_fit_metrics(results)
        
        # Log max activation magnitude from clients
        if "max_activation_avg" in summary:
            self.history["max_activation_magnitude"].append(summary["max_activation_avg"])
            logger.info(f"Round {server_round}: Max activation magnitude (avg across clients) = {summary['max_activation_avg']:.4f}")
        else:
            self.history["max_activation_magnitude"].append(np.nan)

        # Bug fix: Append cnn_drift and transformer_drift to history
        self.history["cnn_drift_avg"].append(summary.get("cnn_drift_avg", np.nan))
        self.history["transformer_drift_avg"].append(summary.get("transformer_drift_avg", np.nan))

        # Bug fix: Append xai_cross_method_agreement to history
        self.history["xai_cross_method_agreement_mean"].append(
            summary.get("xai_cross_method_agreement_mean_avg", np.nan))

        # Cross-client XAI stability (shared probe set)
        cam_stacks = []
        for _, fit_res in results:
            if fit_res.metrics is None:
                continue
            payload = fit_res.metrics.get("xai_cam_stack_b64", "")
            stack = decode_cam_stack(payload)
            if stack is not None:
                cam_stacks.append(stack)

        if len(cam_stacks) >= 2:
            min_samples = min(s.shape[0] for s in cam_stacks)
            pair_ssim = []
            pair_pearson = []
            for i in range(len(cam_stacks)):
                for j in range(i + 1, len(cam_stacks)):
                    for s_idx in range(min_samples):
                        sim = compute_cam_similarity(cam_stacks[i][s_idx], cam_stacks[j][s_idx])
                        pair_ssim.append(sim["ssim"])
                        pair_pearson.append(sim["pearson"])
            summary["xai_client_stability_mean"] = float(np.nanmean(pair_ssim)) if pair_ssim else float("nan")
            summary["xai_client_stability_std"] = float(np.nanstd(pair_ssim)) if pair_ssim else float("nan")
            summary["xai_client_pearson_mean"] = float(np.nanmean(pair_pearson)) if pair_pearson else float("nan")
            summary["xai_client_pearson_std"] = float(np.nanstd(pair_pearson)) if pair_pearson else float("nan")
        else:
            summary["xai_client_stability_mean"] = float("nan")
            summary["xai_client_stability_std"] = float("nan")
            summary["xai_client_pearson_mean"] = float("nan")
            summary["xai_client_pearson_std"] = float("nan")
        self.history["round"].append(server_round)
        self.history["train_loss"].append(summary["train_loss_avg"])
        self.history["train_accuracy"].append(summary["train_accuracy_avg"])
        self.history["train_f1"].append(summary["train_f1_avg"])
        self.history["val_loss"].append(summary["val_loss_avg"])
        self.history["val_accuracy"].append(summary["val_accuracy_avg"])
        self.history["val_f1"].append(summary["val_f1_avg"])
        self.history["num_clients"].append(len(results))
        self.history["client_data_sizes"].append(summary["client_data_sizes"])
        self.history["aggregation_time"].append(time.time() - t0)

        # XAI history (only if keys exist in summary)
        self.history["xai_del_auc_mean"].append(summary.get("xai_del_auc_mean_avg", np.nan))
        self.history["xai_del_auc_std"].append(summary.get("xai_del_auc_std_avg", np.nan))
        self.history["xai_heat_in_mask_mean"].append(summary.get("xai_heat_in_mask_mean_avg", np.nan))
        self.history["xai_heat_in_mask_std"].append(summary.get("xai_heat_in_mask_std_avg", np.nan))
        self.history["xai_ins_auc_mean"].append(summary.get("xai_ins_auc_mean_avg", np.nan))
        self.history["xai_ins_auc_std"].append(summary.get("xai_ins_auc_std_avg", np.nan))
        self.history["xai_cam_consistency_mean"].append(summary.get("xai_cam_consistency_mean_avg", np.nan))
        self.history["xai_cam_consistency_std"].append(summary.get("xai_cam_consistency_std_avg", np.nan))
        self.history["xai_temporal_stability_mean"].append(summary.get("xai_temporal_stability_mean_avg", np.nan))
        self.history["xai_temporal_stability_std"].append(summary.get("xai_temporal_stability_std_avg", np.nan))
        self.history["xai_temporal_pearson_mean"].append(summary.get("xai_temporal_pearson_mean_avg", np.nan))
        self.history["xai_temporal_pearson_std"].append(summary.get("xai_temporal_pearson_std_avg", np.nan))
        self.history["xai_client_stability_mean"].append(summary.get("xai_client_stability_mean", np.nan))
        self.history["xai_client_stability_std"].append(summary.get("xai_client_stability_std", np.nan))
        self.history["xai_client_pearson_mean"].append(summary.get("xai_client_pearson_mean", np.nan))
        self.history["xai_client_pearson_std"].append(summary.get("xai_client_pearson_std", np.nan))

        # Track best by validation F1
        if summary["val_f1_avg"] > self.best_f1:
            self.best_f1 = summary["val_f1_avg"]
            self.best_accuracy = summary["val_accuracy_avg"]
            self.best_round = server_round
            self.best_parameters = aggregated_parameters
            self.save_best_model()
            logger.info(
                f"🏆 New best model: round={self.best_round}, val_f1={self.best_f1:.4f}, val_acc={self.best_accuracy:.4f}"
            )

        aggregated_metrics.update({k: v for k, v in summary.items() if k.startswith("xai_")})
        aggregated_metrics["aggregation_time"] = self.history["aggregation_time"][-1]
        aggregated_metrics["mean_client_drift"] = summary["client_drift_avg"]

        # Server-side global model evaluation (on validation set for convergence monitoring)
        global_val_metrics = self.evaluate_global_set(aggregated_parameters, self.global_val_loader, "validation")
        if "accuracy" in global_val_metrics:
            self.history["global_val_accuracy"].append(global_val_metrics["accuracy"]) # New history item
            aggregated_metrics["global_val_accuracy"] = global_val_metrics["accuracy"] # New aggregated metric

        # Bug fix: Also evaluate on global test set per round
        global_test_metrics = self.evaluate_global_set(aggregated_parameters, self.global_test_loader, "test")
        if "accuracy" in global_test_metrics:
            self.history["global_test_accuracy"].append(global_test_metrics["accuracy"])
            aggregated_metrics["global_test_accuracy"] = global_test_metrics["accuracy"]

        # Round-wise convergence tracking
        self.round_history.append({
            "round": server_round,
            "val_accuracy": summary["val_accuracy_avg"],
            "val_loss": summary["val_loss_avg"],
            "auc": summary.get("xai_del_auc_mean_avg", np.nan), # Use Del AUC as 'auc' for simplicity
        })


        # Communication cost tracking
        total_bytes_sent = sum(fit_res.metrics.get("communication_bytes_sent", 0) for _, fit_res in results if fit_res.metrics)
        total_bytes_received = sum(fit_res.metrics.get("communication_bytes_received", 0) for _, fit_res in results if fit_res.metrics)
        self.communication_history.append({
            "round": server_round,
            "bytes_sent": total_bytes_sent,
            "bytes_received": total_bytes_received,
            "total_bytes": total_bytes_sent + total_bytes_received,
        })
        logger.info(f"Communication cost (round {server_round}): "
                    f"Sent: {total_bytes_sent / (1024*1024):.2f} MB, "
                    f"Received: {total_bytes_received / (1024*1024):.2f} MB, "
                    f"Total: {(total_bytes_sent + total_bytes_received) / (1024*1024):.2f} MB")

        # Bug fix: Store last_parameters every round for save_last_model()
        self.last_parameters = aggregated_parameters

        self._log_round_summary(server_round, summary, len(results), global_val_metrics)

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, fl.common.Scalar]]:
        logger.info(
            f"Round {server_round}: aggregating evaluation results (success={len(results)} fail={len(failures)})"
        )
        if not results:
            # Append NaN placeholders to keep test metrics aligned with round numbers
            self.history["test_loss"].append(float("nan"))
            self.history["test_accuracy"].append(float("nan"))
            self.history["test_f1"].append(float("nan"))
            return None, {}
        test = self._calculate_eval_metrics(results)
        if len(self.history["test_loss"]) < len(self.history["round"]):
            self.history["test_loss"].append(test["test_loss_avg"])
            self.history["test_accuracy"].append(test["test_accuracy_avg"])
            self.history["test_f1"].append(test["test_f1_avg"])
        logger.info(
            f"   Test: loss={test['test_loss_avg']:.4f} acc={test['test_accuracy_avg']:.4f} f1={test['test_f1_avg']:.4f}"
        )

        # Periodic snapshot (moved here from aggregate_fit so test metrics are included)
        if server_round % 10 == 0:
            self.save_intermediate_results(server_round)

        return test["test_loss_avg"], test

    def _calculate_fit_metrics(self, results):
        total_examples = sum(fit_res.num_examples for _, fit_res in results) or 1
        metric_keys = [
            "train_loss", "train_accuracy", "train_f1",
            "val_loss", "val_accuracy", "val_f1",
            "xai_del_auc_mean", "xai_del_auc_std",
            "xai_heat_in_mask_mean", "xai_heat_in_mask_std",
            "xai_ins_auc_mean", "xai_ins_auc_std",
            "xai_cam_consistency_mean", "xai_cam_consistency_std",
            "xai_temporal_stability_mean", "xai_temporal_stability_std",
            "xai_temporal_pearson_mean", "xai_temporal_pearson_std",
            "client_drift",
            "cnn_drift", # Added
            "transformer_drift", # Added
            "max_activation",
            "xai_cross_method_agreement_mean",
        ]

        weighted_sums = dict.fromkeys(metric_keys, 0.0)
        present = dict.fromkeys(metric_keys, False)
        client_data_sizes, client_metrics_list = [], []
        
        client_drifts_list = [] # New list to store client drifts

        for client_proxy, fit_res in results:
            weight = fit_res.num_examples / total_examples
            client_data_sizes.append(fit_res.num_examples)
            metrics_for_client = {}
            for key in metric_keys:
                val = fit_res.metrics.get(key, None) if fit_res.metrics else None
                if isinstance(val, (int, float, np.integer, np.floating)):
                    weighted_sums[key] += float(val) * weight
                    present[key] = True
                    metrics_for_client[key] = float(val)
            # Collect client_drift specifically
            if "client_drift" in fit_res.metrics and isinstance(fit_res.metrics["client_drift"], (int, float, np.integer, np.floating)):
                client_drifts_list.append(float(fit_res.metrics["client_drift"]))

            client_metrics_list.append({
                "client_id": client_proxy.cid,
                "num_examples": fit_res.num_examples,
                "metrics": metrics_for_client,
            })
            # Store client-wise accuracy
            self.client_accuracy_history[client_proxy.cid].append(
                float(fit_res.metrics.get("val_accuracy", 0.0))
            )

        current_round = len(self.history["round"]) + 1
        self.client_metrics_history[current_round] = client_metrics_list

        out = {
            "train_loss_avg": weighted_sums["train_loss"],
            "train_accuracy_avg": weighted_sums["train_accuracy"],
            "train_f1_avg": weighted_sums["train_f1"],
            "val_loss_avg": weighted_sums["val_loss"],
            "val_accuracy_avg": weighted_sums["val_accuracy"],
            "val_f1_avg": weighted_sums["val_f1"],
            "total_examples": total_examples,
            "client_data_sizes": client_data_sizes,
            "num_participating_clients": len(results),
            "client_drift_avg": np.mean(client_drifts_list) if client_drifts_list else 0.0, # New entry
        }
        if present["xai_del_auc_mean"]:
            out["xai_del_auc_mean_avg"] = weighted_sums["xai_del_auc_mean"]
            out["xai_del_auc_std_avg"]  = weighted_sums["xai_del_auc_std"]
        if present["xai_heat_in_mask_mean"]:
            out["xai_heat_in_mask_mean_avg"] = weighted_sums["xai_heat_in_mask_mean"]
            out["xai_heat_in_mask_std_avg"]  = weighted_sums["xai_heat_in_mask_std"]
        if present["xai_ins_auc_mean"]:
            out["xai_ins_auc_mean_avg"] = weighted_sums["xai_ins_auc_mean"]
            out["xai_ins_auc_std_avg"]  = weighted_sums["xai_ins_auc_std"]
        if present["xai_cam_consistency_mean"]:
            out["xai_cam_consistency_mean_avg"] = weighted_sums["xai_cam_consistency_mean"]
            out["xai_cam_consistency_std_avg"]  = weighted_sums["xai_cam_consistency_std"]
        if present["xai_temporal_stability_mean"]:
            out["xai_temporal_stability_mean_avg"] = weighted_sums["xai_temporal_stability_mean"]
            out["xai_temporal_stability_std_avg"]  = weighted_sums["xai_temporal_stability_std"]
        if present["xai_temporal_pearson_mean"]:
            out["xai_temporal_pearson_mean_avg"] = weighted_sums["xai_temporal_pearson_mean"]
            out["xai_temporal_pearson_std_avg"]  = weighted_sums["xai_temporal_pearson_std"]
        if present["cnn_drift"]: # If present in any client metrics
            out["cnn_drift_avg"] = weighted_sums["cnn_drift"]
        if present["transformer_drift"]: # If present in any client metrics
            out["transformer_drift_avg"] = weighted_sums["transformer_drift"]
        if present["max_activation"]:
            out["max_activation_avg"] = weighted_sums["max_activation"]
        if present["xai_cross_method_agreement_mean"]:
            out["xai_cross_method_agreement_mean_avg"] = weighted_sums["xai_cross_method_agreement_mean"]
        return out

    def _calculate_eval_metrics(
        self, results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]]
    ) -> dict:
        total_examples = sum(eval_res.num_examples for _, eval_res in results) or 1
        weighted_loss = 0.0
        weighted_accuracy = 0.0
        weighted_f1 = 0.0

        for _, eval_res in results:
            weight = eval_res.num_examples / total_examples
            weighted_loss += float(eval_res.loss or 0.0) * weight
            if eval_res.metrics:
                weighted_accuracy += float(eval_res.metrics.get("accuracy", 0.0)) * weight
                weighted_f1 += float(eval_res.metrics.get("f1_macro", 0.0)) * weight

        return {
            "test_loss_avg": weighted_loss,
            "test_accuracy_avg": weighted_accuracy,
            "test_f1_avg": weighted_f1,
            "total_test_examples": total_examples,
            "num_eval_clients": len(results),
        }

    def _log_round_summary(self, round_num: int, summary: dict, num_clients: int, global_eval_metrics: dict = None):
        logger.info("=" * 80)
        logger.info(f"ROUND {round_num} SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Clients: {num_clients} | examples: {summary['total_examples']:,}")
        logger.info(
            f"Train: loss={summary['train_loss_avg']:.4f} "
            f"acc={summary['train_accuracy_avg']:.4f} f1={summary['train_f1_avg']:.4f}"
        )
        logger.info(
            f"Val  : loss={summary['val_loss_avg']:.4f} "
            f"acc={summary['val_accuracy_avg']:.4f} f1={summary['val_f1_avg']:.4f}"
        )
        logger.info(f"Best : round={self.best_round} val_f1={self.best_f1:.4f} val_acc={self.best_accuracy:.4f}")

        xai_mean = summary.get("xai_del_auc_mean_avg", None)
        if xai_mean is not None:
            logger.info(
                f"XAI  : delAUC={xai_mean:.4f} "
                f"mask_in={summary.get('xai_heat_in_mask_mean_avg','NA')}"
            )
        xai_temp = summary.get("xai_temporal_stability_mean_avg", None)
        if xai_temp is not None:
            logger.info(f"XAI  : temporal_ssim={xai_temp:.4f}")
        xai_client = summary.get("xai_client_stability_mean", None)
        if xai_client is not None:
            logger.info(f"XAI  : client_ssim={xai_client:.4f}")

        if global_eval_metrics and "accuracy" in global_eval_metrics:
            logger.info(f"Global Validation Accuracy: {global_eval_metrics['accuracy']:.4f}")

        # Log client drift if available
        client_drift_avg = summary.get("client_drift_avg")
        if client_drift_avg is not None:
            logger.info(f"Client drift avg: {client_drift_avg:.4f}")

        # Log communication cost for the current round
        current_comm_data = next((item for item in self.communication_history if item["round"] == round_num), None)
        if current_comm_data:
            total_mb = current_comm_data["total_bytes"] / (1024 * 1024)
            sent_mb = current_comm_data["bytes_sent"] / (1024 * 1024)
            received_mb = current_comm_data["bytes_received"] / (1024 * 1024)
            logger.info(f"Comm : Sent={sent_mb:.2f} MB, Received={received_mb:.2f} MB, Total={total_mb:.2f} MB")

        logger.info("=" * 80)

    def _recalibrate_global_batchnorm(self, loader: DataLoader, num_batches: int = 100) -> None:
        """Gently adapt BN running stats on server using momentum-based blending.
        
        CRITICAL FIX: Do NOT call reset_running_stats(). In FL, the aggregated
        model carries BN stats from client training data. Resetting them and
        recomputing from the small server validation set causes model collapse
        (all predictions collapse to a single class).
        
        Instead, we use PyTorch's default momentum (0.1) to *blend* server data
        statistics into the existing client-aggregated stats. This prevents
        catastrophic stat replacement while still adapting slightly to the
        server's data distribution for more representative evaluation.
        """
        bn_layers = [
            m for m in self.global_model_instance.modules()
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d))
        ]
        if not bn_layers:
            return

        # Save the aggregated BN stats so we can verify stability
        saved_stats = {}
        for name, m in self.global_model_instance.named_modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                saved_stats[name] = {
                    'running_mean': m.running_mean.clone(),
                    'running_var': m.running_var.clone(),
                }

        # Set model to eval mode first, then ONLY set BN layers to train
        # This allows BN layers to update running stats using their momentum
        # WITHOUT resetting them (default momentum=0.1 means gentle blending)
        self.global_model_instance.eval()
        for m in bn_layers:
            # DO NOT call m.reset_running_stats() — this destroys client stats
            m.train()  # enables running stat updates via momentum

        seen = 0
        nan_batches = 0
        with torch.no_grad():
            for data, _ in loader:
                data = data.to(self.global_model_device).float()
                try:
                    _ = self.global_model_instance(data)
                except RuntimeError as e:
                    if "Numerical instability" in str(e):
                        nan_batches += 1
                        if nan_batches > 5:
                            logger.warning("Server: Too many NaN batches during BN recalibration, aborting")
                            # Restore original stats if recalibration fails
                            for bname, m in self.global_model_instance.named_modules():
                                if bname in saved_stats:
                                    m.running_mean.copy_(saved_stats[bname]['running_mean'])
                                    m.running_var.copy_(saved_stats[bname]['running_var'])
                            break
                        continue
                    raise
                seen += 1
                if seen >= num_batches:
                    break

        self.global_model_instance.eval()
        
        # Verify no collapse: check if BN stats contain NaN/Inf
        for name, m in self.global_model_instance.named_modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                if not torch.isfinite(m.running_mean).all() or not torch.isfinite(m.running_var).all():
                    logger.warning(f"Server: BN layer {name} has non-finite stats after recalibration, "
                                   f"restoring original aggregated stats")
                    if name in saved_stats:
                        m.running_mean.copy_(saved_stats[name]['running_mean'])
                        m.running_var.copy_(saved_stats[name]['running_var'])

        logger.info(f"Server: BatchNorm adapted (momentum-blend) using {seen} batches on global validation data"
                    + (f" ({nan_batches} batches had NaN)" if nan_batches else ""))

    def evaluate_global_set(self, parameters: fl.common.Parameters, data_loader: DataLoader, dataset_name: str) -> dict[str, float]:
        """Evaluate the aggregated global model on a given server-side dataset."""
        if data_loader is None:
            logger.warning(f"No global {dataset_name} loader available, skipping server-side global model evaluation.")
            return {}

        logger.info(f"Server: Evaluating aggregated global model on global {dataset_name} set...")

        # CRITICAL FIX: Exclude BN stats when loading parameters
        global_state_dict = OrderedDict()
        aggregated_ndarrays_iter = iter(fl.common.parameters_to_ndarrays(parameters))
        
        for name, param in self.global_model_instance.state_dict().items():
            # Skip BN running statistics - they're not in aggregated_parameters
            if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                # Keep existing BN stats
                global_state_dict[name] = param
                continue
            
            # Load aggregated trainable parameters
            try:
                arr = next(aggregated_ndarrays_iter)
                global_state_dict[name] = torch.as_tensor(arr, dtype=param.dtype)
            except StopIteration:
                logger.error(f"Ran out of aggregated parameters at {name}")
                raise RuntimeError(f"Parameter count mismatch in evaluate_global_set")
        
        self.global_model_instance.load_state_dict(global_state_dict, strict=False)
        self.global_model_instance.to(self.global_model_device)

        # FedBN stabilization: gently adapt BN stats using momentum-blend with server data.
        # Use all available batches for stable statistics (not just 20).
        try:
            total_batches = len(data_loader)
            self._recalibrate_global_batchnorm(data_loader, num_batches=total_batches)
        except Exception as exc:
            logger.warning(f"Server: BN recalibration skipped due to {type(exc).__name__}: {exc}")

        # Perform evaluation using ModelTrainer's evaluate method
        try:
            metrics = self.global_model_trainer.evaluate(
                data_loader,
                save_name=f"global_round_{self.history['round'][-1] if self.history['round'] else 0}_{dataset_name}_matrix.png"
            )
        except RuntimeError as e:
            logger.error(f"Server: Global model evaluation failed on {dataset_name}: {e}")
            logger.warning("Server: Returning default metrics due to evaluation failure (likely NaN/Inf from aggregation)")
            metrics = {"accuracy": 0.0, "f1": 0.0, "loss": float("inf")}

        logger.info(f"Server: Global model evaluation completed on {dataset_name}. Accuracy: {metrics.get('accuracy', 0.0):.4f}")
        return metrics

    def save_best_model(self) -> None:
        if self.best_parameters is None:
            logger.warning("No best parameters available, skipping model save.")
            return
        try:
            models_dir = os.path.join(os.path.dirname(__file__), "models")
            if models_dir not in sys.path:
                sys.path.insert(0, models_dir)

            model = get_model(self.model_name, num_classes=self.num_classes, pretrained=True)

            # CRITICAL FIX: Exclude BN stats when loading parameters
            best_state_dict = OrderedDict()
            best_ndarrays_iter = iter(fl.common.parameters_to_ndarrays(self.best_parameters))
            
            for name, param in model.state_dict().items():
                # Skip BN running statistics - they're not in best_parameters
                if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                    # Keep pretrained BN stats
                    best_state_dict[name] = param
                    continue
                
                # Load best trainable parameters
                try:
                    arr = next(best_ndarrays_iter)
                    best_state_dict[name] = torch.as_tensor(arr, dtype=param.dtype)
                except StopIteration:
                    logger.error(f"Ran out of best parameters at {name}")
                    raise RuntimeError("Parameter count mismatch when saving best model")

            checkpoint = {
                "round": self.best_round,
                "model_state_dict": best_state_dict,
                "best_f1": self.best_f1,
                "best_accuracy": self.best_accuracy,
                "model_name": self.model_name,
                "num_classes": self.num_classes,
                "timestamp": datetime.now().isoformat(),
            }

            save_path = os.path.join(self.results_base_dir, f"best_model_round_{self.best_round}.pth")
            torch.save(checkpoint, save_path)
            logger.info(f"Best model saved successfully → {save_path}")

        except Exception as exc:
            logger.error(f"Failed to save best model: {exc}", exc_info=True)

    def save_last_model(self) -> None:
        if self.last_parameters is None:
            logger.warning("No last parameters available, skipping model save.")
            return
        try:
            models_dir = os.path.join(os.path.dirname(__file__), "models")
            if models_dir not in sys.path:
                sys.path.insert(0, models_dir)

            model = get_model(self.model_name, num_classes=self.num_classes, pretrained=True)

            # CRITICAL FIX: Exclude BN stats when loading parameters
            last_state_dict = OrderedDict()
            last_ndarrays_iter = iter(fl.common.parameters_to_ndarrays(self.last_parameters))
            
            for name, param in model.state_dict().items():
                # Skip BN running statistics - they're not in last_parameters
                if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                    # Keep pretrained BN stats
                    last_state_dict[name] = param
                    continue
                
                # Load last trainable parameters
                try:
                    arr = next(last_ndarrays_iter)
                    last_state_dict[name] = torch.as_tensor(arr, dtype=param.dtype)
                except StopIteration:
                    logger.error(f"Ran out of last parameters at {name}")
                    raise RuntimeError("Parameter count mismatch when saving last model")

            checkpoint = {
                "model_state_dict": last_state_dict,
                "model_name": self.model_name,
                "num_classes": self.num_classes,
                "timestamp": datetime.now().isoformat(),
            }

            save_path = os.path.join(self.results_base_dir, "last_global_model.pth")
            torch.save(checkpoint, save_path)
            logger.info(f"Last global model saved successfully → {save_path}")

        except Exception as exc:
            logger.error(f"Failed to save last model: {exc}", exc_info=True)

    def save_intermediate_results(self, rnd: int):
        try:
            _atomic_json_dump(self.history,
                              os.path.join(self.results_base_dir, f"history_round_{rnd}.json"))
            _atomic_json_dump(self.client_metrics_history,
                              os.path.join(self.results_base_dir, f"client_metrics_round_{rnd}.json"))
            self.plot_training_curves(save_suffix=f"_round_{rnd}")
            logger.info(f"Intermediate results saved for round {rnd}")
        except Exception as e:
            logger.error(f"Failed to save intermediate results: {e}", exc_info=True)

    def save_final_results(self):
        try:
            _atomic_json_dump(self.history,
                              os.path.join(self.results_base_dir, "final_training_history.json"))
            _atomic_json_dump(self.client_metrics_history,
                              os.path.join(self.results_base_dir, "final_client_metrics.json"))
            _atomic_json_dump(self.round_history,
                              os.path.join(self.results_base_dir, "convergence_history.json"))
            _atomic_json_dump(self.communication_history,
                              os.path.join(self.results_base_dir, "communication_history.json"))
            self.plot_training_curves(save_suffix="_final")
            logger.info(f"Final results saved in {self.results_base_dir}")
        except Exception as e:
            logger.error(f"Failed to save final results: {e}", exc_info=True)

    def plot_training_curves(self, save_suffix: str = ""):
        if not self.history["round"]:
            return
        rounds = self.history["round"]

        # Determine number of subplots dynamically
        has_xai = any(
            key in self.history and any(not np.isnan(v) for v in self.history[key])
            for key in [
                "xai_del_auc_mean",
                "xai_ins_auc_mean",
                "xai_temporal_stability_mean",
                "xai_client_stability_mean",
            ]
        )
        has_global_test = "global_test_accuracy" in self.history and self.history["global_test_accuracy"]
        has_global_val = "global_val_accuracy" in self.history and self.history["global_val_accuracy"]
        has_drift = ("cnn_drift_avg" in self.history and self.history["cnn_drift_avg"]) or \
                    ("transformer_drift_avg" in self.history and self.history["transformer_drift_avg"])

        num_plots = 3  # loss, accuracy, f1
        if has_global_test:
            num_plots += 1
        if has_global_val:
            num_plots += 1
        if has_xai:
            num_plots += 1  # del AUC
            if "xai_ins_auc_mean" in self.history and self.history["xai_ins_auc_mean"]:
                num_plots += 1
            if "xai_temporal_stability_mean" in self.history and self.history["xai_temporal_stability_mean"]:
                num_plots += 1
            if "xai_client_stability_mean" in self.history and self.history["xai_client_stability_mean"]:
                num_plots += 1
        if "cnn_drift_avg" in self.history and self.history["cnn_drift_avg"]:
            num_plots += 1
        if "transformer_drift_avg" in self.history and self.history["transformer_drift_avg"]:
            num_plots += 1
        # Add mandatory plots (clients per round, aggregation time, data distribution)
        num_plots += 1  # Clients per round
        if self.history["aggregation_time"]:
            num_plots += 1  # Aggregation time
        if self.history["client_data_sizes"]:
            num_plots += 1  # Data distribution
        num_cols = 4 if (has_xai or has_global_test or has_global_val or has_drift) else 3
        num_rows = int(math.ceil(num_plots / num_cols))

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
        axes_flat = axes.flatten()
        plot_idx = 0

        def _plot_series(ax, x_vals, y_vals, **kwargs):
            if not y_vals:
                return
            n = min(len(x_vals), len(y_vals))
            if n <= 0:
                return
            ax.plot(x_vals[:n], y_vals[:n], **kwargs)

        # Loss
        _plot_series(axes_flat[plot_idx], rounds, self.history["train_loss"], label="Train Loss", linewidth=2, marker="o")
        _plot_series(axes_flat[plot_idx], rounds, self.history["val_loss"], label="Val Loss", linewidth=2, marker="s")
        _plot_series(axes_flat[plot_idx], rounds, self.history["test_loss"], label="Test Loss", linewidth=2, marker="^")
        axes_flat[plot_idx].set_title("Loss")
        plot_idx += 1

        # Accuracy
        _plot_series(axes_flat[plot_idx], rounds, self.history["train_accuracy"], label="Train Acc", linewidth=2, marker="o")
        _plot_series(axes_flat[plot_idx], rounds, self.history["val_accuracy"], label="Val Acc", linewidth=2, marker="s")
        _plot_series(axes_flat[plot_idx], rounds, self.history["test_accuracy"], label="Test Acc", linewidth=2, marker="^")
        axes_flat[plot_idx].set_title("Accuracy")
        plot_idx += 1

        # F1-Score
        _plot_series(axes_flat[plot_idx], rounds, self.history["train_f1"], label="Train F1", linewidth=2, marker="o")
        _plot_series(axes_flat[plot_idx], rounds, self.history["val_f1"], label="Val F1", linewidth=2, marker="s")
        _plot_series(axes_flat[plot_idx], rounds, self.history["test_f1"], label="Test F1", linewidth=2, marker="^")
        axes_flat[plot_idx].set_title("F1-Score")
        plot_idx += 1

        # Global Test Accuracy
        if has_global_test:
            _plot_series(axes_flat[plot_idx], rounds, self.history["global_test_accuracy"], label="Global Test Acc", linewidth=2, marker="x", color="purple")
            axes_flat[plot_idx].set_title("Global Test Accuracy")
            plot_idx += 1
        
        # Global Validation Accuracy (newly added plot)
        if has_global_val:
            _plot_series(axes_flat[plot_idx], rounds, self.history["global_val_accuracy"], label="Global Val Acc", linewidth=2, marker="P", color="orange")
            axes_flat[plot_idx].set_title("Global Validation Accuracy")
            plot_idx += 1

        # XAI Deletion AUC
        if has_xai:
            _plot_series(axes_flat[plot_idx], rounds, self.history["xai_del_auc_mean"], label="XAI Del AUC", linewidth=2, marker="d", color="green")
            axes_flat[plot_idx].set_title("XAI Deletion AUC")
            plot_idx += 1
            if "xai_ins_auc_mean" in self.history and self.history["xai_ins_auc_mean"]:
                _plot_series(axes_flat[plot_idx], rounds, self.history["xai_ins_auc_mean"], label="XAI Ins AUC", linewidth=2, marker="o", color="teal")
                axes_flat[plot_idx].set_title("XAI Insertion AUC")
                plot_idx += 1
            if "xai_temporal_stability_mean" in self.history and self.history["xai_temporal_stability_mean"]:
                _plot_series(axes_flat[plot_idx], rounds, self.history["xai_temporal_stability_mean"], label="XAI Temporal SSIM", linewidth=2, marker="s", color="orange")
                axes_flat[plot_idx].set_title("XAI Temporal Stability")
                plot_idx += 1
            if "xai_client_stability_mean" in self.history and self.history["xai_client_stability_mean"]:
                _plot_series(axes_flat[plot_idx], rounds, self.history["xai_client_stability_mean"], label="XAI Client SSIM", linewidth=2, marker="^", color="purple")
                axes_flat[plot_idx].set_title("XAI Cross-Client Stability")
                plot_idx += 1

        # CNN Drift
        if "cnn_drift_avg" in self.history and self.history["cnn_drift_avg"]:
            _plot_series(axes_flat[plot_idx], rounds, self.history["cnn_drift_avg"], label="CNN Drift", linewidth=2, marker="^", color="blue")
            axes_flat[plot_idx].set_title("CNN Drift")
            plot_idx += 1

        # Transformer Drift
        if "transformer_drift_avg" in self.history and self.history["transformer_drift_avg"]:
            _plot_series(axes_flat[plot_idx], rounds, self.history["transformer_drift_avg"], label="Transformer Drift", linewidth=2, marker="v", color="red")
            axes_flat[plot_idx].set_title("Transformer Drift")
            plot_idx += 1

        # Clients per round
        axes_flat[plot_idx].bar(rounds, self.history["num_clients"], alpha=0.8, label="Clients")
        axes_flat[plot_idx].set_title("Clients per Round")
        plot_idx += 1

        # Aggregation time
        if self.history["aggregation_time"]:
            _plot_series(axes_flat[plot_idx], rounds, self.history["aggregation_time"], linewidth=2, marker="d", label="Agg Time (s)")
            axes_flat[plot_idx].set_title("Aggregation Time (s)")
            plot_idx += 1

        # Data distribution
        if self.history["client_data_sizes"]:
            latest = self.history["client_data_sizes"][-1]
            labels = [f"C{i+1}" for i in range(len(latest))]
            axes_flat[plot_idx].pie(latest, labels=labels, autopct="%1.1f%%", startangle=90)
            axes_flat[plot_idx].set_title("Data Distribution (latest)")
            plot_idx += 1

        # Style and Legends
        for ax in axes_flat:
            ax.grid(True, alpha=0.3)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="best")

        # Hide unused subplots
        for i in range(plot_idx, len(axes_flat)):
            fig.delaxes(axes_flat[i])


        # --- NEW: Add Super Title ---
        fig.suptitle(f"Federated Learning Training Metrics ({self.model_name})", fontsize=20, y=0.98)
        # Note: Changed y to 0.98 or 1.0 to ensure it doesn't get cropped.
        # If 1.03 cuts off, lower it. If it overlaps plots, raise it or adjust tight_layout padding.

        plt.tight_layout()

        # Save
        out = os.path.join(self.results_base_dir, f"training_curves{save_suffix}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved plot → {out}")

        # Cross-round XAI consistency plot
        if "xai_temporal_stability_mean" in self.history and any(
            not np.isnan(v) for v in self.history["xai_temporal_stability_mean"]
        ):
            xai_out = os.path.join(self.results_base_dir, f"xai_consistency{save_suffix}.png")
            try:
                plot_xai_consistency(self.history["xai_temporal_stability_mean"], xai_out)
                logger.info(f"Saved XAI consistency plot → {xai_out}")
            except Exception as exc:
                logger.warning(f"Failed to save XAI consistency plot: {exc}")

class LoggingClientManager(SimpleClientManager):
    def __init__(self, expected_clients: int):
        super().__init__()
        self.expected_clients = expected_clients

    def register(self, client: ClientProxy) -> bool:
        ok = super().register(client)
        n = self.num_available()
        remaining = max(self.expected_clients - n, 0)
        logger.info(f"Client connected: {client.cid} | connected={n} | waiting={remaining}")
        if n >= self.expected_clients:
            logger.info("Required clients connected. Starting rounds as soon as strategy is ready.")
        return ok

    def unregister(self, client: ClientProxy) -> None:
        super().unregister(client)
        n = self.num_available()
        remaining = max(self.expected_clients - n, 0)
        logger.info(f"Client disconnected: {client.cid} | connected={n} | waiting={remaining}")

def start_waiting_heartbeat(cm: SimpleClientManager, target: int, interval_sec: float = 2.0):
    stop_evt = threading.Event()

    def _loop():
        while not stop_evt.is_set():
            connected = cm.num_available()
            remaining = max(target - connected, 0)
            if remaining <= 0:
                stop_evt.set()
                break
            logger.info(f"⏳ Waiting for clients… connected={connected} | waiting={remaining}")
            time.sleep(interval_sec)

    thr = threading.Thread(target=_loop, daemon=True)
    thr.start()
    return stop_evt

def create_server_strategy(
    *,
    min_clients: int,
    fraction_fit: float,
    fraction_evaluate: float,
    model_name: str = "customcnn",
    num_classes: int,
    num_rounds: int,
    local_epochs: int,
    aggregation: str = "fedavg",
    mu: float = 0.01,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    global_val_data_dir: str | None = None,
    global_final_test_data_dir: str | None = None,
    lsetnet_num_transformer_blocks: int | None = None,
    lsetnet_num_heads: int | None = None,
    lsetnet_ff_dim_multiplier: int | None = None,
) -> MedicalFLStrategy:
    initial_parameters = get_init_parameters(
        model_name,
        num_classes,
        lsetnet_num_transformer_blocks=lsetnet_num_transformer_blocks,
        lsetnet_num_heads=lsetnet_num_heads,
        lsetnet_ff_dim_multiplier=lsetnet_ff_dim_multiplier
    )
    if initial_parameters is None:
        raise RuntimeError("Failed to initialize model parameters")

    # Instantiate the strategy first
    strategy_instance = MedicalFLStrategy(
        model_name=model_name,
        num_classes=num_classes,
        num_rounds=num_rounds,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_clients,
        min_evaluate_clients=1,
        min_available_clients=min_clients,
        # on_fit_config_fn will be set below
        on_evaluate_config_fn=evaluate_config,
        accept_failures=True, # Ensure this is explicitly set if not default, for clarity
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        aggregation=aggregation,
        mu=mu,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        global_val_data_dir=global_val_data_dir,
        global_final_test_data_dir=global_final_test_data_dir,
        lsetnet_num_transformer_blocks=lsetnet_num_transformer_blocks,
        lsetnet_num_heads=lsetnet_num_heads,
        lsetnet_ff_dim_multiplier=lsetnet_ff_dim_multiplier,
    )
    
    # Now assign the instance method to on_fit_config_fn
    # local_epochs needs to be captured in the lambda for the _fit_config method
    strategy_instance.on_fit_config_fn = lambda server_round: strategy_instance._fit_config(server_round, local_epochs)

    return strategy_instance


def main():
    parser = argparse.ArgumentParser("Federated Learning Server (Medical)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=5050, help="Port")
    parser.add_argument("--rounds", type=int, default=20, help="FL rounds")
    parser.add_argument("--min-clients", type=int, default=3, help="Minimum clients per round")
    parser.add_argument("--fraction-fit", type=float, default=1.0, help="Fraction of clients to train each round")
    parser.add_argument("--fraction-evaluate", type=float, default=1.0, help="Fraction of clients to evaluate each round")
    parser.add_argument("--model", type=str, default="resnet50",
                        choices=["resnet50", "hybridmodel", "mobilenetv3", "hybridswin", "densenet121", "LSeTNet", "swin_tiny", "vit", "vit_tiny"])
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--lsetnet-num-transformer-blocks", type=int, default=4,
                        help="Number of TransformerBlocks for LSeTNet model")
    parser.add_argument("--lsetnet-num-heads", type=int, default=8,
                        help="Number of attention heads for LSeTNet model")
    parser.add_argument("--lsetnet-ff-dim-multiplier", type=int, default=4,
                        help="FFN dimension multiplier for LSeTNet TransformerBlocks")
    parser.add_argument("--local-epochs", type=int, default=3, help="Local epochs per round")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--expected-clients", type=int, default=None,
                        help="How many clients you expect to connect (for logs). Defaults to --min-clients.")
    parser.add_argument("--aggregation", type=str, default="fedavg",
                        choices=["fedavg", "fedprox", "trimmed_mean"], help="Aggregation method: fedavg, fedprox, or trimmed_mean")
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx mu parameter (proximal term weight)")
    parser.add_argument("--learning-rate", type=float, default=0.0003, help="Learning rate for clients (tuned for batch_size=16, 3 IID clients)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--global-val-data-dir", type=str, default=None,
                        help="Path to global validation data directory for server-side convergence monitoring (optional)")
    parser.add_argument("--global-final-test-data-dir", type=str, default=None,
                        help="Path to global final test data directory for final server-side evaluation (optional)")
    args = parser.parse_args()

    # Auto-detect global validation/test dirs when not provided explicitly.
    if args.global_val_data_dir is None:
        candidate_val = os.path.abspath(os.path.join("Federated_Dataset", "val"))
        if os.path.isdir(candidate_val):
            args.global_val_data_dir = candidate_val
            logger.info(f"Auto-detected --global-val-data-dir: {candidate_val}")

    if args.global_final_test_data_dir is None:
        candidate_test = os.path.abspath(os.path.join("Federated_Dataset", "test"))
        if os.path.isdir(candidate_test):
            args.global_final_test_data_dir = candidate_test
            logger.info(f"Auto-detected --global-final-test-data-dir: {candidate_test}")

    if args.expected_clients is None:
        args.expected_clients = args.min_clients

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        force=True,
    )
    # SECTION 12 — RTX 4060 OPTIMIZATION
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    strategy = None # Initialize strategy to None

    logger.info("Starting Federated Learning Server")
    logger.info("=" * 80)
    logger.info(f"Host={args.host}  Port={args.port}  Rounds={args.rounds}")
    logger.info(f"Model={args.model}  NumClasses={args.num_classes}")
    logger.info(f"MinClients={args.min_clients}  FitFrac={args.fraction_fit}  EvalFrac={args.fraction_evaluate}")
    logger.info(f"Debug mode: {'enabled' if args.debug else 'disabled'}")
    logger.info("=" * 80)

    try:
        strategy = create_server_strategy(
            min_clients=args.min_clients,
            fraction_fit=args.fraction_fit,
            fraction_evaluate=args.fraction_evaluate,
            model_name=args.model,
            num_classes=args.num_classes,
            num_rounds=args.rounds,
            aggregation=args.aggregation,
            mu=args.mu,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            local_epochs=args.local_epochs,
            global_val_data_dir=args.global_val_data_dir,
            global_final_test_data_dir=args.global_final_test_data_dir,
            lsetnet_num_transformer_blocks=args.lsetnet_num_transformer_blocks,
            lsetnet_num_heads=args.lsetnet_num_heads,
            lsetnet_ff_dim_multiplier=args.lsetnet_ff_dim_multiplier,
        )

        # Create custom client manager for connection logging
        client_manager = LoggingClientManager(expected_clients=args.expected_clients)
        
        # Start heartbeat thread to log waiting status
        hb_stop = start_waiting_heartbeat(client_manager, args.expected_clients)

        fl.server.start_server(
            server_address=f"0.0.0.0:{args.port}",
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
            client_manager=client_manager,  # ← CRITICAL: Use custom client manager
            grpc_max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        )

        try:
            hb_stop.set()
        except Exception:
            pass
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        try:
            # Persist results
            strategy.save_final_results()
            strategy.save_last_model()
            if strategy.history["round"]:
                logger.info("\nExperiment finished")
                logger.info(f"Results: {strategy.results_base_dir}")
                logger.info(
                    f"Best round: {strategy.best_round}  "
                    f"Acc={strategy.best_accuracy:.4f}  "
                    f"F1={strategy.best_f1:.4f}"
                )
        except Exception:
            pass


if __name__ == "__main__":
    try:
        import multiprocessing as mp
    except RuntimeError:
        pass
    main()

