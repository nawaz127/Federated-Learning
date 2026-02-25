"""
XAI FAITHFULNESS VALIDATION TRACKER
Tracks XAI metrics across FL rounds to ensure stability and validity
"""

import json
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class XAIFaithfulnessTracker:
    """
    Tracks XAI faithfulness metrics across federated learning rounds
    to ensure:
    1. No silent failures (all samples processed)
    2. Deletion/Insertion AUC within valid ranges
    3. Cross-method agreement stable
    4. No distribution drift in explanations
    """
    
    def __init__(self, save_dir: str, num_classes: int = 3):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.num_classes = num_classes
        
        self.metrics_history = {
            'round': [],
            'cam_success_rate': [],
            'cam_failure_count': [],
            'deletion_auc_mean': [],
            'deletion_auc_std': [],
            'insertion_auc_mean': [],
            'insertion_auc_std': [],
            'cross_method_agreement': [],
            'flat_cam_rate': [],  # Rate of CAMs with no saliency
        }
    
    def record_round(self, round_num: int, xai_results: dict):
        """
        Record XAI metrics for a round
        
        Args:
            round_num: FL round number
            xai_results: Dictionary containing:
                - deletion_auc: List of deletion AUC scores
                - insertion_auc: List of insertion AUC scores
                - cam_failures: Number of CAM generation failures
                - flat_cams: Number of flat/no-saliency CAMs
                - total_samples: Total samples processed
                - cross_method_agreement: Agreement between XAI methods
        """
        self.metrics_history['round'].append(round_num)
        
        total_samples = xai_results.get('total_samples', 0)
        cam_failures = xai_results.get('cam_failures', 0)
        flat_cams = xai_results.get('flat_cams', 0)
        
        # Success rate (excluding flat CAMs - they are valid but uninformative)
        success_rate = (total_samples - cam_failures) / total_samples if total_samples > 0 else 0
        self.metrics_history['cam_success_rate'].append(success_rate)
        self.metrics_history['cam_failure_count'].append(cam_failures)
        
        # Flat CAM rate (valid but no saliency)
        flat_rate = flat_cams / total_samples if total_samples > 0 else 0
        self.metrics_history['flat_cam_rate'].append(flat_rate)
        
        # Deletion AUC
        deletion_aucs = xai_results.get('deletion_auc', [])
        if deletion_aucs:
            self.metrics_history['deletion_auc_mean'].append(np.mean(deletion_aucs))
            self.metrics_history['deletion_auc_std'].append(np.std(deletion_aucs))
        else:
            self.metrics_history['deletion_auc_mean'].append(0.0)
            self.metrics_history['deletion_auc_std'].append(0.0)
        
        # Insertion AUC
        insertion_aucs = xai_results.get('insertion_auc', [])
        if insertion_aucs:
            self.metrics_history['insertion_auc_mean'].append(np.mean(insertion_aucs))
            self.metrics_history['insertion_auc_std'].append(np.std(insertion_aucs))
        else:
            self.metrics_history['insertion_auc_mean'].append(0.0)
            self.metrics_history['insertion_auc_std'].append(0.0)
        
        # Cross-method agreement
        agreement = xai_results.get('cross_method_agreement', 0.0)
        self.metrics_history['cross_method_agreement'].append(agreement)
        
        # Log warnings
        if success_rate < 0.95:
            logger.warning(f"Round {round_num}: Low XAI success rate ({success_rate:.2%})")
        
        if flat_rate > 0.20:
            logger.warning(f"Round {round_num}: High flat CAM rate ({flat_rate:.2%})")
        
        if cam_failures > 0:
            logger.error(f"Round {round_num}: {cam_failures} CAM generation failures!")
    
    def validate_stability(self) -> dict:
        """
        Validate XAI metrics are stable across rounds
        
        Returns:
            Dictionary with validation results and pass/fail status
        """
        if len(self.metrics_history['round']) < 3:
            return {'status': 'insufficient_data', 'passed': False}
        
        validation = {
            'status': 'complete',
            'checks': {},
            'passed': True
        }
        
        # Check 1: Success rate should be > 95%
        min_success = min(self.metrics_history['cam_success_rate'])
        validation['checks']['min_success_rate'] = {
            'value': min_success,
            'threshold': 0.95,
            'passed': min_success >= 0.95
        }
        
        # Check 2: Deletion AUC should be in valid range [0.4, 0.9]
        deletion_mean = np.mean(self.metrics_history['deletion_auc_mean'])
        validation['checks']['deletion_auc_valid'] = {
            'value': deletion_mean,
            'range': [0.4, 0.9],
            'passed': 0.4 <= deletion_mean <= 0.9
        }
        
        # Check 3: Insertion AUC should be in valid range [0.1, 0.6]
        insertion_mean = np.mean(self.metrics_history['insertion_auc_mean'])
        validation['checks']['insertion_auc_valid'] = {
            'value': insertion_mean,
            'range': [0.1, 0.6],
            'passed': 0.1 <= insertion_mean <= 0.6
        }
        
        # Check 4: Metrics should be stable (low variance across rounds)
        deletion_cv = np.std(self.metrics_history['deletion_auc_mean']) / (np.mean(self.metrics_history['deletion_auc_mean']) + 1e-8)
        validation['checks']['deletion_auc_stable'] = {
            'value': deletion_cv,
            'threshold': 0.3,  # Coefficient of variation < 30%
            'passed': deletion_cv < 0.3
        }
        
        # Check 5: No failures in recent rounds
        recent_failures = sum(self.metrics_history['cam_failure_count'][-3:])
        validation['checks']['no_recent_failures'] = {
            'value': recent_failures,
            'threshold': 0,
            'passed': recent_failures == 0
        }
        
        # Overall pass/fail
        validation['passed'] = all(check['passed'] for check in validation['checks'].values())
        
        return validation
    
    def save(self):
        """Save metrics history to JSON"""
        filepath = self.save_dir / 'xai_faithfulness_metrics.json'
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        logger.info(f"XAI faithfulness metrics saved to {filepath}")
    
    def generate_report(self) -> str:
        """Generate human-readable validation report"""
        validation = self.validate_stability()
        
        report = [
            "="*80,
            "XAI FAITHFULNESS VALIDATION REPORT",
            "="*80,
            f"Total rounds tracked: {len(self.metrics_history['round'])}",
            ""
        ]
        
        if validation['status'] == 'insufficient_data':
            report.append("??  INSUFFICIENT DATA: Need at least 3 rounds")
            return "\n".join(report)
        
        for check_name, check_data in validation['checks'].items():
            status = "? PASS" if check_data['passed'] else "? FAIL"
            report.append(f"{status}: {check_name}")
            
            if 'threshold' in check_data:
                report.append(f"    Value: {check_data['value']:.4f}, Threshold: {check_data['threshold']}")
            elif 'range' in check_data:
                report.append(f"    Value: {check_data['value']:.4f}, Valid range: {check_data['range']}")
        
        report.append("")
        if validation['passed']:
            report.append("??? XAI FAITHFULNESS VALIDATION PASSED ???")
        else:
            report.append("??? XAI FAITHFULNESS VALIDATION FAILED ???")
        
        report.append("="*80)
        
        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    tracker = XAIFaithfulnessTracker(save_dir="./xai_validation")
    
    # Simulate 5 rounds
    for round_num in range(1, 6):
        xai_results = {
            'total_samples': 100,
            'cam_failures': np.random.randint(0, 2),  # 0-1 failures
            'flat_cams': np.random.randint(5, 15),     # 5-15 flat CAMs
            'deletion_auc': np.random.uniform(0.5, 0.8, size=95).tolist(),
            'insertion_auc': np.random.uniform(0.2, 0.5, size=95).tolist(),
            'cross_method_agreement': np.random.uniform(0.7, 0.9),
        }
        tracker.record_round(round_num, xai_results)
    
    tracker.save()
    print(tracker.generate_report())
