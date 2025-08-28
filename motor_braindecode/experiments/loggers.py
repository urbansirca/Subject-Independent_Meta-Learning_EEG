from abc import ABC, abstractmethod
import logging

# Get the root logger or create a properly configured one
log = logging.getLogger()  # Use root logger instead of __name__

# Alternative: Use the same logger configuration as experiment.py
log = logging.getLogger("motor_braindecode.experiments.experiment")


class Logger(ABC):
    @abstractmethod
    def log_epoch(self, epochs_df):
        raise NotImplementedError("Need to implement the log_epoch function!")


class Printer(Logger):
    """
    Prints output to the terminal using Python's logging module.
    """

    def log_epoch(self, epochs_df, fold_info=None):
        # -1 due to doing one monitor at start of training
        i_epoch = len(epochs_df) - 1
        last_row = epochs_df.iloc[-1]
        
        # Create a comprehensive single-line log with all metrics
        log_parts = []

        if fold_info:
            log_parts.append(f"LOSO FOLD: {fold_info['loso_fold']} | CV FOLD: {fold_info['cv_fold']}")

        log_parts.append(f"Epoch {i_epoch:3d}")
        
        # Add meta-learning metrics if available
        if "meta_support_loss" in last_row:
            log_parts.append(f"Meta_S: {last_row['meta_support_loss']:.4f}")
        if "meta_query_loss" in last_row:
            log_parts.append(f"Meta_Q: {last_row['meta_query_loss']:.4f}")
            
        # Add training and validation losses
        if "train_loss" in last_row:
            log_parts.append(f"Train_L: {last_row['train_loss']:.4f}")
        if "valid_loss" in last_row:
            log_parts.append(f"Val_L: {last_row['valid_loss']:.4f}")
            
        # Add accuracy metrics
        if "train_accuracy" in last_row:
            log_parts.append(f"Train_Acc: {last_row['train_accuracy']:.4f}")
        if "val_accuracy" in last_row:
            log_parts.append(f"Val_Acc: {last_row['val_accuracy']:.4f}")
            
        # Add misclassification rates if available
        if "train_misclass" in last_row:
            log_parts.append(f"Train_Mis: {last_row['train_misclass']:.4f}")
        if "valid_misclass" in last_row:
            log_parts.append(f"Val_Mis: {last_row['valid_misclass']:.4f}")
            
        # Add runtime if available
        if "runtime" in last_row:
            log_parts.append(f"Runtime: {last_row['runtime']:.2f}s")
            
        # Log everything in one line
        log.info(" | ".join(log_parts))


class TensorboardWriter(Logger):
    """
    Logs all values for tensorboard visualization using tensorboardX.
            
    Parameters
    ----------
    log_dir: string
        Directory path to log the output to
    fold_info: dict, optional
        Dictionary containing fold information like {'loso_fold': 1, 'cv_fold': 2}
    """

    def __init__(self, log_dir):
        # import inside to prevent dependency of braindecode onto tensorboardX
        from tensorboardX import SummaryWriter

        self.writer = SummaryWriter(log_dir)
        self.fold_info = {}

    def log_epoch(self, epochs_df, fold_info=None):
        # -1 due to doing one monitor at start of training
        self.fold_info = fold_info or {}
        i_epoch = len(epochs_df) - 1
        last_row = epochs_df.iloc[-1]
        
        for key, val in last_row.items():
            val = last_row[key]
            
            # Create fold-specific metric names
            if self.fold_info:
                fold_suffix = self._create_fold_suffix()
                metric_name = f"{key}/{fold_suffix}"
            else:
                metric_name = key
                
            self.writer.add_scalar(metric_name, val, i_epoch)
    
    def _create_fold_suffix(self):
        """Create a descriptive suffix for the fold information."""
        parts = []
        if 'loso_fold' in self.fold_info:
            parts.append(f"LOSO_{self.fold_info['loso_fold']}")
        if 'cv_fold' in self.fold_info:
            parts.append(f"CV_{self.fold_info['cv_fold']}")
        return "_".join(parts) if parts else "fold"
