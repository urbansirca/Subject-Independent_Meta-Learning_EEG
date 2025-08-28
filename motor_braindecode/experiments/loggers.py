from abc import ABC, abstractmethod
import logging

log = logging.getLogger(__name__)


class Logger(ABC):
    @abstractmethod
    def log_epoch(self, epochs_df):
        raise NotImplementedError("Need to implement the log_epoch function!")


class Printer(Logger):
    """
    Prints output to the terminal using Python's logging module.
    """

    def log_epoch(self, epochs_df):
        # -1 due to doing one monitor at start of training
        i_epoch = len(epochs_df) - 1
        last_row = epochs_df.iloc[-1]
        
        # Create a comprehensive single-line log with all metrics
        log_parts = [f"Epoch {i_epoch:3d}"]
        
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
    Logs all values for tensorboard visualiuzation using tensorboardX.
            
    Parameters
    ----------
    log_dir: string
        Directory path to log the output to
    """

    def __init__(self, log_dir):
        # import inside to prevent dependency of braindecode onto tensorboardX
        from tensorboardX import SummaryWriter

        self.writer = SummaryWriter(log_dir)

    def log_epoch(self, epochs_df):
        # -1 due to doing one monitor at start of training
        i_epoch = len(epochs_df) - 1
        last_row = epochs_df.iloc[-1]
        for key, val in last_row.items():
            val = last_row[key]
            self.writer.add_scalar(key, val, i_epoch)
