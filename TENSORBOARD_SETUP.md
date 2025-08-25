# TensorBoard Setup for EEG Meta-Learning Project

This guide explains how to set up and use TensorBoard to monitor your EEG meta-learning training progress.

## What You'll Get

With TensorBoard, you'll be able to visualize:
- **Training Loss**: Monitor how your model's loss decreases over epochs
- **Validation Loss**: Track validation performance to detect overfitting
- **Accuracy Metrics**: See classification accuracy improvements
- **Learning Curves**: Visualize the learning process across different folds and cross-validation splits

## Prerequisites

Make sure you have the required packages installed:

```bash
pip install tensorboardX tensorboard
```

## How It Works

The project has been updated to automatically log training metrics to TensorBoard. Here's what happens:

1. **Automatic Logging**: During training, metrics are automatically logged to TensorBoard
2. **Organized Logs**: Logs are organized by fold and cross-validation split
3. **Real-time Monitoring**: View training progress in real-time through the web interface

## Directory Structure

After training, your logs will be organized as follows:

```
results/
└── tensorboard_logs/
    ├── fold_0/
    │   ├── cv_0/
    │   ├── cv_1/
    │   └── ...
    ├── fold_1/
    │   ├── cv_0/
    │   ├── cv_1/
    │   └── ...
    └── ...
```

## Starting TensorBoard

### Option 1: Use the Helper Script (Recommended)

```bash
python start_tensorboard.py
```

This will start TensorBoard on port 6006 and point to the default logs directory.

### Option 2: Custom Configuration

```bash
python start_tensorboard.py --log_dir path/to/your/logs --port 6007
```

### Option 3: Direct Command

```bash
tensorboard --logdir results/tensorboard_logs --port 6006 --host 0.0.0.0
```

## Accessing TensorBoard

1. Start TensorBoard using one of the methods above
2. Open your web browser
3. Navigate to: `http://localhost:6006`
4. You'll see the TensorBoard interface with your training metrics

## What You'll See

### Scalars Tab
- **Training Loss**: How the loss decreases during training
- **Validation Loss**: Validation performance over time
- **Accuracy**: Classification accuracy improvements
- **Other Metrics**: Any other metrics logged during training

### Customization
You can:
- Compare different runs (folds/cross-validation splits)
- Smooth curves for better visualization
- Export data for further analysis
- Set custom time ranges

## Example Training Command

```bash
python train_motor_LOSO_all.py data/your_data.h5 results/ --gpu 0
```

After running this, TensorBoard will automatically collect logs in `results/tensorboard_logs/`.

## Troubleshooting

### TensorBoard Won't Start
- Make sure `tensorboard` is installed: `pip install tensorboard`
- Check if the port is already in use
- Verify the log directory exists

### No Data in TensorBoard
- Ensure you've run the training script first
- Check that the log directory path is correct
- Verify that the training script completed successfully

### Permission Issues
- Make sure you have write permissions to the output directory
- Check if the log directories can be created

## Advanced Usage

### Custom Metrics
You can add custom metrics by modifying the training script to log additional values:

```python
# Example: Log custom metric
tensorboard_logger.writer.add_scalar('custom_metric', value, epoch)
```

### Multiple Experiments
Compare different training configurations by organizing logs in separate directories:

```
results/
├── experiment_1/
│   └── tensorboard_logs/
├── experiment_2/
│   └── tensorboard_logs/
└── baseline/
    └── tensorboard_logs/
```

### Remote Access
To access TensorBoard from another machine:

```bash
tensorboard --logdir results/tensorboard_logs --port 6006 --host 0.0.0.0
```

Then access via: `http://your_machine_ip:6006`

## Tips for Best Results

1. **Start TensorBoard Early**: Start it before training begins to catch any issues
2. **Monitor Regularly**: Check TensorBoard during training to spot problems early
3. **Compare Runs**: Use TensorBoard to compare different model configurations
4. **Export Data**: Use TensorBoard's export functionality for further analysis

## Next Steps

1. Install the required packages
2. Run your training script
3. Start TensorBoard
4. Monitor your training progress in real-time!

Happy training! 🧠📊
