#!/bin/bash

# TensorBoard launcher script for CPlantBox GNN training visualization
# Usage: ./launch_tensorboard.sh [port]

PORT=${1:-6006}
LOG_DIR="logs/tensorboard"

echo "🚀 Launching TensorBoard for CPlantBox GNN training visualization"
echo "📁 Log directory: $LOG_DIR"
echo "🌐 Port: $PORT"
echo ""

if [ ! -d "$LOG_DIR" ]; then
    echo "⚠️  Warning: TensorBoard log directory '$LOG_DIR' does not exist yet."
    echo "   Run a training session first to generate logs."
    echo ""
fi

echo "🔗 TensorBoard will be available at: http://localhost:$PORT"
echo "🛑 Press Ctrl+C to stop TensorBoard"
echo ""

# Launch TensorBoard
tensorboard --logdir="$LOG_DIR" --port="$PORT" --bind_all

echo ""
echo "✅ TensorBoard stopped."
