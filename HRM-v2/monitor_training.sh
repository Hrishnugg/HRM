#!/bin/bash
# Monitor HRM-v2 training progress

echo "=== HRM-v2 Training Monitor ==="
echo ""

while true; do
    clear
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader
    echo ""
    
    echo "=== Training Process ==="
    ps aux | grep "python train_maze" | grep -v grep | grep -v monitor | head -1
    echo ""
    
    echo "=== Recent Training Output (last 30 lines) ==="
    # Try to find the process output
    TRAIN_PID=$(ps aux | grep "python train_maze_optimized" | grep -v grep | head -1 | awk '{print $2}')
    if [ ! -z "$TRAIN_PID" ]; then
        # Check if we have access to fd
        if [ -r "/proc/$TRAIN_PID/fd/1" ]; then
            tail -30 "/proc/$TRAIN_PID/fd/1" 2>/dev/null || echo "Output not accessible"
        else
            echo "Process $TRAIN_PID running (output redirected to background)"
        fi
    else
        echo "No training process found"
    fi
    
    echo ""
    echo "Press Ctrl+C to exit monitor (training will continue)"
    echo "Refresh in 10 seconds..."
    sleep 10
done

