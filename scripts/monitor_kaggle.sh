#!/bin/bash
# WakeWord Kaggle Training Monitor
# Monitors notebook status and downloads results when complete

export KAGGLE_API_TOKEN=KGAT_9df69eaa447345a6958db41a81af50da
KERNEL="hammadmunir959/wakeword-keyword-spotting-training"
OUTPUT_DIR="models/kaggle_output"

echo "========================================"
echo "  WakeWord Kaggle Training Monitor"
echo "========================================"
echo "Kernel: $KERNEL"
echo "Started: $(date)"
echo ""

check_count=0
while true; do
    check_count=$((check_count + 1))
    status=$(kaggle kernels status $KERNEL 2>&1)
    
    # Extract status
    if echo "$status" | grep -q "RUNNING"; then
        current_status="RUNNING"
        symbol="⏳"
    elif echo "$status" | grep -q "COMPLETE"; then
        current_status="COMPLETE"
        symbol="✅"
    elif echo "$status" | grep -q "ERROR"; then
        current_status="ERROR"
        symbol="❌"
    elif echo "$status" | grep -q "QUEUED"; then
        current_status="QUEUED"
        symbol="⏸️"
    else
        current_status="UNKNOWN"
        symbol="❓"
    fi
    
    # Print status
    printf "\r[Check #%03d] %s Status: %-10s | Time: %s" "$check_count" "$symbol" "$current_status" "$(date +%H:%M:%S)"
    
    # Handle completion
    if [ "$current_status" == "COMPLETE" ]; then
        echo ""
        echo ""
        echo "========================================"
        echo "  Training Complete!"
        echo "========================================"
        echo ""
        echo "Downloading output files..."
        mkdir -p $OUTPUT_DIR
        kaggle kernels output $KERNEL -p $OUTPUT_DIR
        
        echo ""
        echo "Downloaded files:"
        ls -la $OUTPUT_DIR/
        
        echo ""
        echo "Training finished at: $(date)"
        break
    fi
    
    # Handle error
    if [ "$current_status" == "ERROR" ]; then
        echo ""
        echo ""
        echo "========================================"
        echo "  Training Failed!"
        echo "========================================"
        echo "Check notebook for errors: https://www.kaggle.com/code/$KERNEL"
        break
    fi
    
    # Wait before next check
    sleep 30
done
