#!/bin/bash
"""Quick system baseline for AGI Research Lab. Captures CPU, memory, and disk info."""

echo "=== AGI Research Lab - System Baseline ==="
echo "Date: $(date)"
echo ""

echo "=== CPU ==="
if command -v lscpu &> /dev/null; then
    lscpu | grep -E "^(Model name|CPU\(s\)|Thread|Core|Socket|CPU MHz)"
else
    cat /proc/cpuinfo | grep "model name" | head -1
    echo "CPUs: $(nproc)"
fi
echo ""

echo "=== Memory ==="
if command -v free &> /dev/null; then
    free -h
else
    cat /proc/meminfo | head -5
fi
echo ""

echo "=== Disk ==="
df -h / 2>/dev/null || df -h .
echo ""

echo "=== GPU ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
else
    echo "No NVIDIA GPU detected"
fi
echo ""

echo "=== Python ==="
python3 --version 2>/dev/null || echo "Python not found"
echo ""

echo "=== Network ==="
ip addr show 2>/dev/null | grep "inet " | head -5 || ifconfig | grep "inet " | head -5
echo ""

echo "=== AGI Lab Directory ==="
du -sh /home/himanshu/Desktop/AGI_RESEARCH_LAB/ 2>/dev/null || echo "Dir not accessible"
