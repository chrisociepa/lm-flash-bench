# LM Flash Bench

Welcome to LM Flash Bench, a streamlined framework designed for rapid and customizable benchmarking of fine-tuned Large Language Models (LLMs), offering quick performance insights with minimal setup.

## Basic Usage

To evaluate a model in the Hugging Face format on Polish tasks, you can use the following command (assuming you are using a CUDA-compatible GPU):

```bash
python main.py \
    -m "speakleash/Bielik-7B-Instruct-v0.1" \
    -o "../result" \
    -r "Bielik-7B-Instruct-v0.1" \
    --tasks_dir="./polish_tasks"
```
