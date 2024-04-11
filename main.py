import argparse
import json
import logging
import os
from datetime import datetime
from model_manager import ModelManager
from task_manager import TaskManager

class FlashBench:

    def __init__(self, config):
        self.init_logger()
        self.config = config
        self.task_manager = TaskManager(config)
        self.model_manager = ModelManager(config)
        
    def init_logger(self):
        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
        self.logger = logging.getLogger('FlashBench')

    def run_eval(self, task_names, output_dir):
        self.logger.info("Benchmark starting")
        os.makedirs(output_dir, exist_ok=True)
        results = self.task_manager.evaluate(self.model_manager, task_names)
        total_score = sum(r['score'] for r in results)
        avg_score = total_score / len(results)
        num_of_tasks = len(results)
        num_of_subtasks = sum(len(r['tasks']) for r in results)
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report = {
            "config": config,
            "total_score": total_score,
            "avg_score": avg_score,
            "number_of_tasks": num_of_tasks,
            "number_of_subtasks": num_of_subtasks,
            "results": [t for t in results]
        }
        with open(os.path.join(output_dir, f"result_{timestamp_str}.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        self.print_summary(results, total_score, avg_score, num_of_tasks, num_of_subtasks)
        
    def print_summary(self, results, total_score, avg_score, num_of_tasks, num_of_subtasks):
        summary = [f"{'Task (subtasks)'.ljust(40)} Score"]
        summary.append('-'*60)
        for r in results:
            num_of_subtasks = len(r['tasks'])
            summary.append(f"{(r['friendly_name'] + ' ('+str(num_of_subtasks)+')').ljust(40)} {r['score']:.4f}")
        summary.append('-'*60)
        summary.append(f"{'TOTAL SCORE:'.ljust(40)} {total_score:.4f}")
        summary.append(f"{'AVG SCORE:'.ljust(40)} {avg_score:.4f}")
        summary = '\n'.join(summary)
        self.logger.info(f"Summary report for {num_of_tasks} tasks with {num_of_subtasks} subtasks:\n{summary}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FastBench')
    parser.add_argument('-c', '--config_file', type=str, help='Config file')
    parser.add_argument('-m', '--model_path', type=str, help='Model path')
    parser.add_argument('--model_config_path', type=str, help='Json file with model config')
    parser.add_argument('-o', '--output_dir', type=str, default="result", help='Output dir for results')
    parser.add_argument('--tasks_dir', type=str, default="polish_tasks", help='Directory with task definitions')
    parser.add_argument('-t', '--tasks', type=str, help='Comma-separated list of task names. Default to all available')
    parser.add_argument('-r', '--run_name', type=str, help='Set name for this run. Default to the model path')
    args = parser.parse_args()
    
    config = {}
    if args.config_file:
        with open(args.config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
    config["model_path"] = config.get("model_path", args.model_path)
    config["model_config_path"] = config.get("model_config_path", args.model_config_path)
    config["output_dir"] = config.get("output_dir", args.output_dir)
    config["tasks_dir"] = config.get("tasks_dir", args.tasks_dir)
    config["tasks"] = config.get("tasks", args.tasks)
    config["run_name"] = config.get("run_name", args.run_name)
    if not config["run_name"]:
        config["run_name"] = config["model_path"]
    
    flashBench = FlashBench(config)
    flashBench.run_eval(config['tasks'], config['output_dir'])

