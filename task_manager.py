import json
import os
import re
import logging

class TaskManager:

    def __init__(self, config):
        self.logger = logging.getLogger('TaskManager')
        self.load_tasks(config)

    def get_tasks(self, names=None):
        if names is not None:
            names = set(names.split(','))
            return [t for t in self.tasks if t['name'] in names]
        else:
            return self.tasks
        
    def load_tasks(self, config):
        self.tasks = []
        for filename in os.listdir(config["tasks_dir"]):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(config["tasks_dir"], filename), 'r') as f:
                        self.tasks.append(json.load(f))
                except Exception as err:
                    self.logger.warning(f"Unable to load {filename}. Error: {err}")

    def evaluate(self, model_manager, task_names=None):
        results = []
        for task in self.get_tasks(task_names):
            self.logger.info(f"Starting task {task['name']}")
            subtask_results = []
            for subtask in task['tasks']:
                assert 'input' in subtask
                assert 'target' in subtask
                response = model_manager.generate(subtask['input'])
                hits, misses = self.process_response(response, subtask)
                subtask_results.append({
                    'id': subtask['id'],
                    'score': hits / (hits + misses),
                    'hits': hits,
                    'misses': misses,
                    'response': response
                })
            score = sum(s['score'] for s in subtask_results) / len(subtask_results)
            results.append({
                'name': task['name'],
                'friendly_name': task['friendly_name'],
                'score': score,
                'tasks': subtask_results
            })
            self.logger.info(f"Task {task['name']} finished with score {score:.4f}")
        return results
        
    def process_response(self, response, task):
        hits = 0
        misses = 0
        output = response['model_response']
        target = task['target']
        search_type = target['type'] if 'type' in target else ''
        if 'negative' in target:
            if self.find_value_in_response(search_type, target["negative"], output):
                return 0, 1
        elif 'negatives' in target:
            for negative in target["negatives"]:
                if self.find_value_in_response(search_type, negative, output):
                    return 0, 1
                    
        if 'value' in target:
            if self.find_value_in_response(search_type, target["value"], output):
                hits += 1
            else:
                misses += 1
        if 'values' in target:
            for value in target["values"]:
                if self.find_value_in_response(search_type, value, output):
                    if target.get('values_logical_operator', 'and') == 'or':
                        return 1, 0
                    else:
                        hits += 1
                else:
                    if target.get('values_logical_operator', 'and') == 'or':
                        misses = 1
                    else:
                        misses += 1
        return hits, misses
    
    def find_value_in_response(self, search_type, value, output):
        if search_type == 'contains':
            return value in output.lower()
        elif search_type == 'contains_word':
            return re.search(rf'\b{re.escape(value)}\b', output, re.IGNORECASE)
        elif search_type == 'regex':
            return re.search(rf'{value}', output, re.IGNORECASE)
        elif search_type == 'exact_match':
            return value == output
        elif search_type == 'json_contains':
            try:
                start_index = output.find('{')
                end_index = output.rfind('}') + 1
                output = json.loads(output[start_index:end_index])
                for k,v in value.items():
                    if k not in output:
                        return False
                    if not self.find_value_in_response(v['type'], v['value'], output[k]):
                        return False
                return True
            except Exception as err:
                self.logger.warning(f"Unable to parse json:\n{output}\n\nError: {err}")
                return False
        elif search_type == 'python_code':
            if 'import ' in output:
                self.logger.warning(f"Forbidden operation detected in python code:\n{output}")
                return False
            try:
                code = self.extract_python_code(output)
                if code is None:
                    return False
                
                code = code.strip() + f"\n\nresult = {value['call']}"
                local_scope = {}
                exec(code, {}, local_scope)
                
                code_exec_result = local_scope['result']
                return self.find_value_in_response(value['result']['type'], value['result']['value'], code_exec_result)
            except Exception as err:
                self.logger.warning(f"Unable to execute python code:\n{output}\n\nError: {err}")
                return False
        
        raise Exception(f"Unsupported type: {search_type}")
        
    def extract_python_code(self, text):
        lines = text.split('\n')
        start_index = -1
        end_index = -1
        for i, line in enumerate(lines):
            if "def " in line:
                start_index = i
            elif "return" in line and start_index != -1:
                end_index = i
                break
        if start_index != -1 and end_index != -1:
            return '\n'.join(lines[start_index:end_index+1])
        else:
            return None
        
