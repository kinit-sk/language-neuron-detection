def get_task_lines(tasks): 
    basic_tasks = "" 
    temperature_tasks = "" 

    for task in tasks: 
        if not "*" in task:
            basic_tasks += f"{task}|0," 
        else:
            task = task.replace("*", "")
            temperature_tasks += f"{task}|0," 
    return f'"{basic_tasks}"', f'"{temperature_tasks}"'

def build_basic_command(cmd, model, task_block, task_dir, max_samples):
    base = (
        f"{cmd} \\\n"
        f'  "model_name={model}" \\\n'
        f"  {task_block} \\\n"
        f"  --custom-tasks {task_dir}"
    )

    if max_samples is not None:
        base += f" \\\n  --max-samples {max_samples}"

    return base
def build_temperature_command(cmd, model, task_block, task_dir, max_samples, params):
    base = (
        f"{cmd} \\\n"
        f'  "model_name={model},generation_parameters={{{params}}}" \\\n'
        f"  {task_block} \\\n"
        f"  --custom-tasks {task_dir}"
    )

    if max_samples is not None:
        base += f" \\\n  --max-samples {max_samples}"

    return base


if __name__=="__main__":
    cmd       = "lighteval accelerate"
    model     = "Qwen/Qwen2.5-0.5B-Instruct"
    task_dir  = "community_tasks/benczechmark.py"
    params    = '\\"temperature\\":0.7'
    samples   = 250
    nohup     = False

    """
    tasks marked with asterisk require temperature parameter
    
    example:
        tasks = [
            "umimeto_qa_biology_clu", <- without temperature
            "propaganda_zamereni_nli*",  <- with temperature
        ]
    """
    tasks = [
        "umimeto_qa_biology_clu",
        "propaganda_zamereni_nli",
    ]

    # build command
    basic_task, temperature_task = get_task_lines(tasks)
    basic_cmd = build_basic_command(cmd, model, basic_task, task_dir, samples)
    temperature_cmd = build_temperature_command(cmd, model, temperature_task, task_dir, samples, params)

    if nohup:
        basic_cmd = f'nohup {basic_cmd} > lighteval.log 2>&1 &'
        temperature_cmd = f'nohup {temperature_cmd} > lighteval.log 2>&1 &'

    print("\nCmd for basic tasks:")
    print("-"*50)
    print(basic_cmd)
    print("\nCmd for tasks that require temperature parameter:")
    print("-"*50)
    print(temperature_cmd)
    print("-"*50)

    


