import json
import os
import re
from typing import List, Callable, Dict, Any
from pathlib import Path

from vllm import LLM, SamplingParams
from drgrpo_grader import r1_zero_reward_fn


def load_math_validation_data(file_path: str) -> List[Dict[str, Any]]:
    """
    加载 MATH 验证数据集
    
    Args:
        file_path: MATH 验证数据集的路径
        
    Returns:
        包含所有数学问题的列表
    """
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples


def extract_final_answer(answer_text: str) -> str:
    """
    从答案文本中提取最终答案（#### 后面的部分）
    
    Args:
        answer_text: 包含推理过程和最终答案的文本
        
    Returns:
        提取的最终答案
    """
    # 使用正则表达式匹配 #### 后面的内容
    match = re.search(r'####\s*([^\n]+)', answer_text)
    if match:
        return match.group(1).strip()
    else:
        # 如果没有找到 #### 标记，返回原始文本
        return answer_text.strip()


def format_r1_zero_prompt(problem: str) -> str:
    """
    使用 r1_zero 提示格式格式化数学问题
    
    Args:
        problem: 原始数学问题
        
    Returns:
        格式化后的提示
    """
    # 从文件读取提示模板
    prompt_file_path = "./cs336_alignment/prompts/r1_zero.prompt"
    
    try:
        with open(prompt_file_path, 'r', encoding='utf-8') as file:
            prompt_template = file.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"提示文件未找到: {prompt_file_path}")
    except Exception as e:
        raise Exception(f"读取提示文件时出错: {e}")
    
    # 格式化提示
    formatted_prompt = prompt_template.format(question=problem)
    return formatted_prompt


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    prompts: List[str],
    gold_answers: List[str],
    eval_sampling_params: SamplingParams,
    output_file: str
) -> Dict[str, Any]:
    """
    评估语言模型在提示列表上的表现，计算评估指标并将结果序列化到磁盘
    
    Args:
        vllm_model: 初始化的 vLLM 模型
        reward_fn: 奖励函数，接受模型输出和真实答案，返回评分字典
        prompts: 提示列表
        gold_answers: 对应的真实答案列表
        eval_sampling_params: 采样参数
        output_file: 输出文件路径
        
    Returns:
        包含评估指标的字典
    """
    print(f"Generating responses for {len(prompts)} prompts...")
    
    # 使用 vLLM 生成响应
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    
    results = []
    correct_count = 0
    format_correct_count = 0
    
    print("Evaluating responses...")
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        gold_answer = gold_answers[i]
        
        # 计算奖励/正确性 - 根据实际的 r1_zero_reward_fn 调整
        reward_result = reward_fn(generated_text, gold_answer)
        
        # 根据实际的奖励函数结构判断正确性
        is_correct = reward_result.get('reward', 0.0) == 1.0
        is_format_correct = reward_result.get('format_reward', 0.0) == 1.0
        
        if is_correct:
            correct_count += 1
        if is_format_correct:
            format_correct_count += 1
        
        # 保存结果
        result = {
            'prompt': prompt,
            'generated_text': generated_text,
            'gold_answer': gold_answer,
            'is_correct': is_correct,
            'is_format_correct': is_format_correct,
            'reward_result': reward_result,
            'example_index': i
        }
        results.append(result)
    
    # 计算评估指标
    total_examples = len(results)
    accuracy = correct_count / total_examples if total_examples > 0 else 0.0
    format_accuracy = format_correct_count / total_examples if total_examples > 0 else 0.0
    
    metrics = {
        'total_examples': total_examples,
        'correct_count': correct_count,
        'format_correct_count': format_correct_count,
        'accuracy': accuracy,
        'format_accuracy': format_accuracy,
        'model_name': 'Qwen2.5-Math-1.5B',
        'evaluation_type': 'zero_shot_math'
    }
    
    # 序列化结果到磁盘
    output_data = {
        'metrics': metrics,
        'results': results,
        'sampling_params': {
            'temperature': eval_sampling_params.temperature,
            'top_p': eval_sampling_params.top_p,
            'max_tokens': eval_sampling_params.max_tokens,
            'stop': eval_sampling_params.stop
        }
    }
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Evaluation completed!")
    print(f"Accuracy: {accuracy:.4f} ({correct_count}/{total_examples})")
    print(f"Format Accuracy: {format_accuracy:.4f} ({format_correct_count}/{total_examples})")
    print(f"Results saved to: {output_file}")
    
    return metrics


def analyze_response_format(response: str) -> Dict[str, bool]:
    """
    分析模型响应的格式是否符合要求
    
    Args:
        response: 模型生成的响应
        
    Returns:
        格式分析结果
    """
    has_think_tag = "<think>" in response and "</think>" in response
    has_answer_tag = "<answer>" in response and "</answer>" in response
    has_correct_structure = "</think> <answer>" in response
    
    return {
        'has_think_tag': has_think_tag,
        'has_answer_tag': has_answer_tag,
        'has_correct_structure': has_correct_structure,
        'is_fully_formatted': has_think_tag and has_answer_tag and has_correct_structure
    }


def main():
    """主函数：评估 Qwen 2.5 Math 1.5B 的零样本 MATH 性能"""
    
    # 配置路径
    math_validation_path = "./data/gsm8k/test.jsonl"
    output_path = "./qwen2.5_math_1.5b_zero_shot_math_results.json"
    
    # 检查文件是否存在
    if not os.path.exists(math_validation_path):
        raise FileNotFoundError(f"MATH validation file not found: {math_validation_path}")
    
    # 1. 加载 MATH 验证数据
    print("Loading MATH validation data...")
    math_examples = load_math_validation_data(math_validation_path)
    print(f"Loaded {len(math_examples)} examples")
    
    # 2. 格式化提示
    print("Formatting prompts...")
    prompts = []
    gold_answers = []
    
    for example in math_examples:
        # 使用 'question' 字段而不是 'problem'
        problem_text = example.get('question', '')
        answer_text = example.get('answer', '')
        
        # 从答案文本中提取最终答案
        final_answer = extract_final_answer(answer_text)
        
        formatted_prompt = format_r1_zero_prompt(problem_text)
        prompts.append(formatted_prompt)
        gold_answers.append(final_answer)
    
    # 打印一些示例用于调试
    print(f"\n示例提示: {prompts[0][:400]}...")
    print(f"\n示例提示: {prompts[1][:400]}...")
    print(f"\n示例提示: {prompts[2][:400]}...")
    print(f"示例答案: {gold_answers[0]}")
    print(f"示例答案: {gold_answers[1]}")
    print(f"示例答案: {gold_answers[2]}")

    
    # 3. 设置采样参数 - 确保模型能生成完整的格式
    sampling_params = SamplingParams(
        temperature=0.0,  # 对于评估，通常使用确定性生成
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],  # 确保模型在生成完答案后停止
        include_stop_str_in_output=True
    )
    
    # 4. 初始化 vLLM 模型
    print("Initializing vLLM model...")
    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B-Instruct")
    
    
    # 5. 评估模型
    metrics = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        gold_answers=gold_answers,
        eval_sampling_params=sampling_params,
        output_file=output_path
    )
    
    # 6. 打印详细结果
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Model: {metrics['model_name']}")
    print(f"Evaluation Type: {metrics['evaluation_type']}")
    print(f"Total Examples: {metrics['total_examples']}")
    print(f"Correct Answers: {metrics['correct_count']}")
    print(f"Format Correct: {metrics['format_correct_count']}")
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Format Accuracy: {metrics['format_accuracy']:.4f} ({metrics['format_accuracy']*100:.2f}%)")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()