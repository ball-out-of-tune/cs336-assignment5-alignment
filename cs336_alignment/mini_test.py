"""
测试 Qwen2.5-Math-1.5B-Instruct 模型的格式输出
用于诊断为什么模型不生成 <think></think> <answer></answer> 格式
"""

from vllm import LLM, SamplingParams


def test_qwen_formats():
    """测试不同的 prompt 格式"""
    
    print("初始化模型...")
    llm = LLM(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code=True
    )
    print("模型加载完成!\n")
    
    # 定义多个测试用例
    tests = []
    
    # 测试 1: 直接要求格式
    tests.append({
        "name": "Test 1 - Direct Format Request",
        "prompt": "What is 2+2? Answer in this format: <think>reasoning</think> <answer>result</answer>",
        "description": "直接要求使用特定格式"
    })
    
    # 测试 2: 使用课程提供的 prompt 格式
    tests.append({
        "name": "Test 2 - CS336 r1_zero Format",
        "prompt": """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: What is 2+2?
Assistant:""",
        "description": "CS336 课程的 r1_zero.prompt 格式"
    })
    
    # 测试 3: 使用 Qwen ChatML 格式
    tests.append({
        "name": "Test 3 - Qwen ChatML Format",
        "prompt": """<|im_start|>system
You are a helpful math assistant.<|im_end|>
<|im_start|>user
What is 2+2? Please format your response as: <think>your reasoning</think> <answer>final answer</answer><|im_end|>
<|im_start|>assistant
""",
        "description": "使用 Qwen 的 ChatML 格式"
    })
    
    # 测试 4: 使用 Few-shot 示例
    tests.append({
        "name": "Test 4 - Few-shot Example",
        "prompt": """A conversation between User and Assistant. The Assistant always formats responses as: <think> reasoning </think> <answer> result </answer>

Example:
User: What is 1+1?
Assistant: <think>
I need to add 1 and 1.
1 + 1 = 2
</think> <answer>
2
</answer>

Now solve this:
User: What is 2+2?
Assistant:""",
        "description": "提供一个完整的格式示例"
    })
    
    # 测试 5: 直接开始标签
    tests.append({
        "name": "Test 5 - Start with Tag",
        "prompt": """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: What is 2+2?
Assistant: <think>""",
        "description": "在 prompt 末尾直接开始 <think> 标签"
    })
    
    # 测试 6: GSM8K 真实问题
    tests.append({
        "name": "Test 6 - Real GSM8K Problem",
        "prompt": """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Assistant:""",
        "description": "GSM8K 数据集的真实问题"
    })
    
    # 运行所有测试
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        stop=["</answer>", "<|im_end|>"],
        include_stop_str_in_output=True
    )
    
    results = []
    
    for i, test in enumerate(tests, 1):
        print(f"\n{'='*80}")
        print(f"{test['name']}")
        print(f"描述: {test['description']}")
        print(f"{'='*80}")
        print(f"\nPrompt (前300字符):")
        print(test['prompt'][:300])
        if len(test['prompt']) > 300:
            print("...(truncated)")
        
        print(f"\n生成中...")
        output = llm.generate([test['prompt']], sampling_params)
        response = output[0].outputs[0].text
        
        print(f"\n模型输出:")
        print(response)
        
        # 检查格式
        has_think = "<think>" in response and "</think>" in response
        has_answer = "<answer>" in response and "</answer>" in response
        has_correct_structure = "</think> <answer>" in response
        
        print(f"\n格式检查:")
        print(f"  ✓ 包含 <think></think>: {has_think}")
        print(f"  ✓ 包含 <answer></answer>: {has_answer}")
        print(f"  ✓ 正确结构 (</think> <answer>): {has_correct_structure}")
        
        format_ok = has_think and has_answer and has_correct_structure
        print(f"\n  总体格式正确: {'✓ YES' if format_ok else '✗ NO'}")
        
        results.append({
            "test_name": test['name'],
            "format_ok": format_ok,
            "has_think": has_think,
            "has_answer": has_answer,
            "response_length": len(response)
        })
        
        print('='*80)
    
    # 总结
    print(f"\n\n{'='*80}")
    print("测试总结")
    print('='*80)
    
    for result in results:
        status = "✓" if result['format_ok'] else "✗"
        print(f"{status} {result['test_name']}: "
              f"格式{'正确' if result['format_ok'] else '不正确'} "
              f"(长度: {result['response_length']} 字符)")
    
    passed = sum(1 for r in results if r['format_ok'])
    total = len(results)
    print(f"\n通过率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    # 给出建议
    print(f"\n{'='*80}")
    print("建议:")
    print('='*80)
    
    if passed == 0:
        print("❌ 所有测试都失败了！")
        print("   可能的原因:")
        print("   1. Qwen2.5-Math-1.5B-Instruct 模型没有被训练使用这种格式")
        print("   2. 需要使用不同的 prompt 策略")
        print("   3. 可能需要 fine-tune 模型来支持这种格式")
        print("\n   建议:")
        print("   - 检查模型文档，看是否有推荐的输出格式")
        print("   - 考虑使用后处理来添加格式标签")
        print("   - 或者修改 reward_fn 来接受模型的原生输出格式")
    elif passed < total:
        print(f"⚠️  部分测试通过 ({passed}/{total})")
        print("   检查哪些格式有效，并在实际评估中使用那些格式")
        best_test = max(results, key=lambda x: x['format_ok'])
        print(f"\n   最佳格式: {best_test['test_name']}")
    else:
        print("✓ 所有测试都通过了！")
        print("  模型可以正确生成所需格式")


if __name__ == "__main__":
    test_qwen_formats()