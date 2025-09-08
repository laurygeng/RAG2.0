import os
import time
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 模拟LLM类
class MockLLM:
    def __init__(self):
        self.name = "Mock LLM"
    
    def complete(self, prompt):
        """模拟complete方法"""
        print(f"模拟LLM收到提示: {prompt}")
        time.sleep(1)  # 模拟网络延迟
        
        # 根据不同的提示返回不同的模拟响应
        if "William Shakespeare" in prompt:
            response = "William Shakespeare is widely regarded as the greatest writer in the English language and the world's pre-eminent dramatist. He was an English playwright, poet, and actor, widely regarded as the greatest writer in the English language and the world's pre-eminent dramatist."
        elif "brave knight" in prompt:
            response = "Once upon a time, there was a brave knight who lived in a castle on a hill. He protected the village from dragons and helped the poor. The villagers loved him for his courage and kindness."
        else:
            response = f"这是一个模拟响应，针对提示: {prompt[:50]}..."
        class MockResponse:
            def __init__(self, text):
                self.text = text
            
            def __str__(self):
                return self.text
        
        return MockResponse(response)   
        
    def stream_complete(self, prompt):
        """模拟stream_complete方法"""
        print(f"模拟LLM流式响应提示: {prompt}")
        
        response = "Once upon a time, there was a brave knight who lived in a castle on a hill. He protected the village from dragons and helped the poor. The villagers loved him for his courage and kindness."
        
        # 模拟流式响应，逐词返回
        words = response.split()
        for i, word in enumerate(words):
            time.sleep(0.1)  # 模拟流式延迟
            yield MockChunk(word + " ")
    
    def chat(self, messages):
        """模拟chat方法"""
        print(f"模拟LLM收到聊天消息: {len(messages)}条消息")
        time.sleep(1)  # 模拟网络延迟
        
        # 获取最后一条用户消息
        last_message = messages[-1] if messages else None
        if last_message and hasattr(last_message, 'content'):
            user_content = last_message.content
        else:
            user_content = "Hello"
        
        # 根据消息内容生成响应
        if "capital of france" in user_content.lower():
            response_text = "The capital of France is Paris."
        elif "capital of japan" in user_content.lower():
            response_text = "The capital of Japan is Tokyo."
        elif "france" in user_content.lower():
            response_text = "The capital of France is Paris."
        elif "japan" in user_content.lower():
            response_text = "The capital of Japan is Tokyo."
        else:
            response_text = f"This is a mock response to: {user_content[:50]}..."
        
        # 返回一个类似OpenAI响应的对象
        class MockChatResponse:
            def __init__(self, text):
                self.text = text
            
            def __str__(self):
                return self.text
        
        return MockChatResponse(response_text)

class MockChunk:
    def __init__(self, delta):
        self.delta = delta

def main():
    print("=== 使用模拟LLM测试 ===\n")
    
    try:
        llm = MockLLM()
        print("--- 1. 文本完成 (Completion) ---")
        print("发送请求: \"William Shakespeare is \"")
        response = llm.complete("William Shakespeare is ")
        print(f"LLM响应:\n{response.text}\n")
        print("--- 2. 流式文本完成 (Streaming Completion) ---")
        print("发送请求: \"Tell me a short story about a brave knight.\"")
        stream_response = llm.stream_complete("Tell me a short story about a brave knight.")
        print("LLM响应 (流式):")
        full_text = ""
        for chunk in stream_response:
            print(chunk.delta, end="", flush=True) 
            full_text += chunk.delta
        print("\n")
        print("--- 3. 聊天 (Chat) ---")
        print("开始一个多轮对话...")
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            ChatMessage(role=MessageRole.USER, content="What is the capital of France?"),
        ]
        chat_response = llm.chat(messages)
        print(f"助手响应: {chat_response.text}")
        messages.append(chat_response)
        messages.append(ChatMessage(role=MessageRole.USER, content="And the capital of Japan?"))
        chat_response_2 = llm.chat(messages)
        print(f"助手第二次响应: {chat_response_2.text}")
        print("\n")
        print("--- 4. 多模态（Multi-modal） ---")
        print("这是一个概念示例，展示如何结合文本和图像输入。")
        print("需要一个支持多模态的模型 (如 gpt-4-vision-preview)。")
        print("示例输入: [图片 of a puppy] 和文本 'What kind of animal is this?'")
        print("\n--- 5. 工具调用 (Tool Calling) ---")
        print("这是一个概念示例，展示LLM如何自动调用工具。")
        print("需要预先定义好工具，并提供给LLM。")
        print("示例: LLM可以自动调用一个天气查询工具来回答 'What's the weather like in Boston?'")   
        print("测试1: 基本complete调用")
        print("正在向模拟LLM发送请求...")
        response = llm.complete("William Shakespeare is ")
        print("模拟LLM响应:")
        print(response)
        print("\n" + "="*50)
        
        print("测试2: 流式响应")
        print("正在向模拟LLM发送流式请求...")
        print("流式响应:")
        stream_response = llm.stream_complete("Tell me a short story about a brave knight.")
        for chunk in stream_response:
            print(chunk.delta, end="", flush=True)
        print("\n")
        print("="*50)
        
    except Exception as e:
        print(f"调用模拟LLM时发生错误: {e}")

if __name__ == "__main__":
    main()
