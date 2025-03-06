# Constants for memory decision
THRESHOLD = 0.40

# Import prompts from text files
def load_prompt(prompt_name):
    """Load a prompt from a text file in the web_manager/prompts directory"""
    from pathlib import Path
    import os
    
    # Get the directory of this file
    current_dir = Path(__file__).parent
    prompt_file = current_dir / "web_manager" / "prompts" / f"{prompt_name}.txt"
    
    # If the prompt file exists, load it
    if prompt_file.exists():
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    
    # Otherwise, return a default prompt
    return ""

# Define prompt names as constants but load content dynamically
ANSWER_PROMPT = load_prompt("ANSWER_PROMPT")
GENERAL_PROMPT = load_prompt("GENERAL_PROMPT")
GENERAL_PROMPT_WITHOUT_MEMORY = load_prompt("GENERAL_PROMPT_WITHOUT_MEMORY")
MEMORY_STORE_PROMPT = load_prompt("MEMORY_STORE_PROMPT")
MULTIMODAL_QUERY_PROMPT = load_prompt("MULTIMODAL_QUERY_PROMPT")
TEXT_SYSTEM_PROMPT = load_prompt("TEXT_SYSTEM_PROMPT")
TEXT_GENERAL_PROMPT = load_prompt("TEXT_GENERAL_PROMPT")
TEXT_MEMORY_CONTEXT_SECTION = load_prompt("TEXT_MEMORY_CONTEXT_SECTION")
TEXT_CONVERSATION_HISTORY_SECTION = load_prompt("TEXT_CONVERSATION_HISTORY_SECTION")
TEXT_MEMORY_STORE_PROMPT = load_prompt("TEXT_MEMORY_STORE_PROMPT")

# Default prompt values if files don't exist
# These will be used as fallbacks if the text files aren't found

_DEFAULT_ANSWER_PROMPT = """你是小欧，一个由 Om AI 创建的 AI 助手，专门用于回答与图像相关的问题。请始终基于可用信息提供有帮助、准确和简洁的回答。"""

_DEFAULT_GENERAL_PROMPT = """
请回答问题，并参考提供的相关记忆和对话历史。重要指引：

1. 保持回答简洁，最多50个汉字
2. 如果问题涉及过去的事件，且没有找到相关记忆，请回答"抱歉，我没有找到相关的记录"
3. 始终使用中文回答

相关记忆：
{memory_context}

对话历史：
{conversation_history}

当前时间：{datetime}

问题：{user_instruction}"""

_DEFAULT_GENERAL_PROMPT_WITHOUT_MEMORY = """
请回答问题，并参考提供的对话历史。始终使用中文回答。

对话历史：
{conversation_history}

问题：{user_instruction}"""

_DEFAULT_MEMORY_STORE_PROMPT = """请根据图片内容创建一条简短的记忆记录。要求：

1. 结合图片，用20字以内简洁描述需要记忆的内容
2. 如果存在相对时间，例如"昨天"，请参考当前时间：{datetime}

对话历史：
{conversation_history}

记忆请求：{user_instruction}

请按以下格式回复：
[复述记忆内容]
好的，我已经记住了。"""

_DEFAULT_MULTIMODAL_QUERY_PROMPT = """Given the user's question and the image, first briefly describe the key details of the image, then generate a clear and specific query to search in memory. If the user's question is in Chinese, respond in Chinese.

Examples:
User question: Did I eat this before?
IMAGE: A round pizza with cheese and pepperoni toppings on a wooden serving board.
SEARCH_QUERY: previous instances of eating pepperoni pizza

User question: 这个我以前吃过吗？
IMAGE: 一个10寸的芝士披萨，表面铺满了融化的马苏里拉奶酪。
SEARCH_QUERY: 之前吃芝士披萨的记录

User question: 我什么时候买的这个？
IMAGE: 一个棕色的中号皮包，有金色的金属扣件和长肩带。
SEARCH_QUERY: 购买棕色中号皮包的时间记录

Format your response with:
IMAGE: <brief description of the key details in the image>
SEARCH_QUERY: <your specific search query>

User question: {user_instruction}"""

_DEFAULT_TEXT_SYSTEM_PROMPT = """你是小欧，一个由 Om AI 创建的 AI 助手。请始终基于可用信息提供有帮助、准确和简洁的回答。"""

_DEFAULT_TEXT_GENERAL_PROMPT = """
请回答以下问题。重要指引：

1. 始终使用中文回答

{datetime_section}{memory_section}{conversation_section}

问题：{user_instruction}"""

_DEFAULT_TEXT_MEMORY_CONTEXT_SECTION = """
2. 保持回答简洁：
   - 回答最多50个汉字

3. 关于记忆：
   - 只使用与当前问题直接相关的信息
   - 对于关于过去事件/购买的问题（例如"我什么时候买过这个？"）：
     - 如果没有找到相关记忆，回答"抱歉，我没有找到相关的记录"

相关记忆：
{memory_context}
"""

_DEFAULT_TEXT_CONVERSATION_HISTORY_SECTION = """
对话历史：
{conversation_history}
"""

_DEFAULT_TEXT_MEMORY_STORE_PROMPT = """
这是一个记忆存储请求。请按以下方式回应：

1. 保持回答简洁：
   - 首先用"好的，记住了"或"明白了"等简短话语确认
   - 然后复述需要记忆的信息
   
2. 始终使用中文回答

对话历史：
{conversation_history}

要存储的内容：{user_instruction}"""

# Use default values if the loaded prompts are empty
if not ANSWER_PROMPT:
    ANSWER_PROMPT = _DEFAULT_ANSWER_PROMPT
if not GENERAL_PROMPT:
    GENERAL_PROMPT = _DEFAULT_GENERAL_PROMPT
if not GENERAL_PROMPT_WITHOUT_MEMORY:
    GENERAL_PROMPT_WITHOUT_MEMORY = _DEFAULT_GENERAL_PROMPT_WITHOUT_MEMORY
if not MEMORY_STORE_PROMPT:
    MEMORY_STORE_PROMPT = _DEFAULT_MEMORY_STORE_PROMPT
if not MULTIMODAL_QUERY_PROMPT:
    MULTIMODAL_QUERY_PROMPT = _DEFAULT_MULTIMODAL_QUERY_PROMPT
if not TEXT_SYSTEM_PROMPT:
    TEXT_SYSTEM_PROMPT = _DEFAULT_TEXT_SYSTEM_PROMPT
if not TEXT_GENERAL_PROMPT:
    TEXT_GENERAL_PROMPT = _DEFAULT_TEXT_GENERAL_PROMPT
if not TEXT_MEMORY_CONTEXT_SECTION:
    TEXT_MEMORY_CONTEXT_SECTION = _DEFAULT_TEXT_MEMORY_CONTEXT_SECTION
if not TEXT_CONVERSATION_HISTORY_SECTION:
    TEXT_CONVERSATION_HISTORY_SECTION = _DEFAULT_TEXT_CONVERSATION_HISTORY_SECTION
if not TEXT_MEMORY_STORE_PROMPT:
    TEXT_MEMORY_STORE_PROMPT = _DEFAULT_TEXT_MEMORY_STORE_PROMPT