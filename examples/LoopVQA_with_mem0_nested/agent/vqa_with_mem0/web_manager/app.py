import streamlit as st
import os
import json
import datetime
import re
from pathlib import Path

# Configuration
PROMPTS_DIR = Path(__file__).parent / "prompts"
HISTORY_DIR = Path(__file__).parent / "history"
PASSWORD_FILE = Path(__file__).parent / "config.json"

# Ensure directories exist
PROMPTS_DIR.mkdir(exist_ok=True)
HISTORY_DIR.mkdir(exist_ok=True)

# Initialize password if not exists
if not PASSWORD_FILE.exists():
    with open(PASSWORD_FILE, "w") as f:
        json.dump({"password": "hzlhadmin123"}, f)  # Default password

def load_password():
    with open(PASSWORD_FILE, "r") as f:
        return json.load(f)["password"]

def save_password(new_password):
    with open(PASSWORD_FILE, "w") as f:
        json.dump({"password": new_password}, f)

def extract_default_prompts():
    """Extract default prompts from the prompts.py file and save them as individual txt files if they don't exist"""
    prompts_py_path = Path(__file__).parent.parent / "prompts.py"
    
    if not prompts_py_path.exists():
        st.error(f"prompts.py not found at {prompts_py_path}")
        return
    
    with open(prompts_py_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Extract all default prompt variables
    prompt_pattern = r"_DEFAULT_([A-Z_]+)_PROMPT\s*=\s*(?:\"\"\"|\')([^\"\']*(?:\"\"\"|\'))"
    matches = re.findall(prompt_pattern, content, re.DOTALL)
    
    for name, prompt_text in matches:
        # Clean up the prompt text (remove trailing quotes)
        prompt_text = prompt_text.strip()
        if prompt_text.endswith('"""'):
            prompt_text = prompt_text[:-3]
        elif prompt_text.endswith("'"):
            prompt_text = prompt_text[:-1]
        
        # Save to individual file if it doesn't exist yet
        prompt_file = PROMPTS_DIR / f"{name}_PROMPT.txt"
        if not prompt_file.exists():
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(prompt_text)

def get_prompt_files():
    """Get all prompt files in the prompts directory"""
    return sorted([f for f in PROMPTS_DIR.glob("*.txt")])

def load_prompt(file_path):
    """Load a prompt from a file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def save_prompt(file_path, content, username):
    """Save a prompt to a file and record the change in history"""
    # First, save the current version to history
    if file_path.exists():
        current_content = load_prompt(file_path)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = HISTORY_DIR / f"{file_path.stem}_{timestamp}.txt"
        
        with open(history_file, "w", encoding="utf-8") as f:
            f.write(current_content)
        
        # Save metadata about the change
        metadata_file = HISTORY_DIR / f"{file_path.stem}_{timestamp}_meta.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": timestamp,
                "user": username,
                "filename": file_path.name
            }, f)
    
    # Now save the new content
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

def get_prompt_history(prompt_name):
    """Get the history of a prompt"""
    history_files = sorted([
        f for f in HISTORY_DIR.glob(f"{prompt_name}_*.txt") 
        if not f.name.endswith("_meta.json")
    ], reverse=True)
    
    history = []
    for hf in history_files:
        meta_file = HISTORY_DIR / f"{hf.stem}_meta.json"
        if meta_file.exists():
            with open(meta_file, "r") as f:
                metadata = json.load(f)
            
            # Convert timestamp to readable format
            timestamp = metadata.get("timestamp", "")
            if timestamp:
                try:
                    dt = datetime.datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                    readable_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    readable_time = timestamp
            else:
                readable_time = "Unknown"
                
            history.append({
                "file": hf,
                "timestamp": readable_time,
                "user": metadata.get("user", "Unknown")
            })
    
    return history

def find_placeholders(text):
    """Find all placeholders in the format {placeholder}"""
    return re.findall(r'\{([^{}]+)\}', text)

# Dictionary of prompt descriptions in Chinese
PROMPT_DESCRIPTIONS = {
    "ANSWER_PROMPT": {
        "title": "回答提示词",
        "description": "这是系统提示词，用于指导AI如何回答用户的问题。它设置了AI的身份和回答风格。",
        "usage": "修改此提示词可以改变AI的整体回答风格和语气。"
    },
    "GENERAL_PROMPT": {
        "title": "通用提示词（带记忆）",
        "description": "当AI需要参考记忆和对话历史来回答问题时使用的提示词。",
        "usage": "包含{memory_context}、{conversation_history}、{datetime}和{user_instruction}占位符，请确保保留这些占位符。"
    },
    "GENERAL_PROMPT_WITHOUT_MEMORY": {
        "title": "通用提示词（无记忆）",
        "description": "当AI不需要参考记忆，只需要参考对话历史来回答问题时使用的提示词。",
        "usage": "包含{conversation_history}和{user_instruction}占位符，请确保保留这些占位符。"
    },
    "MEMORY_STORE_PROMPT": {
        "title": "记忆存储提示词",
        "description": "当用户要求AI记住某些信息时使用的提示词。",
        "usage": "包含{datetime}、{conversation_history}和{user_instruction}占位符，请确保保留这些占位符。"
    },
    "MULTIMODAL_QUERY_PROMPT": {
        "title": "多模态查询提示词",
        "description": "用于生成基于图像和文本的记忆搜索查询的提示词。",
        "usage": "包含{user_instruction}占位符，请确保保留此占位符。此提示词指导AI如何分析图像并生成搜索查询。"
    },
    "TEXT_SYSTEM_PROMPT": {
        "title": "文本系统提示词",
        "description": "纯文本交互时使用的系统提示词，设置AI的身份和回答风格。",
        "usage": "修改此提示词可以改变AI在纯文本交互中的整体回答风格和语气。"
    },
    "TEXT_GENERAL_PROMPT": {
        "title": "文本通用提示词",
        "description": "纯文本交互时使用的通用提示词框架。",
        "usage": "包含{datetime_section}、{memory_section}、{conversation_section}和{user_instruction}占位符，请确保保留这些占位符。"
    },
    "TEXT_MEMORY_CONTEXT_SECTION": {
        "title": "文本记忆上下文部分",
        "description": "在纯文本交互中插入记忆上下文的部分。",
        "usage": "包含{memory_context}占位符，请确保保留此占位符。"
    },
    "TEXT_CONVERSATION_HISTORY_SECTION": {
        "title": "文本对话历史部分",
        "description": "在纯文本交互中插入对话历史的部分。",
        "usage": "包含{conversation_history}占位符，请确保保留此占位符。"
    },
    "TEXT_MEMORY_STORE_PROMPT": {
        "title": "文本记忆存储提示词",
        "description": "纯文本交互中，当用户要求AI记住某些信息时使用的提示词。",
        "usage": "包含{conversation_history}和{user_instruction}占位符，请确保保留这些占位符。"
    }
}

# Main app
def main():
    st.set_page_config(page_title="提示词管理系统", layout="wide")
    st.title("提示词管理系统 / Prompt Management System")
    
    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "selected_prompt" not in st.session_state:
        st.session_state.selected_prompt = None
    if "first_run" not in st.session_state:
        st.session_state.first_run = True
    if "language" not in st.session_state:
        st.session_state.language = "zh"  # Default to Chinese
        
    # Extract prompts on first run
    if st.session_state.first_run:
        extract_default_prompts()
        st.session_state.first_run = False
    
    # Login form
    if not st.session_state.authenticated:
        st.subheader("登录 / Login")
        with st.form("login_form"):
            username = st.text_input("用户名 / Username")
            password = st.text_input("密码 / Password", type="password")
            submit = st.form_submit_button("登录 / Login")
            
            if submit:
                if password == load_password():
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("密码错误 / Invalid password")
    else:
        # Language selector and logout button
        col1, col2, col3 = st.columns([8, 1, 1])
        with col2:
            if st.button("中/En", help="切换语言 / Switch language"):
                st.session_state.language = "en" if st.session_state.language == "zh" else "zh"
                st.rerun()
        with col3:
            if st.button("登出 / Logout"):
                st.session_state.authenticated = False
                st.session_state.username = ""
                st.rerun()
        
        with col1:
            st.write(f"当前用户 / Logged in as: **{st.session_state.username}**")
        
        # Main interface
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("提示词文件 / Prompt Files")
            prompt_files = get_prompt_files()
            
            for pf in prompt_files:
                prompt_name = pf.stem
                display_name = PROMPT_DESCRIPTIONS.get(prompt_name, {}).get("title", prompt_name) if st.session_state.language == "zh" else prompt_name
                if st.button(display_name, key=f"btn_{prompt_name}"):
                    st.session_state.selected_prompt = pf
                    
            # Add information about how prompts work
            with st.expander("提示词工作原理 / How Prompts Work"):
                if st.session_state.language == "zh":
                    st.info("""
                    提示词存储为"prompts"目录中的文本文件。
                    应用程序启动时，会在运行时加载这些提示词。
                    
                    在此处所做的更改将立即应用于文本文件，
                    并将在应用程序下次加载提示词时使用。
                    
                    不会直接修改prompts.py文件。
                    
                    **使用说明：**
                    1. 点击左侧的提示词名称查看和编辑
                    2. 修改提示词内容（保留所有占位符，如{user_instruction}）
                    3. 点击"保存更改"按钮
                    4. 更改将在下次应用程序启动时生效
                    """)
                else:
                    st.info("""
                    Prompts are stored as text files in the 'prompts' directory. 
                    When the application starts, it loads these prompts at runtime.
                    
                    Changes made here will be applied immediately to the text files
                    and will be used the next time the application loads the prompts.
                    
                    No changes are made to the prompts.py file directly.
                    
                    **Instructions:**
                    1. Click on a prompt name on the left to view and edit
                    2. Modify the prompt content (preserve all placeholders like {user_instruction})
                    3. Click the "Save Changes" button
                    4. Changes will take effect the next time the application starts
                    """)
                    
            # Change password section
            with st.expander("修改密码 / Change Password"):
                with st.form("change_password_form"):
                    current_password = st.text_input("当前密码 / Current Password", type="password")
                    new_password = st.text_input("新密码 / New Password", type="password")
                    confirm_password = st.text_input("确认新密码 / Confirm New Password", type="password")
                    submit = st.form_submit_button("修改密码 / Change Password")
                    
                    if submit:
                        if current_password != load_password():
                            st.error("当前密码错误 / Current password is incorrect")
                        elif new_password != confirm_password:
                            st.error("新密码不匹配 / New passwords do not match")
                        else:
                            save_password(new_password)
                            st.success("密码修改成功！ / Password changed successfully!")
        
        with col2:
            if st.session_state.selected_prompt:
                prompt_path = st.session_state.selected_prompt
                prompt_name = prompt_path.stem
                
                # Get prompt description
                prompt_info = PROMPT_DESCRIPTIONS.get(prompt_name, {})
                title = prompt_info.get("title", prompt_name) if st.session_state.language == "zh" else prompt_name
                description = prompt_info.get("description", "")
                usage = prompt_info.get("usage", "")
                
                st.subheader(f"编辑 / Editing: {title}")
                
                # Show description and usage
                if st.session_state.language == "zh" and description:
                    st.markdown(f"**描述：** {description}")
                    st.markdown(f"**使用说明：** {usage}")
                
                # Load the prompt content
                prompt_content = load_prompt(prompt_path)
                
                # Find placeholders
                placeholders = find_placeholders(prompt_content)
                if placeholders:
                    if st.session_state.language == "zh":
                        st.info(f"此提示词包含以下必须保留的占位符: {', '.join(['{'+p+'}' for p in placeholders])}")
                    else:
                        st.info(f"This prompt contains the following placeholders that must be preserved: {', '.join(['{'+p+'}' for p in placeholders])}")
                
                # Edit the prompt
                new_content = st.text_area("提示词内容 / Prompt Content", prompt_content, height=400)
                
                # Check if placeholders are preserved
                new_placeholders = find_placeholders(new_content)
                missing_placeholders = set(placeholders) - set(new_placeholders)
                
                if st.button("保存更改 / Save Changes"):
                    if missing_placeholders:
                        if st.session_state.language == "zh":
                            st.error(f"以下占位符缺失: {', '.join(['{'+p+'}' for p in missing_placeholders])}")
                        else:
                            st.error(f"The following placeholders are missing: {', '.join(['{'+p+'}' for p in missing_placeholders])}")
                    else:
                        save_prompt(prompt_path, new_content, st.session_state.username)
                        if st.session_state.language == "zh":
                            st.success("提示词保存成功！")
                        else:
                            st.success("Prompt saved successfully!")
                
                # Show history
                if st.session_state.language == "zh":
                    st.subheader("版本历史")
                else:
                    st.subheader("Version History")
                    
                history = get_prompt_history(prompt_path.stem)
                
                if not history:
                    if st.session_state.language == "zh":
                        st.info("此提示词没有可用的历史记录。")
                    else:
                        st.info("No history available for this prompt.")
                else:
                    for i, entry in enumerate(history):
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            if st.session_state.language == "zh":
                                st.write(f"版本时间: {entry['timestamp']}")
                            else:
                                st.write(f"Version from {entry['timestamp']}")
                        with col2:
                            if st.session_state.language == "zh":
                                st.write(f"修改者: {entry['user']}")
                            else:
                                st.write(f"By: {entry['user']}")
                        with col3:
                            button_text = "恢复" if st.session_state.language == "zh" else "Restore"
                            if st.button(button_text, key=f"restore_{i}"):
                                historical_content = load_prompt(entry['file'])
                                save_prompt(prompt_path, historical_content, st.session_state.username)
                                if st.session_state.language == "zh":
                                    st.success("已恢复到之前的版本！")
                                else:
                                    st.success("Restored to previous version!")
                                st.rerun()

if __name__ == "__main__":
    main() 