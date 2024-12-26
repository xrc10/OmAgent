# 图片互动RPG游戏示例

这个示例展示了如何使用框架创建一个基于图片输入和对话的RPG游戏。游戏会根据用户上传的图片和对话内容动态生成故事情节和目标。

## 概述

这个示例实现了一个交互式RPG游戏工作流，包含以下主要组件：

1. **图片输入处理**
   - RPGImageInput: 处理用户上传的图片
   - 作为故事情节的起点和灵感来源

2. **故事生成器**
   - StoryGenerator: 基于图片内容生成故事背景和目标
   - 可以随机生成不同类型的故事（如间谍、外星人、鬼怪等）

3. **交互式对话循环**
   - RPGDialogue: 处理用户的对话输入并推进故事情节
   - StoryProgress: 评估故事进展并决定是否结束
   - 使用DoWhileTask在5个回合内完成故事

4. **故事结局生成**
   - StoryEnding: 根据之前的互动生成合适的结局

## 工作流程

```
开始 -> 图片输入 -> 故事生成 -> 对话循环(对话+进度评估) -> 故事结局 -> 结束
```

## 前提条件

- Python 3.10+
- 所需包已安装（见requirements.txt）
- OpenAI API访问权限
- Redis服务器（本地或远程）
- Conductor服务器（本地或远程）

## 运行示例

1. 运行RPG游戏工作流：

   终端/CLI使用：
   ```bash
   python run_cli.py
   ```

   应用/GUI使用：
   ```bash
   python run_app.py
   ```

## 故事类型

游戏可以随机生成多种类型的故事，包括但不限于：
- 间谍任务
- 外星人探索
- 灵异事件
- 魔法冒险
- 侦探推理

每个故事都会在5个回合内完成，并确保有完整的开始、发展和结局。 