"""
简易 MCP Server — 提供几个实用工具

工具列表:
    - get_current_time : 获取当前日期和时间
    - calculate        : 简单四则运算

启动方式:
    python tools_server.py
    （通过 stdio 与 MCP 客户端通信，不直接交互）
"""

import asyncio
from datetime import datetime
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

server = Server("demo-tools")


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_current_time",
            description="获取当前的日期和时间。当用户询问'现在几点'、'今天是什么日期'、'当前时间'等问题时使用此工具。",
            inputSchema={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "时区，例如 'Asia/Shanghai'、'America/New_York'，默认为北京时间",
                    }
                },
            },
        ),
        Tool(
            name="calculate",
            description="执行简单的数学计算。支持加减乘除、幂运算等。当用户需要进行数学计算时使用此工具。",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如 '2+3*4'、'sqrt(16)'、'2**10'",
                    }
                },
                "required": ["expression"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "get_current_time":
        tz = arguments.get("timezone", "Asia/Shanghai")
        now = datetime.now()
        result = f"当前日期和时间：{now.strftime('%Y年%m月%d日 %H:%M:%S')}（时区配置: {tz}）"
        return [TextContent(type="text", text=result)]

    elif name == "calculate":
        expression = arguments.get("expression", "")
        if not expression:
            return [TextContent(type="text", text="错误：请提供数学表达式。")]

        # 安全计算：仅允许数字、运算符、括号、空格、sqrt 等
        import re
        if not re.match(r'^[\d\s+\-*/().%^**sqrtabs]+$', expression):
            return [TextContent(type="text", text=f"错误：表达式包含不允许的字符。仅支持数字和基本运算符。")]

        try:
            # 替换数学函数为 Python 等效
            safe_expr = expression.replace("^", "**").replace("sqrt", "math.sqrt").replace("abs", "abs")
            import math
            result = eval(safe_expr, {"__builtins__": {}}, {"math": math})
            return [TextContent(type="text", text=f"计算结果：{expression} = {result}")]
        except Exception as e:
            return [TextContent(type="text", text=f"计算出错：{e}")]

    else:
        return [TextContent(type="text", text=f"未知工具: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
