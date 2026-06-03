import asyncio      #异步框架
import base64       #base64编码框架
import os
import sys
import json
from dataclasses import dataclass, field
from datetime import datetime
from openai import OpenAI

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent     #引入mcp的工具类型和文本内容类型


#日志函数
def _log(msg: str) -> None:
    """写 stderr，不影响 stdio 协议的 stdout 通信"""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[mcp-image-recognizer {ts}] {msg}", file=sys.stderr, flush=True)


# ============================================================
# 配置层 —— 存放 API Key、模型类型等，与业务逻辑解耦
# ============================================================
@dataclass
class ImageRecognitionConfig:
    """图像识别模型配置。可通过环境变量覆盖，也可在代码中直接赋值。"""
    api_key: str = field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", ""))    #apikey，设置在环境变量
    model: str = "qwen3.7-plus" #模型名称
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1" #模型路径，从阿里云那边复制
    max_tokens: int = 2048  # 结构描述需要更多 token
    temperature: float = 0.1


# ============================================================
# 服务层 —— 负责调用外部多模态模型，与 MCP 协议无关
# ============================================================
class ImageRecognitionService:
    """封装对多模态模型的调用，只暴露一个 recognize 方法。"""

    def __init__(self, config: ImageRecognitionConfig) -> None:
        self._config = config
        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    _DEFAULT_PROMPT = (
        "请详细描述这张神经网络/算法架构图的结构信息，目的是让一个不会看图的人能根据你的描述用 SVG 精确复现这张图。"
        "按以下顺序逐一描述：\n"
        "1. **画布与方向**：图是横向（左→右）还是纵向（上→下）？大致有几个主列/主行？\n"
        "2. **模块清单**：从输入到输出，逐一列出每个矩形/圆形/柱状模块——它的文字标签是什么、大致长宽比（扁的/方的/竖长的）、"
        "填充色或边框色、在整体中的位置（第几列第几个）。\n"
        "3. **连接与箭头**：模块之间有哪些箭头连接？方向是→、←、↔、⇢（残差/跳跃）？单箭头还是双箭头？"
        "箭头是否有文字标注（如维度、运算名）？\n"
        "4. **堆叠与并列**：哪些模块是上下堆叠的，哪些是左右并列的？用“A 在 B 上方”“C 与 D 水平对齐”这类表述。\n"
        "5. **括号与分组**：图中是否有大括号/虚线框把若干模块圈成一个大的 layer/block/stage？颜色和标签是什么？\n"
        "6. **图例与标注**：角落是否有图例、标题、尺寸标注（如 H×W×C）、公式？\n"
        "7. **特殊符号**：是否有 ⊕ 加法、⊗ 乘法、⊗ 拼接/concat、σ softmax 等运算符号？标在哪些位置？\n"
        "请用绘图者能直接照着画的语言描述，越具体越好，不要遗漏任何模块或箭头。"
    )

    def recognize(self, image_path: str, prompt: str | None = None) -> str:
        """读取本地图片，base64 编码后发送给多模态模型，返回文字描述。"""
        if prompt is None:
            prompt = self._DEFAULT_PROMPT
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片不存在: {image_path}")

        # 推断 MIME 类型
        ext = os.path.splitext(image_path)[1].lower()
        mime_map = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
        }

        # 如果图片扩展名不在映射中，默认使用 PNG
        mime_type = mime_map.get(ext, "image/png")

        with open(image_path, "rb") as f:
            image_bytes = f.read()
            data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode()}"

        _log(f"调用模型 {self._config.model}，图片大小: {len(image_bytes)} bytes")

        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }],
            extra_body={"enable_thinking": True},
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
        )
        return response.choices[0].message.content


# ============================================================
# MCP 协议层 —— 对外暴露 stdio 接口供 Claude Code 调用
# ============================================================
#实例化
config = ImageRecognitionConfig()
service = ImageRecognitionService(config)
app = Server("image-recognizer")


@app.list_tools()
async def list_tools() -> list[Tool]:
    tools = [
        Tool(
            name="recognize_image",
            description=(
                "识别图片内容，返回面向SVG绘图的结构化描述。将图片发送给外部多模态模型（如 Qwen）进行识别，"
                "默认返回聚焦于模块形状、空间布局、箭头连接、颜色标注的细粒度描述，"
                "便于 DeepSeek 等纯文本模型根据描述精确复现图表（如神经网络架构图）。"
                "也可通过 prompt 参数覆盖为通用图片描述。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "图片文件的绝对路径",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "对图片的提问或识别指令。默认为面向SVG绘图的结构化描述指令，如需通用描述可自行覆盖",
                    },
                },
                "required": ["image_path"],
            },
        )
    ]
    _log(f"Client 请求工具列表 → 返回 {len(tools)} 个工具")
    return tools


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    _log(f"收到调用: name={name}, args={json.dumps(arguments, ensure_ascii=False)}")

    if name == "recognize_image":
        image_path = arguments.get("image_path", "")
        prompt = arguments.get("prompt", "") or None  # 空字符串视为未传，走默认 prompt

        try:
            result = service.recognize(image_path, prompt)
            preview = result[:100] + "..." if len(result) > 100 else result
            _log(f"识别成功，返回 {len(result)} 字符: {preview}")
            return [TextContent(type="text", text=result)]
        except FileNotFoundError as e:
            _log(f"文件不存在: {e}")
            return [TextContent(type="text", text=str(e))]
        except Exception as e:
            _log(f"识别失败: {e}")
            return [TextContent(type="text", text=f"识别失败: {e}")]

    _log(f"未知工具: {name}")
    return [TextContent(type="text", text=f"未知工具: {name}")]


async def main() -> None:
    _log("MCP Server 启动，等待 Claude Code 连接...")
    async with stdio_server() as (read_stream, write_stream):
        _log("Claude Code 已连接 (stdio)")
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
