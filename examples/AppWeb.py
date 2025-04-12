from flask import Flask, request, jsonify, render_template, send_file
import asyncio
import os
from datetime import datetime
import aiohttp
import json

app = Flask(__name__)


class Message:
    def __init__(self, content="", instruct_content=None):
        self.content = content
        self.instruct_content = instruct_content


class FashionResearcher:
    async def run(self, require):
        # 这里可以接入实际的AI模型或API
        return Message(
            content="",
            instruct_content={
                "topic": "时尚分析",
                "content": f"根据您的问题：{require}，我为您提供以下时尚建议..."
            }
        )


class FashionerSum:
    async def run(self, message):
        # 这里可以接入实际的AI模型或API
        return Message(
            content="根据分析，我建议您...",
            instruct_content=message.instruct_content
        )


class TutorialAssistantWithActionNode:
    async def run(self, message):
        # 这里可以接入实际的AI模型或API
        return "static/images/fashion_guide.png"


def Download(prompt, link):
    # 确保目录存在
    os.makedirs("static/images", exist_ok=True)

    # 这里可以添加实际的图片生成或下载逻辑
    # 示例：创建一个简单的占位图片
    with open(link, "wb") as f:
        f.write(b"")  # 实际项目中需要替换为真实的图片数据


async def main(require):
    role1 = FashionResearcher()
    information = await role1.run(require)

    role2 = FashionerSum()
    my_dict = {
        "topic": information.instruct_content["topic"],
        "content": information.instruct_content["content"]
    }
    res = await role2.run(Message(content="", instruct_content=my_dict))

    role3 = TutorialAssistantWithActionNode()
    image_path = await role3.run(res)

    # 生成唯一的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fashion_guide_{timestamp}.png"
    full_path = os.path.join("static/images", filename)

    Download("时尚图示", full_path)

    return {
        "content": res.content,
        "image_url": f"/static/images/{filename}"
    }


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        require = request.form.get("require")
        if not require:
            return jsonify({"error": "请输入您的问题"}), 400

        try:
            result = asyncio.run(main(require))
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)