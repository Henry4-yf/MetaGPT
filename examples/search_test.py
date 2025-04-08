# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import asyncio
import json

from pydantic import BaseModel
from metagpt.logs import logger
from metagpt.actions import Action, CollectLinks
from metagpt.actions.action_node import ActionNode
from metagpt.roles.role import Role, RoleReactMode
from metagpt.schema import Message


class Report(BaseModel):
    topic: str
    links: dict[str, list[str]] = None
    summaries: dict[str, dict[str, str]] = None
    content: str = ""


# 角色指令
FASHION_SUMMARY_PROMPT = """
你现在是一个时尚信息筛选员,请你提取总结出网页文本中有关时尚的信息。
"""

FASHION_SUMMARY_TEMPLATE = """
信息如下:
{information}

你的任务:
1.由于网页内容可能是英文，你的输出必须是 **中文**，但是其中的一些专有名词可以是英文。
2.重点关注 **时尚趋势、服装种类、流行颜色、搭配方式**。
3.使用 **精炼的语言**，总结约 200 字。
4.输出必须是 **连续文本**，不含空格和换行符。
5.输出的结果中可能会混杂这一些英文单词与符号导致文本不流畅或错误，要认真检查排除错误。
"""

# 定义 ActionNode
FASHION_SUMMARY_NODE = ActionNode(
    key="FashionSummary",
    expected_type=str,
    instruction=FASHION_SUMMARY_PROMPT,
    example="",
)


class FashionTrendExtractor(Action):
    """一个 Action，用于抓取 URL 并提取时尚趋势信息"""

    name: str = "FashionTrendExtractor"

    async def run(self, topic_url_dict: dict[str, list[str]]) -> dict[str, dict[str, str]]:
        """处理包含多个主题和 URL 列表的字典"""
        results = {}

        async def fetch_and_summarize(url):
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                with requests.get(url, headers=headers) as response:
                    if response.status_code != 200:
                        return f"访问失败: {url}, 状态码: {response.status_code}"

                soup = BeautifulSoup(response.text, "html.parser")
                text = soup.get_text()

                # 构造 LLM 提示词
                prompt = FASHION_SUMMARY_TEMPLATE.format(information=text)

                # 发送到 LLM 进行文本摘要
                resp_node = await FASHION_SUMMARY_NODE.fill(context=prompt, llm=self.llm, schema="raw")
                summary = resp_node.content.strip()

                return summary
            except Exception as e:
                return f"抓取失败: {url}, 错误: {e}"

        # 遍历主题并处理 URL
        for topic, urls in topic_url_dict.items():
            tasks = [fetch_and_summarize(url) for url in urls]
            summaries = await asyncio.gather(*tasks)
            results[topic] = {url: summary for url, summary in zip(urls, summaries)}

        return results


DIRECTORY_STRUCTION_FA = """
你现在是一个时尚分析员,请你根据总结的网页信息对相关领域的时尚趋势进行分析
"""
Content_FA = """
信息如下
{require}

对于这个要求
1.输出语言为中文，专有名词可用原语种
2.请根据信息你分析时尚的趋势，给出一些时尚的搭配
3.总结的信息可能存在语意不通的情况，请根据全文适当总结，提取精华
2.可以分段进行回答，说出“具体会穿什么款式衣服，以什么颜色为主”
3.大概,400字左右，语言精炼
4.不要有空格
"""

# 实例化一个ActionNode，输入对应的参数
DIRECTORY_WRITE_FA = ActionNode(
    # ActionNode的名称
    key="Partition",
    # 期望输出的格式
    expected_type=str,
    # 命令文本
    instruction=DIRECTORY_STRUCTION_FA,
    # 例子输入，在这里我们可以留空
    example="",
)


class FashionAna(Action):
    async def run(self, require_data: dict[str, dict[str, str]], *args, **kwargs):
        """
        处理时尚趋势数据，提取文本信息并生成分析内容。
        参数:
        - require_data (dict[str, dict[str, str]]): 时尚趋势数据，格式为 {主题: {URL: 摘要}}。
        返回:
        - str: 生成的时尚趋势分析结果
        """
        if not require_data:
            return "没有可用的时尚趋势数据。"
        require_text = "\n".join(summary for topic in require_data.values() for summary in topic.values())
        prompt = Content_FA.format(require=require_text)
        print("进入大模型")
        resp_node = await DIRECTORY_WRITE_FA.fill(context=prompt, llm=self.llm, schema="raw")
        # 选取 ActionNode.content，获得期望的返回信息
        return resp_node.content


class FashionResearcher(Role):
    name: str = "FashionResearcher"
    profile: str = "Researcher"
    language: str = "en-us"  # 设定默认输出语言为中文
    enable_concurrency: bool = True  # 并发处理

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([CollectLinks, FashionTrendExtractor, FashionAna])
        self._set_react_mode(RoleReactMode.BY_ORDER.value, len(self.actions))

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo

        msg_list = self.rc.memory.get(k=1)
        if not msg_list:
            raise ValueError("Memory is empty, no message to process.")
        msg = msg_list[0]

        if isinstance(msg.instruct_content, Report):
            instruct_content = msg.instruct_content
            topic = instruct_content.topic
        else:
            topic = msg.content


        if isinstance(todo, CollectLinks):
            links = await todo.run(topic, 4, 4)
            ret = Message(
                content="", instruct_content=Report(topic=topic, links=links), role=self.profile, cause_by=todo
            )
        elif isinstance(todo, FashionTrendExtractor):
            # if not msg.instruct_content or "links" not in msg.instruct_content:
            #     raise ValueError("Missing links data in instruct_content.")

            links_data = instruct_content.links
            summaries = await todo.run(links_data)
            with open("SuFashion.json", "w", encoding="utf-8") as f:
                json.dump(summaries, f, ensure_ascii=False, indent=4)
            print("时尚趋势总结完成，结果已保存到 SuFashion.json")
            ret = Message(
                content="", instruct_content=Report(topic=topic, summaries=summaries), role=self.profile, cause_by=todo
            )
        elif isinstance(todo, FashionAna):
            summaries = instruct_content.summaries
            content = await todo.run(summaries)
            ret = Message(
                content="", instruct_content=Report(topic=topic, content=content), role=self.profile, cause_by=todo
            )
        self.rc.memory.add(ret)
        return ret




# class FashionerAna(Role):
#     name: str = "Fashioner"
#     profile: str = "analyze"
#     story: str = ""
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.set_actions([FashionAna])
#         self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
#
#     async def _act(self) -> Message:
#         logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
#         todo = self.rc.todo
#         msg_list = self.rc.memory.get(k=1)
#         if not msg_list:
#             raise ValueError("Memory is empty, no message to process.")
#         require_data = msg_list[0]
#
#         # 检查 require_data 是否为字典
#         if not isinstance(require_data, dict):
#             logger.error("FashionAna 的数据格式错误，期望的是 dict[str, dict[str, str]]。")
#             return Message(content="数据格式错误，无法进行分析。")
#
#         # 调用 FashionAna 动作并获取结果
#         resp_content = await todo.run(require_data=require_data)
#
#         ret = Message(
#             content="", instruct_content=resp_content, role=self.profile, cause_by=todo
#         )
#         # 确保返回的是 Message 对象
#         return ret



async def main(topic: str, language: str = "en-us", enable_concurrency: bool = True):
    researcher = FashionResearcher(language=language, enable_concurrency=enable_concurrency)
    ret = await researcher.run(topic)
    print(ret.instruct_content.content)


if __name__ == "__main__":
    asyncio.run(main("2025年春季的时尚女装趋势", "en-us", True))