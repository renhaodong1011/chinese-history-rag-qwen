"""
Created on 2025/12/20 by RenHaodong
Description:
    用于爬取中国重大历史事件数据，用于大模型检索增强。
    数据来源： https://zhonghua.5000yan.com/
"""
import requests
from lxml import etree
from typing import *
import os

BASE_URL = 'https://zhonghua.5000yan.com/'
HEADERS = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0'
}
if not os.path.exists("./history_data"):
    os.mkdir("./history_data")

def save_txt(path: str, content: str) -> None:

    with open(path, "w", encoding='utf-8') as f:

        f.write(content)

def fetchall_url() -> Tuple[list, list]:

    response = requests.get(url=BASE_URL, headers=HEADERS)
    response.encoding = response.apparent_encoding
    html = etree.HTML(response.text)
    title_list = html.xpath("//li[@class='list-inline-item px-2 text-body-secondary m-2']/a/text()")
    title_url = html.xpath("//li[@class='list-inline-item px-2 text-body-secondary m-2']/a/@href")
    return (title_list, title_url)

def extract_content(title: str, title_url: str) -> None:

    saving_path = './history_data/' + title + '.txt'
    response = requests.get(url=title_url, headers=HEADERS)
    response.encoding = response.apparent_encoding
    html = etree.HTML(response.text)
    res = html.xpath('//div[@class="grap"]//text()')
    content = "".join(res).replace('\n', '').replace('\t','')
    save_txt(saving_path, content)
    print("{}爬取完毕，来源：{}".format(title, title_url))

if __name__ == '__main__':

    title_list, title_url = fetchall_url()
    for index in range(len(title_list)):
        extract_content(title_list[index], title_url[index])