import httpx
from typing import List, Optional
import torch
from graia.ariadne.entry import Ariadne, Friend, MessageChain, config, Group, LogConfig
from graia.ariadne.message.element import Image, Plain, At, Voice
from graia.ariadne.message.parser.twilight import Twilight, Sparkle
from graia.ariadne.event.message import GroupMessage
import io
import os
import requests
from PIL import Image as ImagePIL
import base64
import time
import cpuinfo
import psutil
import datetime
import pynvml
from io import BytesIO
from pppaimon import RRRRRR
from pathvalidate import sanitize_filepath
import json
import random
import re
import time
# 进度条展示
from progressbar import ProgressBar, Percentage, Bar, Timer, ETA, FileTransferSpeed
from pathlib import Path
from PicClass import *


# 贷款数据库
sdDataArr = []
curFileDir = Path(__file__).absolute().parent  # 当前文件路径
with open(curFileDir / "sd.json", "r", encoding="utf-8") as f:
    plpCtx = json.load(f)
    dataArr = plpCtx['sdData']
    if (dataArr and len(dataArr) > 0):
        sdDataArr = dataArr


def GetSDLeftTimes(nGrouID):
    nGrouID = str(nGrouID)
    if nGrouID in sdDataArr:
        return sdDataArr[nGrouID]
    else:
        sdDataArr[nGrouID] = 10
        return sdDataArr[nGrouID]


def CheckSDLeftTimes(nGrouID):
    nGrouID = str(nGrouID)
    nLeftNum = GetSDLeftTimes(nGrouID)
    if nLeftNum > 0:
        sdDataArr[nGrouID] = nLeftNum - 1
        SaveSDData()
        return True, sdDataArr[nGrouID]
    else:
        return False, sdDataArr[nGrouID]


def AddSDLeftTimes(nGrouID):
    nGrouID = str(nGrouID)
    nLeftNum = GetSDLeftTimes(nGrouID)
    sdDataArr[nGrouID] = nLeftNum + 10
    return sdDataArr[nGrouID]


def SaveSDData():
    with open(curFileDir / "sd.json", 'w', encoding="utf-8")as f:
        data = {"sdData": sdDataArr}
        json.dump(data, f)

# print('=============>'+str(sdDataArr))


proxies = {
    'http': 'http://127.0.0.1:2090',  # 本地的代理转发
    'https': 'https://127.0.0.1:2090'
}
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
           'Chrome/63.0.3239.132 Safari/537.36Name',
           'Referer': 'https://91porn.com'}


# 拉丁文转换为utf-8
def _latin2utf8(strings):
    # 拉丁文转为utf-8标砖字符，例如：\xe6\x9e\x81\  => 猴赛雷
    utf8_stings = strings.encode("latin1").decode(
        "unicode_escape").encode('latin1').decode('utf8')
    return utf8_stings


# 定义随机ip地址
def _random_ip():
    a = random.randint(1, 255)
    b = random.randint(1, 255)
    c = random.randint(1, 255)
    d = random.randint(1, 255)
    ip = (str(a) + '.' + str(b) + '.' + str(c) + '.' + str(d))
    return ip


# page_url = "https://91porn.com/index.php"
page_url = "https://91porn.com/v.php?category=top&viewtype=basic&page="
base_url = 'https://91porn.com/view_video.php?viewkey='


# 个key获取的m3u8信息（普通方法）


def _get_m3u8_info(key):
    header_single = {'Accept-Language': 'zh-CN,zh;q=0.9',
                 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:66.0) Gecko/20100101 Firefox/66.0',
                 'X-Forwarded-For': _random_ip(),
                 'referer': page_url,
                 'Content-Type': 'multipart/form-data; session_language=cn_CN',
                 'Connection': 'keep-alive',
                 'Upgrade-Insecure-Requests': '1', }

    base_req = requests.get(url=base_url + key, headers=header_single,
                            timeout=60, verify=False, proxies=proxies)
    base_req.encoding = base_req.apparent_encoding
    from urllib.parse import unquote
    if base_req.status_code == 200:
        # 获取视频名称
        raw_title = re.findall(r'<h4 class="login_register_header" align=left>(.*?)</h4>',
                               str(base_req.content), re.S)[0]
        raw_title = sorted(raw_title.split(
            " "), key=lambda x: len(x), reverse=True)[0]
        """
         \xe6\x9e\x81\xe5\x93\x81\xe9\x98\xbf\xe5\xa7\xa8\xef\xbc\x8c\xe8\xba\xab\xe6\x9d\x90\xe8\x8b\x97\xe6\
         x9d\xa1\xe5\x85\xb3\xe9\x94\xae\xe5\xa5\xb6\xe5\xad\x90\xe8\xbf\x99\xe4\xb9\x88\xe6\x8c\xba\xe6\x8b\x94\n
        """
        tittle = _latin2utf8(raw_title).strip("\n").strip(" ")
        # 匹配中文
        tittle = re.findall(r'[\u4e00-\u9fa5]+', tittle)
        if isinstance(tittle, list):
            tittle = "".join(tittle)
        # 获取视频内容
        str_encode = re.findall(
            r'strencode2\(\"(.*?)\"\)', str(base_req.content))
        m3u8_base_url = ""
        if len(str_encode) >= 1:
            m3u8_base_url = re.findall(
                r'src=(.*?) type', unquote(str_encode[0]))[0][1:-1]
            # print(f"标题为{tittle}对应的m3u8地址为{m3u8_base_url}")
        else:
            m3u8_base_url = ""
            # print(f"首次获取{tittle}对应的m3u8地址失败，正在重新获取...")

        return m3u8_base_url, tittle
    else:
        return "", ""
        print(f"获取key: {key}视频内容信息失败!")


def getMainVeiwKey(nPageIndex=1):
    header_single = {'Accept-Language': 'zh-CN,zh;q=0.9',
                 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:66.0) Gecko/20100101 Firefox/66.0',
                 'X-Forwarded-For': _random_ip(),
                 'referer': page_url,
                 'Content-Type': 'multipart/form-data; session_language=cn_CN',
                 'Connection': 'keep-alive',
                 'Upgrade-Insecure-Requests': '1', }
                 
    get_page = requests.get(page_url + str(nPageIndex),
                            headers=header_single, proxies=proxies)
    # print("aaaaaaaaaaaaaaa===>"+str(get_page.status_code))
    if get_page.status_code == 200:
        # 更新获取方式（2022-08-11）
        view_keys = re.findall(r'viewkey=(.*?)&', str(get_page.content))
        # 去重
        view_keys = list(set(view_keys))
        # 调试
        # print(f"本次一共收集到{len(view_keys)}个91porn学习视频...")
        # for key in view_keys:
        # print('find:' + key)
    return view_keys


def getOne91VTitleandUrl():
    nPag = random.randint(1, 255)
    view_keys = getMainVeiwKey(nPag)
    nMaxVK = len(view_keys)
    nVindex = random.randint(0, nMaxVK-1)
    vkvk = view_keys[nVindex]
    m3u8_base_url, tittle = _get_m3u8_info(vkvk)
    return m3u8_base_url, tittle


session = requests.Session()

model2 = torch.hub.load(
    "AK391/animegan2-pytorch:main",
    "generator",
    pretrained=True,
    device="cuda",
    progress=False
)

model1 = torch.hub.load("AK391/animegan2-pytorch:main",
                        "generator", pretrained="face_paint_512_v1",  device="cuda")
face2paint = torch.hub.load(
    'AK391/animegan2-pytorch:main', 'face2paint',
    size=512, device="cuda", side_by_side=False
)


def AnimFace(img, ver):
    if ver == 'v1':
        out = face2paint(model2, img)
    else:
        out = face2paint(model1, img)
    return out


def image_to_base64(img, fmt='png'):
    output_buffer = BytesIO()
    img.save(output_buffer, format=fmt)
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    # return f'data:image/{fmt};base64,' + base64_str
    return base64_str

# -------------------------------------------------------------


def get_cpu_info():
    info = cpuinfo.get_cpu_info()  # 获取CPU型号等
    cpu_count = psutil.cpu_count(logical=False)  # 1代表单核CPU，2代表双核CPU
    xc_count = psutil.cpu_count()  # 线程数，如双核四线程
    cpu_percent = round((psutil.cpu_percent()), 2)  # cpu使用率
    try:
        model = info["hardware_raw"]  # cpu型号
    except Exception:
        model = info["brand_raw"]  # cpu型号
    try:  # 频率
        freq = info["hz_actual_friendly"]
    except Exception:
        freq = "null"
    cpu_info = (model, freq, info["arch"], cpu_count, xc_count, cpu_percent)
    return cpu_info


def get_memory_info():
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    total_nc = round((float(memory.total) / 1024 / 1024 / 1024), 3)  # 总内存
    used_nc = round((float(memory.used) / 1024 / 1024 / 1024), 3)  # 已用内存
    available_nc = round(
        (float(memory.available) / 1024 / 1024 / 1024), 3)  # 空闲内存
    percent_nc = memory.percent  # 内存使用率
    swap_total = round((float(swap.total) / 1024 / 1024 / 1024), 3)  # 总swap
    swap_used = round((float(swap.used) / 1024 / 1024 / 1024), 3)  # 已用swap
    swap_free = round((float(swap.free) / 1024 / 1024 / 1024), 3)  # 空闲swap
    swap_percent = swap.percent  # swap使用率
    men_info = (
        total_nc,
        used_nc,
        available_nc,
        percent_nc,
        swap_total,
        swap_used,
        swap_free,
        swap_percent,
    )
    return men_info


def uptime():
    now = time.time()
    boot = psutil.boot_time()
    boottime = datetime.datetime.fromtimestamp(
        boot).strftime("%Y-%m-%d %H:%M:%S")
    nowtime = datetime.datetime.fromtimestamp(
        now).strftime("%Y-%m-%d %H:%M:%S")
    up_time = str(
        datetime.datetime.utcfromtimestamp(now).replace(microsecond=0)
        - datetime.datetime.utcfromtimestamp(boot).replace(microsecond=0)
    )
    alltime = (boottime, nowtime, up_time)
    return alltime


def gpu_Info():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 这里的0是GPU id
    gpuName = str(pynvml.nvmlDeviceGetName(handle), encoding='utf-8')
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    all = (meminfo.total / 1024 / 1024)  # 第二块显卡总的显存大小
    usese = (meminfo.used / 1024 / 1024)  # 这里是字节bytes，所以要想得到以兆M为单位就需要除以1024**2
    free = (meminfo.free / 1024 / 1024)  # 第二块显卡剩余显存大小
    msg = "\n显卡名:"+gpuName+"  \n显存总容量:"+str(all)+"MB  \n已用显存:"+str(usese)+"MB"
    pynvml.nvmlShutdown()
    return msg


def sysinfo():
    cpu_info = get_cpu_info()
    mem_info = get_memory_info()
    up_time = uptime()
    msg = (
        "CPU型号:{0}\r\n频率:{1}\r\n架构:{2}\r\n核心数:{3}\r\n线程数:{4}\r\n负载:{5}%\r\n{6}\r\n"
        "总内存:{7}G\r\n已用内存:{8}G\r\n空闲内存:{9}G\r\n内存使用率:{10}%\r\n{6}\r\n"
        "swap:{11}G\r\n已用swap:{12}G\r\n空闲swap:{13}G\r\nswap使用率:{14}%\r\n{6}\r\n"
        "开机时间:{15}\r\n当前时间:{16}\r\n已运行时间:{17}"
    )
    full_meg = msg.format(cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3], cpu_info[4], cpu_info[5], "*" * 20, mem_info[0],
                          mem_info[1], mem_info[2], mem_info[3], mem_info[4], mem_info[5], mem_info[6], mem_info[7], up_time[0], up_time[1], up_time[2],)

    gpuInfo = gpu_Info()

    return full_meg+"\n"+gpuInfo


# 图片转为Base64

def toBase64(imgUrl):
    req = session.get(imgUrl)
    img_str = base64.b64encode(req.content).decode()
    return "data:image/png;base64," + img_str


def img2img(imgBase64: str, prompt: str = ""):

    # print("resp=       ========== =>" + imgBase64)

    payload = {
        "init_images": [
            imgBase64
        ],
        "resize_mode": 2,
        "denoising_strength": 0.55,
        "prompt": prompt,
        "seed": -1,
        "batch_size": 1,
        "n_iter": 1,
        "steps": 28,
        "cfg_scale": 11,
        "width": 640,
        "height": 512,
        "negative_prompt": "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
        "s_noise": 1,
        "sampler_index": "Euler a"
    }
    resp = requests.post(
        url="http://127.0.0.1:7861/sdapi/v1/img2img", json=payload)
    # print("resp=       ========== =>"+str(resp))
    resp = resp.json()
    processed = resp["images"][0]

    strPPP = prompt
    if (len(prompt) > 30):
        strPPP = prompt[0: 30]

    strPicName = strPPP + str(time.time())
    strName = sanitize_filepath(strPicName)
    # I assume you have a way of picking unique filenames
    filename = "./stableD/p2p/" + strName + '.png'
    imgdata = base64.b64decode(processed)
    with open(filename, 'wb') as f:
        f.write(imgdata)

    return processed


def txt2img(prompt: str,
            negative_prompt: str = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
            steps: int = 28,
            sampler_index: str = "Euler a",
            seed: int = -1,
            wwwwww: int = 960,
            hhhhhh: int = 512,
            ):

    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": min(steps, 50),
        "sampler_index": sampler_index,
        "cfg_scale": 11,
        "width": wwwwww,
        "height": hhhhhh,
        "seed": seed
    }

    resp = requests.post(
        url="http://127.0.0.1:7861/sdapi/v1/txt2img", json=payload)
    # print("resp==>"+str(resp))
    resp = resp.json()
    processed = resp["images"][0]

    strPPP = prompt
    if (len(prompt) > 30):
        strPPP = prompt[0: 30]

    strPicName = strPPP + str(time.time())
    strName = sanitize_filepath(strPicName)
    # I assume you have a way of picking unique filenames
    filename = "./stableD/t2p/" + strName + '.png'
    imgdata = base64.b64decode(processed)
    with open(filename, 'wb') as f:
        f.write(imgdata)

    return processed


#################################################################################################
emojis = {'128516': '20201001', '128512': '20201001', '128578': '20201001', '128579': '20201001', '128515': '20201001', '128513': '20201001', '128522': '20201001', '128519': '20201001', '128518': '20201001', '128514': '20201001', '129315': '20201001', '128517': '20201001', '128521': '20201001', '128535': '20201001', '128537': '20201001', '128538': '20201001', '128536': '20201001', '128525': '20201001', '129392': '20201001', '129321': '20201001', '128539': '20201001', '128541': '20201001', '128523': '20201001', '128540': '20201001', '129322': '20201001', '129297': '20201001', '129394': '20201001', '129303': '20201001', '129323': '20201001', '129325': '20201001', '129762': '20211115', '129763': '20211115', '129296': '20201001', '128566': '20201001', '129300': '20201001', '129320': '20201001', '128528': '20201001', '128529': '20201001', '128566-8205-127787-65039': '20210218', '128527': '20201001', '128524': '20201001', '128556': '20201001', '128580': '20201001', '128530': '20201001', '128558-8205-128168': '20210218', '128542': '20201001', '128532': '20201001', '129317': '20201001', '129393': '20201001', '128554': '20201001', '128564': '20201001', '129316': '20201001', '128567': '20201001', '129298': '20201001', '129301': '20201001', '129314': '20201001', '129326': '20201001', '129319': '20201001', '129397': '20201001', '129398': '20201001', '128565': '20201001', '129396': '20201001', '129760': '20211115', '129327': '20201001', '129312': '20201001', '129395': '20201001', '129400': '20201001', '129488': '20201001', '128526': '20201001', '128533': '20201001', '129764': '20211115', '128543': '20201001', '128577': '20201001', '128558': '20201001', '128559': '20201001', '128562': '20201001', '128551': '20201001', '128550': '20201001', '128552': '20201001', '128560': '20201001', '128561': '20201001', '128563': '20201001', '129761': '20211115', '129765': '20211115', '129401': '20211115', '129402': '20201001', '129299': '20201001', '128546': '20201001', '128557': '20201001', '128549': '20201001', '128531': '20201001', '128555': '20201001', '128553': '20201001', '128547': '20201001', '128534': '20201001', '128544': '20201001', '128545': '20201001', '129324': '20201001', '128548': '20201001', '128520': '20201001', '128127': '20201001', '128169': '20201001', '128128': '20201001', '128125': '20201001', '128123': '20201001', '129302': '20201001', '129313': '20201001',
          '127875': '20201001', '127801': '20201001', '127804': '20201001', '127799': '20201001', '127800': '20210218', '128144': '20201001', '127797': '20201001', '127794': '20201001', '129717': '20211115', '127821': '20201001', '129361': '20201001', '127798-65039': '20201001', '127820': '20211115', '127827': '20210831', '127819': '20210521', '127818': '20211115', '127874': '20201001', '129473': '20201001', '129472': '20201001', '127789': '20201001', '127838': '20210831', '9749': '20201001', '127869-65039': '20201001', '129440': '20201001', '9924': '20201001', '127882': '20201001', '127880': '20201001', '128142': '20201001', '128139': '20201001', '128148': '20201001', '128140': '20201001', '128152': '20201001', '128159': '20201001', '128149': '20201001', '128158': '20201001', '128147': '20201001', '128151': '20201001', '10084-65039-8205-129657': '20210218', '10084-65039': '20201001', '129505': '20201001', '128155': '20201001', '128154': '20201001', '128153': '20201001', '128156': '20201001', '129294': '20201001', '129293': '20201001', '128420': '20201001', '128150': '20201001', '128157': '20201001', '127873': '20211115', '127895-65039': '20201001', '127942': '20211115', '129351': '20220203', '129352': '20220203', '129353': '20220203', '127941': '20220203', '128240': '20201001', '127911': '20210521', '128175': '20201001', '128064': '20201001', '127751': '20210831', '128371-65039': '20201001', '129668': '20210521', '128302': '20201001', '128293': '20201001', '128081': '20201001', '128049': '20201001', '129409': '20201001', '128047': '20220110', '128053': '20201001', '128584': '20201001', '128055': '20201001', '129412': '20210831', '129420': '20201001', '128016': '20210831', '129433': '20201001', '128038': '20210831', '129417': '20210831', '128039': '20211115', '129415': '20201001', '128029': '20201001', '128375-65039': '20201001', '128034': '20201001', '128025': '20201001', '128060': '20201001', '128059': '20210831', '128040': '20201001', '129445': '20201001', '128048': '20201001', '128045': '20201001', '129428': '20201001', '128054': '20211115', '128041': '20211115', '129437': '20211115', '128012': '20210218', '129410': '20210218', '128031': '20210831', '127757': '20201001', '127774': '20201001', '127775': '20201001', '11088': '20201001', '127772': '20201001', '127771': '20201001', '128171': '20201001', '127752': '20201001', '9729-65039': '20201001', }


def emoji_to_codes(c): return list(map(lambda x: ord(x), c))


def codes_to_unicode(codes): return "-".join(
    list(map(lambda code: f"u{code:x}", codes))
)


def mix_emoji(emoji_1: str, emoji_2: str) -> Optional[bytes]:
    codes_1 = emoji_to_codes(emoji_1)

    try:
        path = emojis["-".join([str(i) for i in codes_1])]
    except KeyError:
        return "error"

    unicode_1 = codes_to_unicode(codes_1)
    unicode_2 = codes_to_unicode(emoji_to_codes(emoji_2))

    url = f"https://www.gstatic.com/android/keyboard/emojikitchen/{path}/{unicode_1}/{unicode_1}_{unicode_2}.png"
    try:
        resp = httpx.get(url, timeout=10)
        resp.raise_for_status()
    except Exception:
        return "error"
    else:
        return resp.content


def mix_emoji_help() -> str:
    chars = []
    for code in emojis.keys():
        items = [chr(int(i)) for i in code.split('-')]
        chars.append(''.join(items))

    return '支持的emoji有：' + ', '.join(chars)

#################################################################################################


app = Ariadne(
    config(
        verify_key="ServiceVerifyKey",
        account=157199224,
    ), log_config=LogConfig("INFO")
)
app.log_config.clear()


floatTxt2img = -1
floatCDCD = 20

floatCDCDMixDelta = -1
floatCDCDMix = 20

nRandomCodeLenth = 8
random_str = ""


def generate_random_str(randomlength=nRandomCodeLenth):
    global random_str
    """
    生成一个指定长度的随机字符串
    """
    base_str = 'abcdefghigklmnopqrstuvwxyz0123456789'
    length = len(base_str) - 1
    for i in range(randomlength):
        random_str += base_str[random.randint(0, length)]
    return random_str


@app.broadcast.receiver(GroupMessage)
async def group_message_listener(app: Ariadne, group: Group,  message: MessageChain, event: GroupMessage):

    global floatTxt2img
    global floatCDCD
    global floatCDCDMixDelta
    global floatCDCDMix
    global random_str
    global nRandomCodeLenth
    strCont = str(message)
    nMasterQQ = 1973381512
    bWihteUser = False
    if event.sender.id == nMasterQQ:
        bWihteUser = True
    nSendID = event.sender.id

    if random_str == strCont:
        nLeftTT = AddSDLeftTimes(nSendID)
        random_str = ""
        return app.send_message(group, "输入成功,获取10次机会,剩余"+str(nLeftTT), quote=message)

    if At(157199224) in message:
        if "帮助" in strCont:
            repl00 = MessageChain(Plain("\n1,发送pet获取头像表情包功能的菜单\n\n2,发送'生成图'或者'/ai text'加英文英文英文英文英文英文英文英文英文英文英文关键词,使用AI合成图\n\n" +
                                        "3,发送'图生图'或者'/ai image'加一张图片和英文英文英文英文英文英文英文英文英文英文英文关键词描述,使用AI的以图合成图功能\n\n4,发送'高清'加一张图片,使用AI的超分辨率功能,提升4倍分辨率\n\n" +
                                        "5,发送'动漫化'加一张图,使用AI的动漫风格合成功能\n\n6,发送'系统信息'查看机器人使用状态\n\n7,发送'语音合成'加需要合成的文字,可以使用派蒙声优的声线说出你需要的文字\n\n以上功能不需要@机器人\n\n以上功能不需要@机器人\n\n以上功能不需要@机器人\n\n"))
            return app.send_message(group, repl00, quote=message)
    else:
        if "生成图" in strCont or "/ai text" in strCont:
            bCan, nLeft = CheckSDLeftTimes(nSendID)
            if bCan == False:
                if len(random_str) != nRandomCodeLenth:
                    random_str = generate_random_str(nRandomCodeLenth)
                strImage222 = ImgText(random_str).draw_text()
                repl00 = MessageChain(Plain("你的剩余数量不足, \n输入" + str(nRandomCodeLenth) +
                                      "位验证码可获得10次转换机会,\n尽快输入,\n第一个输入正确的玩家才能获取机会"), Image(base64=strImage222))
                return app.send_message(group, repl00, quote=message)
            fNowTime = time.time()
            dtTime = fNowTime - floatTxt2img
            dtTime = int(dtTime)
            strCont = strCont.lower()
            if "nude" in strCont or "nipple" in strCont or "breast" in strCont or "pussy" in strCont or "nsfw" in strCont or "sex" in strCont or "ass" in strCont or "disrobe" in strCont:
                if bWihteUser == False:
                    repl00 = MessageChain(Plain("\n你想要色色?要我喊管理过来吗?"))
                    return app.send_message(group, repl00, quote=message)
            if dtTime < floatCDCD:
                leftTime = floatCDCD-dtTime
                repl1 = MessageChain(Plain(
                    "\n" + str(floatCDCD)+" 秒CD一张图,等等再弄吧,还剩"+str(leftTime)+"秒"))
                if bWihteUser == False:
                    return app.send_message(group, repl1, quote=message)
            floatTxt2img = time.time()
            strSsss = strCont.replace("生成图 ", "")
            strSsss = strCont.replace("生成图", "")
            strSsss = strSsss.replace("/ai text ", "")
            strSsss = strSsss.replace("/ai text", "")
            strSsss = strSsss.replace("，", ",")
            www = 960
            hhh = 512
            if "竖屏" in strCont:
                www = 512
                hhh = 960
            strSsss = strSsss.replace("竖屏", "")
            strSsss = strSsss.replace("横屏", "")
            if len(strSsss) >= 1:
                ch = strSsss[0]
                if u'\u4e00' <= ch <= u'\u9fff':
                    return app.send_message(group, "日你妈弱智, 发英文关键词", quote=message)
            if len(strSsss) >= 2:
                ch = strSsss[1]
                if u'\u4e00' <= ch <= u'\u9fff':
                    return app.send_message(group, "日你妈弱智, 发英文关键词", quote=message)

            repl2 = MessageChain(At(event.sender.id),
                                 Plain("\n收到,生成图片中,可能一分钟后成功...."))
            await app.send_message(group, repl2)

            img64 = ""
            try:
                img64 = txt2img(strSsss, wwwwww=www, hhhhhh=hhh)
            except:
                return app.send_message(group, "合成出错,也可能机器人在维护改bug中", quote=message)
            # if len(strSsss) > 30:
            #     strSsss = strSsss[0: 30]+"....."
            strRepl = "\n你要求的图生成好了,剩余次数="+str(nLeft)
            repl3 = MessageChain(Image(base64=img64), Plain(strRepl))
            floatTxt2img = time.time()
            return app.send_message(group, repl3, quote=message)
        if "图生图" in strCont or "/ai image" in strCont:
            bCan, nLeft = CheckSDLeftTimes(nSendID)
            if bCan == False:
                if len(random_str) != nRandomCodeLenth:
                    random_str = generate_random_str(nRandomCodeLenth)
                strImage222 = ImgText(random_str).draw_text()
                repl00 = MessageChain(Plain("你的剩余数量不足, \n输入"+str(nRandomCodeLenth) +
                                      "位验证码可获得10次转换机会,\n尽快输入,\n第一个输入正确的玩家才能获取机会"), Image(base64=strImage222))
                return app.send_message(group, repl00, quote=message)

            strCont = strCont.lower()
            if "nude" in strCont or "nipple" in strCont or "breast" in strCont or "pussy" in strCont or "nsfw" in strCont or "sex" in strCont or "ass" in strCont or "disrobe" in strCont:
                if bWihteUser == False:
                    repl00 = MessageChain(Plain("\n你想要色色?要我喊管理过来吗?"))
                    return app.send_message(group, repl00, quote=message)

            imageArr = event.message_chain[Image]
            # print("=============>"+str(imageArr))
            if len(imageArr) == 1:
                strAlllll = ""
                strArr = event.message_chain[Plain]
                # print("=============>"+str(strArr))
                for strP in strArr:
                    strAlllll += str(strP)
                # print("=======strAlllll======>" + strAlllll)
                strAlllll = strAlllll.replace("图生图 ", "")
                strAlllll = strAlllll.replace("图生图", "")
                strAlllll = strAlllll.replace("/ai image ", "")
                strAlllll = strAlllll.replace("/ai image", "")
                strUrlqq = imageArr[0].url

                if len(strAlllll) >= 1:
                    ch = strAlllll[0]
                    if u'\u4e00' <= ch <= u'\u9fff':
                        return app.send_message(group, "日你妈弱智, 发英文关键词", quote=message)
                if len(strAlllll) >= 2:
                    ch = strAlllll[1]
                    if u'\u4e00' <= ch <= u'\u9fff':
                        return app.send_message(group, "日你妈弱智, 发英文关键词", quote=message)

                # print("=======strUrlqq======> " + strUrlqq)
                html = toBase64(strUrlqq)

                repl2 = MessageChain(At(event.sender.id),
                                     Plain("\n收到,生成图片中,可能一分钟后成功...."))
                await app.send_message(group, repl2)
                img64 = ""
                try:
                    img64 = img2img(html, strAlllll)
                except:
                    return app.send_message(group, "合成出错,也可能机器人在维护改bug中", quote=message)
                strRepl = "\n你要求的图生成好了,剩余次数="+str(nLeft)
                repl3 = MessageChain(Image(base64=img64), Plain(strRepl))
                return app.send_message(group, repl3, quote=message)
        if "画质" in strCont or "高清" in strCont:
            imageArr = event.message_chain[Image]
            # print("=============>"+str(imageArr))
            if len(imageArr) == 1:
                strRRRR = "realesrgan-x4plus"
                if "漫" in strCont:
                    strRRRR = "realesrgan-x4plus-anime"
                nAAAAAAAA = time.time()
                strUrlqq = imageArr[0].url
                html = requests.get(strUrlqq)
                strPathName = './realesrgan/' + str(nAAAAAAAA) + '.png'
                with open(strPathName, 'wb') as file:
                    file.write(html.content)

                imgTemp = ImagePIL.open(strPathName)
                imgTempSize = imgTemp.size
                maxSize = max(imgTempSize)  # 图片的长边
                if maxSize > 1600:
                    repl31 = MessageChain(
                        At(event.sender.id), Plain("原图分辨率有边长超过1600,不再使用高清化"))
                    return app.send_message(group, repl31)
                strADB = "realesrgan-ncnn-vulkan.exe -i " + \
                    strPathName + " -o ./realesrgan/output.png -n " + strRRRR
                os.system(strADB)
                strRepl = "\n画质提升好了,模型:" + strRRRR + \
                    "\n点击查看原图看效果\n默认使用真实世界模型\n发送'高清漫画'可以使用动漫专属模型"
                repl3 = MessageChain(
                    Image(path="./realesrgan/output.png"), Plain(strRepl))
                return app.send_message(group, repl3, quote=message)
        if "系统信息" in strCont:
            strRepl = sysinfo()
            repl3 = MessageChain(At(event.sender.id), Plain(strRepl))
            return app.send_message(group, repl3)
        if "动漫化" in strCont:
            imageArr = event.message_chain[Image]
            if len(imageArr) == 1:
                try:
                    nAAAAAAAA = time.time()
                    strUrlqq = imageArr[0].url
                    html = requests.get(strUrlqq)
                    strPathName = './anim/' + str(nAAAAAAAA) + '.png'
                    with open(strPathName, 'wb') as file:
                        file.write(html.content)
                    imgTemp = ImagePIL.open(strPathName)
                    strMod = "v1"
                    if "v2" in strCont:
                        strMod = "v2"
                    plImg = AnimFace(imgTemp, strMod)
                    image_data = image_to_base64(plImg)
                    strRepl = "好了, 默认v1模型, 可以发送'动漫化v2'使用另一个模型"
                    repl3 = MessageChain(
                        Image(base64=image_data), Plain(strRepl))
                    return app.send_message(group, repl3, quote=message)
                except:
                    return app.send_message(group, "出错了,换张图吧", quote=message)
        if "合成语音" in strCont or "语音合成" in strCont:
            strAlllll = strCont.replace("语音合成 ", "")
            strAlllll = strAlllll.replace("语音合成", "")
            strAlllll = strAlllll.replace("合成语音 ", "")
            strAlllll = strAlllll.replace("合成语音", "")
            try:
                strName = RRRRRR(strAlllll)
                repl3 = MessageChain(
                    Voice(path="./speach/" + strName + ".wav"))
                return app.send_message(group, repl3)
            except BaseException as eee:
                return app.send_message(group, "合成失败,你发的什么玩意"+str(eee), quote=message)
        if "91视频" in strCont:
            if bWihteUser:
                try:
                    m3u8_base_url, tittle = getOne91VTitleandUrl()
                    if len(m3u8_base_url) > 1 and len(tittle) > 1:
                        return app.send_message(group, "随机发送一个视频\n 标题=>"+tittle+"\n 地址=>"+m3u8_base_url,
                                                quote=message)
                except:
                    asdas = 1
            else:
                return app.send_message(group, "你没权限", quote=message)
        if len(strCont) == 2:
            fNowTime = time.time()
            dtTime = fNowTime - floatCDCDMixDelta
            dtTime = int(dtTime)
            if dtTime < floatCDCDMix:
                leftTime = floatCDCDMix-dtTime
                repl1 = MessageChain(Plain(
                    "\n防刷屏" + str(floatCDCDMix)+"秒CD一张图,等等再弄吧,还剩"+str(leftTime)+"秒"))
                if bWihteUser == False:
                    return app.send_message(group, repl1, quote=message)
            emoji_1 = strCont[0]
            emoji_2 = strCont[1]
            if '\u9fff' < emoji_1 < '\U0001fab5' and '\u9fff' < emoji_2 < '\U0001fab5':
                content = mix_emoji(emoji_1, emoji_2)
                if content == "error":
                    aqw = 1
                    return aqw
                    # return app.send_message(group, "合成失败,不支持,发送'表情合成帮助'查看可以用的emoji", quote=message)
                img_str = base64.b64encode(content).decode()
                repl3 = MessageChain(Image(base64=img_str), Plain("合成好了,发送'表情合成帮助'查看可以用的emoji"))
                floatCDCDMixDelta = time.time()
                return app.send_message(group, repl3, quote=message)
        if "表情" in strCont and "帮助" in strCont:
            strHHH = mix_emoji_help()
            return app.send_message(group, strHHH, quote=message)

Ariadne.launch_blocking()
