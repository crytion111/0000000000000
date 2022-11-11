import base64
import io
import json
import random
import sys
from typing import Tuple
from mirai import Mirai, FriendMessage, WebSocketAdapter, GroupMessage, At,MessageChain,Plain,Image
import numpy as np
import requests
from PIL import ImageEnhance
from PIL import Image as PILImage



nMasterQQ = 1973381512
nBotQQ = 1209916110

class __redirection__:
    def __init__(self):
        self.buff=''
        self.__console__=sys.stdout
        
    def write(self, output_stream):
        self.buff+=output_stream
        
    def flush(self):
        self.buff=''
        
    def reset(self):
        sys.stdout=self.__console__




np.seterr(divide="ignore", invalid="ignore")

#---------------------------------------TJTKTK

def resize_image(im1: PILImage.Image, im2: PILImage.Image, mode: str) -> Tuple[PILImage.Image, PILImage.Image]:
    """
    统一图像大小
    """
    _wimg = im1.convert(mode)
    _bimg = im2.convert(mode)

    wwidth, wheight = _wimg.size
    bwidth, bheight = _bimg.size

    width = max(wwidth, bwidth)
    height = max(wheight, bheight)

    wimg = PILImage.new(mode, (width, height), 255)
    bimg = PILImage.new(mode, (width, height), 0)

    wimg.paste(_wimg, ((width - wwidth) // 2, (height - wheight) // 2))
    bimg.paste(_bimg, ((width - bwidth) // 2, (height - bheight) // 2))

    return wimg, bimg

# 感谢老司机
# https://zhuanlan.zhihu.com/p/32532733
def color_car(
    wimg: PILImage.Image,
    bimg: PILImage.Image,
    wlight: float = 1.0,
    blight: float = 0.3,
    wcolor: float = 0.01,
    bcolor: float = 0.7,
    chess: bool = False,
) -> PILImage.Image:
    """
    发彩色车
    :param wimg: 白色背景下的图片
    :param bimg: 黑色背景下的图片
    :param wlight: wimg 的亮度
    :param blight: bimg 的亮度
    :param wcolor: wimg 的色彩保留比例
    :param bcolor: bimg 的色彩保留比例
    :param chess: 是否棋盘格化
    :return: 处理后的图像
    """
    wimg = ImageEnhance.Brightness(wimg).enhance(wlight)
    bimg = ImageEnhance.Brightness(bimg).enhance(blight)

    wimg, bimg = resize_image(wimg, bimg, "RGB")

    wpix = np.array(wimg).astype("float64")
    bpix = np.array(bimg).astype("float64")

    if chess:
        wpix[::2, ::2] = [255., 255., 255.]
        bpix[1::2, 1::2] = [0., 0., 0.]

    wpix /= 255.
    bpix /= 255.

    wgray = wpix[:, :, 0] * 0.334 + wpix[:, :, 1] * 0.333 + wpix[:, :, 2] * 0.333
    wpix *= wcolor
    wpix[:, :, 0] += wgray * (1. - wcolor)
    wpix[:, :, 1] += wgray * (1. - wcolor)
    wpix[:, :, 2] += wgray * (1. - wcolor)

    bgray = bpix[:, :, 0] * 0.334 + bpix[:, :, 1] * 0.333 + bpix[:, :, 2] * 0.333
    bpix *= bcolor
    bpix[:, :, 0] += bgray * (1. - bcolor)
    bpix[:, :, 1] += bgray * (1. - bcolor)
    bpix[:, :, 2] += bgray * (1. - bcolor)

    d = 1. - wpix + bpix

    d[:, :, 0] = d[:, :, 1] = d[:, :, 2] = d[:, :, 0] * 0.222 + d[:, :, 1] * 0.707 + d[:, :, 2] * 0.071

    p = np.where(d != 0, bpix / d * 255., 255.)
    a = d[:, :, 0] * 255.

    colors = np.zeros((p.shape[0], p.shape[1], 4))
    colors[:, :, :3] = p
    colors[:, :, -1] = a

    colors[colors > 255] = 255

    return PILImage.fromarray(colors.astype("uint8")).convert("RGBA")

def mkTKPic(strP1, strP2):
    im1 = PILImage.open(strP1)
    im2 = PILImage.open(strP2)
    im1 = im1.resize(im2.size, PILImage.ANTIALIAS)
    buffered = io.BytesIO()
    color_car(im1, im2).save(buffered, format="png")
    return base64.b64encode(buffered.getvalue()).decode()


# -------------------------------------------------------------

        
urls = 'http://openapi.turingapi.com/openapi/api/v2'
api_key = "059f9782bab24de6a63d4083590a803b"
# 回复
def chatAI(data = "你好"):
    data_dict = {
        "reqType": 0,
        "perception": {
            "inputText": {
                "text": data
            },
        },
        "userInfo": {
            "apiKey": api_key,
            "userId": "633677"
        }
    }
    try:
        result = requests.post(urls, json=data_dict)
        content = result.text
        # print(content)
        ans = json.loads(content)
        text = ans['results'][0]['values']['text']
        # print('Niubility:',text)  # 机器人取名就叫Niubility
        return text
    except:
        return "error_chatAI"


# -- yuban10703 -------------------------------------------------------------------------------#--------------------------------------------------------------------------------------#---------------------------------------------------------------------------------
def getYubanPic(tags = "", pon="0"):
    try:
        # 0:safe,1:nos,2:all
        api_url = 'https://setu.yuban10703.xyz/setu?r18=' + \
            str(pon) + '&num=1&tags=' + tags
        #data = {'r18': 0, 'num': 1, "tags":[]}
        req = requests.get(api_url).text

        if(json.loads(req)["detail"] and json.loads(req)["detail"][0] == "色"):
            return "获取图片出错===> 老子没找到"
        else:
            datas = json.loads(req)["data"]
            dataatata = datas[0]
            picOriginalUrl = dataatata["urls"]["original"]
            picLargeUrl = dataatata["urls"]["large"].replace(
                "_webp", "").replace("i.pximg.net", "i.pixiv.re")
            picMediumUrl = dataatata["urls"]["medium"].replace(
                "_webp", "").replace("i.pximg.net", "i.pixiv.re")
            picOriginalUrl_Msg = dataatata["urls"]["original"].replace(
                "i.pximg.net", "i.pixiv.re")

            # print("//////====>picOriginalUrl_Msg=> " + str(picMediumUrl))
            return picMediumUrl
    except Exception as e:
        return "获取图片出错===>" + str(e)+" tags "+tags+" pn "+pon


# -------------------------------------------------------------

session = requests.Session()

# 图片转为Base64
def toBase64(imgUrl):
    req = session.get(imgUrl)
    return base64.b64encode(req.content).decode()


def CheckYYYY(strB64):
    pay_load = {
        'api_key': "X5CYnsaJJCgMJXMPo9JGyHWfsqWx80gr",
        'api_secret': "K1zHwlcl1RalyoLOH3vWLsouLDjPcl69",
        'return_attributes': 'age,gender,skinstatus,beauty,smiling',
        'image_base64': strB64
    }
    r = requests.post(
        "https://api-cn.faceplusplus.com/facepp/v3/detect", data=pay_load)
    # print("+++++++++++++++++++++++++++++++++++detect_face++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # print(img_file_path)
    # print(r.status_code)
    # print("====>" + r.text)
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    try:
        r_json = json.loads(r.text)
        if r.status_code == 200:
            faceNum = r_json["face_num"]
            if faceNum > 1:
                return "暂时只能分辨一张脸, 图中的人脸数="+str(faceNum)
            elif faceNum == 0:
                return "传的图没有人脸,也可能传的是动漫图,二次元真恶心"
            else:
                faceData = r_json["faces"][0]
                attributes = faceData["attributes"]
                strXB = attributes["gender"]["value"]
                strMNMNM = "男性"
                if strXB == "Female":
                    strMNMNM = "女性"
                nAge = attributes["age"]["value"]
                skinstatus = attributes["skinstatus"]["health"]
                dark_circle = attributes["skinstatus"]["dark_circle"]
                butyScore = 0
                if strXB == "Female":
                    butyScore = attributes["beauty"]["female_score"]
                else:
                    butyScore = attributes["beauty"]["male_score"]

                strSMsm = ""
                if attributes["smile"]["value"] > attributes["smile"]["threshold"]:
                    strSMsm = "正在笑,"
                strRRRSS = "这个人是"+strMNMNM+", 年龄大概" + \
                    str(nAge)+"岁,"+ strSMsm +" 皮肤健康度为:"+str(skinstatus) + \
                    ',黑眼圈程度为:'+str(dark_circle) + \
                    ' \n最终颜值评分为:'+str(butyScore)
            return strRRRSS
        else:
            return "网络错误!!!!!!!!!!!!!"
    except Exception as error:
        return "识别错误==>" + str(error)


# -------------------------------------------------------------

bStartCaiShuZi = False
nRandomNum = -1
nPlayPlayerID = -1


if __name__ == '__main__':
    print("nBotQQ===>"+str(nBotQQ))
    bot = Mirai(nBotQQ, adapter=WebSocketAdapter(
        verify_key='ServiceVerifyKey', host='localhost', port=8080
    ))

    @bot.on(FriendMessage)
    async def on_friend_message(event: FriendMessage):
        if str(event.message_chain) == '你好':
            await bot.send(event, 'Hello World!')

    @bot.on(GroupMessage)
    async def on_group_message(event: GroupMessage):
        global bStartCaiShuZi
        global nRandomNum
        global nPlayPlayerID
        
        strCont = str(event.message_chain)
        if event.sender.id == nBotQQ:
            aasdasd=1
            return aasdasd

        if At(bot.qq) in event.message_chain:
            if "来张色图" in strCont:
                # print("strMessage===>", strCont)
                args = [i.strip() for i in strCont.split(" ") if i.strip()]
                # print("wwqweqw===>", str(args))
                strTag = ""
                if len(args) == 2:
                    strTag = ""
                elif len(args) >= 3:
                    strTag = args[2]

                print("strTag===>", strTag)
                strPic = getYubanPic(strTag)
                if strPic.startswith("获取图片出错"):
                    return bot.send(event, MessageChain([At(event.sender.id), Plain(strPic)]))
                else:
                    return bot.send(event, MessageChain([At(event.sender.id), Image(url = strPic)]))
            elif "买家秀" in strCont:
                indexInt = random.randint(1, 6)
                taobaoUrl = "https://api.uomg.com/api/rand.img3"
                if indexInt == 1:
                    taobaoUrl = "https://api.ghser.com/tao"
                elif indexInt == 2:
                    taobaoUrl = "https://api.uomg.com/api/rand.img3?sort=胖次猫"
                else:
                    taobaoUrl = "https://api.uomg.com/api/rand.img3?sort=七了个三"
                req = requests.get(taobaoUrl)
                filename = "mz" + str(random.random())[2:] + ".png"
                if req.status_code == 200:
                    with open(filename, 'wb') as f:
                        f.write(req.content)
                        return bot.send(event, MessageChain([At(event.sender.id), Image(path = filename)]))
            else:
                strRep = chatAI(str(event.message_chain))
                return bot.send(event, MessageChain([At(event.sender.id), Plain(strRep)]))
        else:
            # print("=asddddddda...>>>", strCont, str(event.sender.id), str(nMasterQQ))
            if "颜值检测" in strCont:
                imageArr = event.message_chain[Image]
                if len(imageArr) == 1:
                    strUrlqq = imageArr[0].url
                    strRes = CheckYYYY(toBase64(strUrlqq))
                    return bot.send(event, MessageChain([At(event.sender.id), strRes]))
            elif "运行代码" in strCont and str(event.sender.id) == str(nMasterQQ):
                args = [i.strip() for i in strCont.split(" ") if i.strip()]
                strHead1 = args[0]
                strSsss = strCont.replace(strHead1, "")
                strEEE = ""
                try:
                    r_obj = __redirection__()
                    sys.stdout = r_obj
                    sys.stdout.flush()
                    strEEE = str(eval(strSsss))
                    strEEE += "\n"+sys.stdout.buff
                except BaseException as err:
                    strEEE = str(err)
                return bot.send(event, MessageChain([At(event.sender.id), strEEE]))
            elif "幻影坦克" in strCont:
            
                imageArr = event.message_chain[Image]
                
                #Image(image_id='{470F828C-357D-646D-7A4A-BA24E6D8EDB7}.jpg', url=HttpUrl('http://gchat.qpic.cn/gchatpic_new/1973381512/932374887-2719264874-470F828C357D646D7A4ABA24E6D8EDB7/0?term=2&is_origin=1', ), imageType='JPG', size=636783, height=1080, width=2412)
                imageNum = len(imageArr)
                
                if imageNum == 2:
                    nIndex = 1
                    for img in imageArr:
                        strUrlqq = img.url
                        html = requests.get(strUrlqq)

                        if nIndex == 2:
                            with open('./22.png', 'wb') as file:
                                file.write(html.content)
                        if nIndex == 1:
                            nIndex = 2
                            with open('./11.png', 'wb') as file:
                                file.write(html.content)
                    #两张图??
                    f1 = '11.png'  # 上层
                    f2 = '22.png'  # 下层
                    base64_str = mkTKPic(f1, f2)
                    return bot.send(event, MessageChain([At(event.sender.id), Image(base64 = base64_str)]))
                


    bot.run()