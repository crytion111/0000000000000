import torch

from pppaimon import commons
from pppaimon import utils
from pppaimon.models import SynthesizerTrn
from pppaimon.text.symbols import symbols
from pppaimon.text import text_to_sequence

from pathvalidate import sanitize_filepath

from scipy.io.wavfile import write


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

hps = utils.get_hparams_from_file("./pppaimon/configs/biaobei_base.json")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint('./pppaimon/1190G_1434000.pth', net_g, None)
import soundfile as sf



def runGe(text111:str, filename:str):
    length_scale = 1 #@{type:"slider", min:0.1, max:3, step:0.05}
    if "快语速" in text111:
        length_scale = 0.7
    elif "慢语速"in text111:
        length_scale = 1.5
    
    text111 = text111.replace("慢语速","")
    text111 = text111.replace("快语速","")

    text = text111 
    
    audio_path = f'./speach/{filename}.wav'
    stn_tst = get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=length_scale)[0][0,0].data.cpu().float().numpy()

    sf.write(audio_path,audio,samplerate=hps.data.sampling_rate)


def RRRRRR(text:str):
    strName = "temp"
    if len(text) <= 10:
        strName = text
    else:
        strName = text[0:10]
    # print("strName==>"+strName)
    strName = sanitize_filepath(strName)

    runGe(text, strName)
    return strName