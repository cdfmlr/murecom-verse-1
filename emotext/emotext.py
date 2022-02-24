import csv
import os.path
import pathlib
import pickle
from collections import namedtuple, OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import List, IO

import jieba
import jieba.analyse

DICT_FILE_NAME = 'dict.csv'
PKL_FILE_NAME = 'words.pkl'

# 情感分类: 7 大类, 21 小类
categories = {
    'happiness': ['PA', 'PE'],  # 乐
    'goodness': ['PD', 'PH', 'PG', 'PB', 'PK'],  # 好
    'anger': ['NA'],  # 怒
    'sadness': ['NB', 'NJ', 'NH', 'PF'],  # 哀
    'fear': ['NI', 'NC', 'NG'],  # 惧
    'dislike': ['NE', 'ND', 'NN', 'NK', 'NL'],  # 恶
    'surprise': ['PC'],  # 惊
}

# 所有 21 小类情感
emotions = categories['happiness'] + categories['goodness'] + \
           categories['anger'] + categories['sadness'] + categories['fear'] + \
           categories['dislike'] + categories['surprise']


class Polarity(Enum):
    """极性标注

    每个词在每一类情感下都对应了一个极性。其中，0代表中性，1代表褒义，2代表贬义，3代表兼有褒贬两性。
    注：褒贬标注时，通过词本身和情感共同确定，所以有些情感在一些词中可能极性1，而其他的词中有可能极性为0。
    """
    neutrality = 0
    positive = 1
    negative = 2
    both = 3


@dataclass
class Word:
    """一个词
    lex + emotion 确定一个 Word，
    一个词多个 emotion 视为多个不同的 Word。
    """
    word: str
    emotion: str
    intensity: int  # 情感强度: 分为 1, 3, 5, 7, 9 五档，9 表示强度最大，1 为强度最小。
    polarity: Polarity

    # ['词语', '词性种类', '词义数', '词义序号', '情感分类',
    #  '强度', '极性', '辅助情感分类', '强度', '极性']
    DictRow = namedtuple('DictRow',
                         ['word', 'kind', 'means', 'mean', 'emotion',
                          'intensity', 'polarity', 'emotion2', 'intensity2',
                          'polarity2'])

    @staticmethod
    def from_strs(word: str, emotion: str, intensity: str, polarity: str):
        word = word.strip()
        emotion = emotion.strip()

        intensity = int(intensity or 1)
        intensity = intensity if intensity >= 1 else 1
        intensity = intensity if intensity <= 9 else 9

        polarity = int(polarity or 0)
        polarity = polarity if polarity >= 0 else 0
        polarity = polarity if polarity <= 4 else 0

        return Word(word, emotion, intensity, Polarity(polarity))


class EmotionCountResult:
    """一个情感分析的结果

    emotions: {'情感': 出现次数*情感强度}
    polarity: 整句话的极性: {'褒|贬': 出现次数}
    """
    emotions: OrderedDict
    polarity: OrderedDict

    def __init__(self):
        # OrderedDict([('PA', 0), ('PE', 0), ('PD', 0), ...])
        self.emotions = OrderedDict.fromkeys(emotions, 0)
        # OrderedDict([(<Polarity.neutrality: 0>, 0), (<Polarity.positive: 1>, 0), ...])
        self.polarity = OrderedDict.fromkeys(Polarity, 0)


class Emotions(object):
    """该类使用大连理工大学七大类情绪词典作为情绪分析的情绪词库，对文本进行细粒度情感分析。

    +------+--------------+----------+--------------------------------+
    |      | 表2 情感分类 |          |                                |
    +------+--------------+----------+--------------------------------+
    | 编号 | 情感大类     | 情感类   | 例词                           |
    +------+--------------+----------+--------------------------------+
    | 1    | 乐           | 快乐(PA) | 喜悦、欢喜、笑眯眯、欢天喜地   |
    +------+--------------+----------+--------------------------------+
    | 2    |              | 安心(PE) | 踏实、宽心、定心丸、问心无愧   |
    +------+--------------+----------+--------------------------------+
    | 3    | 好           | 尊敬(PD) | 恭敬、敬爱、毕恭毕敬、肃然起敬 |
    +------+--------------+----------+--------------------------------+
    | 4    |              | 赞扬(PH) | 英俊、优秀、通情达理、实事求是 |
    +------+--------------+----------+--------------------------------+
    | 5    |              | 相信(PG) | 信任、信赖、可靠、毋庸置疑     |
    +------+--------------+----------+--------------------------------+
    | 6    |              | 喜爱(PB) | 倾慕、宝贝、一见钟情、爱不释手 |
    +------+--------------+----------+--------------------------------+
    | 7    |              | 祝愿(PK) | 渴望、保佑、福寿绵长、万寿无疆 |
    +------+--------------+----------+--------------------------------+
    | 8    | 怒           | 愤怒(NA) | 气愤、恼火、大发雷霆、七窍生烟 |
    +------+--------------+----------+--------------------------------+
    | 9    | 哀           | 悲伤(NB) | 忧伤、悲苦、心如刀割、悲痛欲绝 |
    +------+--------------+----------+--------------------------------+
    | 10   |              | 失望(NJ) | 憾事、绝望、灰心丧气、心灰意冷 |
    +------+--------------+----------+--------------------------------+
    | 11   |              | 疚(NH)   | 内疚、忏悔、过意不去、问心有愧 |
    +------+--------------+----------+--------------------------------+
    | 12   |              | 思(PF)   | 思念、相思、牵肠挂肚、朝思暮想 |
    +------+--------------+----------+--------------------------------+
    | 13   | 惧           | 慌(NI)   | 慌张、心慌、不知所措、手忙脚乱 |
    +------+--------------+----------+--------------------------------+
    | 14   |              | 恐惧(NC) | 胆怯、害怕、担惊受怕、胆颤心惊 |
    +------+--------------+----------+--------------------------------+
    | 15   |              | 羞(NG)   | 害羞、害臊、面红耳赤、无地自容 |
    +------+--------------+----------+--------------------------------+
    | 16   | 恶           | 烦闷(NE) | 憋闷、烦躁、心烦意乱、自寻烦恼 |
    +------+--------------+----------+--------------------------------+
    | 17   |              | 憎恶(ND) | 反感、可耻、恨之入骨、深恶痛绝 |
    +------+--------------+----------+--------------------------------+
    | 18   |              | 贬责(NN) | 呆板、虚荣、杂乱无章、心狠手辣 |
    +------+--------------+----------+--------------------------------+
    | 19   |              | 妒忌(NK) | 眼红、吃醋、醋坛子、嫉贤妒能   |
    +------+--------------+----------+--------------------------------+
    | 20   |              | 怀疑(NL) | 多心、生疑、将信将疑、疑神疑鬼 |
    +------+--------------+----------+--------------------------------+
    | 21   | 惊           | 惊奇(PC) | 奇怪、奇迹、大吃一惊、瞠目结舌 |
    +------+--------------+----------+--------------------------------+

    (table above: http://sa-nsfc.com/outcome/resource/item-2.html)
    """

    def __init__(self):
        self.pkl_path = pathlib.Path(__file__).parent.joinpath(PKL_FILE_NAME)
        self.dict_path = pathlib.Path(__file__).parent.joinpath(DICT_FILE_NAME)

        self.words = {}  # {"emotion": [words...]}
        if not self._words_from_pkl():
            self._words_from_dict()
            self._save_words_pkl()

    def _words_from_dict(self):
        self.words = {emo: [] for emo in emotions}
        with open(self.dict_path) as f:
            self._read_dict(f)

    def _read_dict(self, csvfile: IO):
        reader = csv.reader(csvfile)
        reader.__next__()  # skip header

        for row in reader:
            try:
                r = map(lambda x: x.strip(), row)
                r = Word.DictRow(*r)
                if r.emotion in emotions:
                    self.words[r.emotion].append(
                        Word.from_strs(r.word, r.emotion, r.intensity, r.polarity))
                if r.emotion2 and r.emotion2 in emotions:
                    self.words[r.emotion].append(
                        Word.from_strs(r.word, r.emotion2, r.intensity2, r.polarity2))
            except Exception as e:
                print(f"Failed to parse word from dict: {row=}")
                raise e

    def _words_from_pkl(self) -> bool:
        """Fill self.words from the pkl file by self._save_words_pkl()

        XXX: pickle 不安全，但快一些: 读 dict.csv 0.9s, 读 words.pkl 0.7s

        :return: True for read, or False if the pkl file does not exist.
        """
        if not os.path.exists(self.pkl_path):
            return False
        with open(self.pkl_path, 'rb') as f:
            self._read_pkl(f)
        return True

    def _read_pkl(self, pkl_file: IO):
        self.words = pickle.load(pkl_file)

    def _save_words_pkl(self):
        with open(self.pkl_path, 'wb') as f:
            pickle.dump(self.words, f)

    def _find_word(self, w: str) -> List[Word]:
        """在 Emotions.words 中找 w

        :param w: 要找的词
        :return: 找到返回对应的 Word 对象们，找不到返回 []
        """
        result = []
        for emotion, words_of_emotion in self.words.items():
            ws = list(map(lambda x: x.word, words_of_emotion))
            if w in ws:
                result.append(words_of_emotion[ws.index(w)])
        return result

    def emotion_count(self, text) -> EmotionCountResult:
        """简单情感分析。计算各个情绪词 出现次数 * 强度

        :param text:  中文文本字符串
        :return: 返回文本情感统计信息 EmotionCountResult

        """
        result = EmotionCountResult()

        # words = jieba.cut(text)
        keywords = jieba.analyse.extract_tags(text, withWeight=True)

        for word, weight in keywords:
            for w in self._find_word(word):
                result.emotions[w.emotion] += w.intensity * weight
                result.polarity[w.polarity] += weight

        return result
