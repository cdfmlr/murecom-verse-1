from unittest import TestCase

from emotext import *

_test_texts = '''
 我感觉这是唯一一首陈奕迅翻唱的并没有原唱好的歌。
 我要说什么你才会觉得这首歌很棒呢！算了，你们还是先听完吧！争取循环几遍。[钟情]
 下次再遇到喜欢的人 不要冷冰冰 不要说反话
 那人年龄比你老嘛 那人有低保吗
 闷闷的骚
 有谁能告诉我这是什么风格吗？
 告诉我我不是唯一没看过花千骨、但也喜欢这首歌的[大哭]
 心情不好 没人陪你的时候 音乐会陪你😊即便是被人甩了脸再被扔在大街上 我还是会打开网易云 瞬间又有了活下去的勇气。在以后的日子里 没你 没什么不可以。
 记得去年我们的语文老师上课时提到了心流，她跟我们说一个人能感受到心流，是一件幸福的事。在说的时候，她的眼睛里仿佛有星星
 不要再原唱下面提其他人啦[亲亲]
'''


class TestEmotions(TestCase):
    def test_emotions_init(self):
        emo = Emotions()
        l = 0
        for e in emo.words.values():
            l += len(e)
        self.assertEqual(l, 31388, 'emo.words missing')

    def test_emotion_count(self):
        # self.fail()
        emo = Emotions()

        for text in _test_texts.strip().split('\n'):
            result = emo.emotion_count(text)
            result_emotions = {key: value for key, value in result.emotions.items() if value != 0}
            result_polarity = {key.name: value for key, value in result.polarity.items() if value != 0}
            print(f'{text=}\n\t{result_emotions=}\n\t{result_polarity=}')

