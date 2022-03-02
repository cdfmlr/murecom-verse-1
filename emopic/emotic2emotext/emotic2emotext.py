from typing import List, Dict

# XXX: 其实我感觉 emotic 的这种分类更科学一点，大连理工这个效果可能不如 emotic，后面可以考虑反转一下

# 大连理工情感本体库 -> emotic cat
dlut2cat = {
    'PA': ['Happiness', 'Pleasure'],
    'PE': ['Confidence', 'Peace'],
    'PD': ['Esteem'],
    'PH': ['Excitement'],
    'PG': ['Anticipation'],
    'PB': ['Affection'],
    'PK': ['Anticipation'],
    'NA': ['Anger'],
    'NB': ['Sadness', 'Suffering', 'Sensitivity'],
    'NJ': ['Aversion', 'Sadness'],
    'NH': ['Sympathy'],
    'PF': ['Engagement'],
    'NI': ['Pain', 'Annoyance'],
    'NC': ['Fear'],
    'NG': ['Embarrassment'],
    'NE': ['Disquietment', 'Fatigue'],
    'ND': ['Aversion', 'Disapproval'],
    'NN': ['Annoyance', 'Disconnection'],
    'NK': ['Yearning'],
    'NL': ['Doubt/Confusion'],
    'PC': ['Surprise']
}

# reverse dlut2cat: emotic cat -> 大连理工情感本体库
cat2dlut = {v: k for k, vs in dlut2cat.items() for v in vs}


def emotic2emotext(cat: Dict[str, float], cont: List[float]):
    dlut = dict.fromkeys(dlut2cat.keys(), 0.0)

    sumv = sum(cat.values())
    for c, v in cat.items():
        dlut[cat2dlut[c]] = v / sumv

    # TODO: 我不知道 cont (VAD) 如何转化。暂时就先不用了。

    return dlut
