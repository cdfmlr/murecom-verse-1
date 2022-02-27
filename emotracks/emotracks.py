#!python3

"""
为数据库中的所有曲目标注情感。

$ cd /path/to/murecom/verse-1/emotracks
$ PYTHONPATH=$PYTHONPATH:/path/to/murecom/verse-1 p3 emotracks.py [start] [end]

start，end 是开始/结束的索引，default start=0, end=len(tracks)
"""

import sys
import time
import json
from math import log

from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session

import emotext

Emo = emotext.Emotions()

# load config

with open('config.json') as f:
    config = json.load(f)

# init db

engine = create_engine(config['db'])
# 用 automap_base 自动反射成库中已有的类型
# https://docs.sqlalchemy.org/en/14/orm/extensions/automap.html
Base = automap_base()
Base.prepare(engine, reflect=True)

# 注意这里反射出的类名是带 s 的，直接是表名。
# 还有 playlist_tracks 是自定义的，automap 不出来 many2many 关系。
# (See https://docs.sqlalchemy.org/en/14/orm/extensions/automap.html#many-to-many-relationships)
# 但这里好像不用 playlist，就先不管了。

Track = Base.classes.tracks
Comment = Base.classes.comments

# new table
# 只是建个表，只需要执行一次。
# 这好像也不比手写 SQL 快耶。。。下次别这么搞了。


if __new_table := False:
    from sqlalchemy import Column, Text, Float, ForeignKey, BigInteger
    from sqlalchemy.orm import relationship


    class TrackEmotion(Base):
        __tablename__ = 'track_emotions'

        track_id = Column(BigInteger, ForeignKey('tracks.id'), primary_key=True)
        emotion = Column(Text, primary_key=True)
        intensity = Column(Float)

        track = relationship('Track')


    Base.metadata.create_all(engine)
    # Track.track_emotions = relationship("TrackEmotion")

    # Base.metadata.create_all(engine)
else:
    TrackEmotion = Base.classes.track_emotions


def track_emotion(t: Track):
    lyrics = t.lyric
    comments = t.comments_collection
    if (not comments) and (not lyrics):
        return

    texts = [c.content for c in comments]
    liked_counts = [c.liked_count for c in comments]
    if lyrics:
        texts.append(lyrics)
        liked_counts.append(1 + sum(liked_counts) * 10)
    # track name
    texts.append(t.name)
    liked_counts.append(1 + sum(liked_counts) * 20)

    liked_weights = [log(1.01 + l, 10) for l in liked_counts]

    track_emotions = emotext.EmotionCountResult().emotions

    for text, weight in zip(texts, liked_weights):
        emo_result = Emo.emotion_count(text)

        # print(text, weight, sorted(emo_result.emotions.items(), key=lambda x: -x[-1]))
        for k, v in emo_result.emotions.items():
            if not v:
                continue
            track_emotions[k] += v * weight

    return track_emotions


def add_emotion(t: Track, verbose=False):
    emo_res = track_emotion(t)
    if not emo_res:
        return

    sum_v = sum(emo_res.values())

    if sum_v:
        t.track_emotions_collection = []

    for e, v in emo_res.items():
        if not v:
            continue
        v = v / sum_v
        t.track_emotions_collection.append(TrackEmotion(emotion=e, intensity=v))
    session.commit()

    if verbose:
        print(t.name, [k for k, v in sorted(emo_res.items(), key=lambda x: -x[-1]) if v])


if __name__ == '__main__':
    is_append = config['append']

    start = 0
    if len(sys.argv) > 1:
        start = int(sys.argv[1])

    session = Session(engine)
    tks = session.query(Track).order_by(Track.pop.desc(), Track.id)

    # total 暂时是 db 中 track 的总数量
    total = tks.count()
    if start > total:
        raise ValueError(f"error: start ({start}) > total ({total})")

    end = total
    if len(sys.argv) > 2:
        end = int(sys.argv[2])

    # 以下 total 是 start 到 end 有多少，即该程序此次运行的任务量
    total = end - start

    print(f"Emotracks for tracks in ncm db, order by pop desc "
          f"(total={total}/{tks.count()}, {is_append=})\n\t{start=}, {end=}")

    i = 0  # counter i = 0...total
    for t in tks[start:end]:
        i += 1
        if i % 20 == 0:
            print(f'📦 {i / total * 100:.2f}%: {i}/{total}')
            time.sleep(1)

        if is_append and t.track_emotions_collection:
            print(f'🪂 exist: {t.name} (emos: {len(t.track_emotions_collection)}), skip')
            continue
        add_emotion(t, verbose=True)
