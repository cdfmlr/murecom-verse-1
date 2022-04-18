import argparse
import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List, Optional, Dict

import joblib
import requests
from aiohttp import web
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.ext.automap import automap_base

import nest_asyncio

import emotext
from emopic.emotic2emotext import emotic2dlut

# TODO: EmotionalRecommenderTrainer
# TODO(saved data): json -> joblib


# region Emotion

Emotion = namedtuple('Emotion', emotext.emotions)


def keys(self):
    return self._fields


def values(self):
    return tuple(self)


Emotion.keys = keys
Emotion.values = values


def softmax_dict(x: dict):
    """softmax_dict is a helper function to softmax values in a dict

    :param x: dict
    :return: softmaxed x
    """
    s = sum(v for v in x.values())
    if s <= 1e-8:
        return x
    for k in x:
        x[k] /= s
    return x


# endregion Emotion

# region data

def RecommendedTrack(track_id: str, track_name: str, artists: List[str]):
    return {
        "track_id": track_id,
        "track_name": track_name,
        "artists": artists,
    }


def EmotionalRecommendResult(seed_emotion: Emotion, distances: List[float], recommended_tracks: List[RecommendedTrack]):
    return {
        "seed_emotion": seed_emotion,
        "distances": distances,
        "recommended_tracks": recommended_tracks,
    }


# endregion data

class EmotionalRecommendBase:
    """EmotionalRecommendBase 是 EmotionalRecommender Trainer/Server 的超累🥱

    提供数据库链接、自动迁移、对象反射
    """

    def __init__(self, db):
        self.db_engine = create_engine(db)

        self.db_automap_base = automap_base()
        self.db_automap_base.prepare(self.db_engine, reflect=True)

        self.Track = self.db_automap_base.classes.tracks
        self.Comment = self.db_automap_base.classes.comments
        self.TrackEmotion = self.db_automap_base.classes.track_emotions

        self.db_session = Session(self.db_engine)

    def get_track_from_db(self, tid):
        t = self.db_session.query(self.Track).where(self.Track.id == tid)[0]
        return t


# region trainer

class EmotionalRecommendTrainer(EmotionalRecommendBase):
    pass


# endregion trainer

# region server

class EmotionalRecommendServer(EmotionalRecommendBase, ABC):
    """Abstract Emotional Recommender Server

    实现了通用的 recommend 流程和 http handler: `ANY /route/this[?k=10]`.

    - 重载 parse_seed 来从请求中获取推荐输入（种子）。
         这里 seed 的定义为：传入 infer_emotion 中推测心情的东西。
    - 重载 infer_emotion 来实现具体的心情推断。
    """

    def __init__(self, db, model_file, data_file):
        super().__init__(db)

        with open(data_file) as df:
            self.data = json.load(df)
        self.model = joblib.load(model_file)

    @abstractmethod
    def parse_seed(self, request: web.Request) -> any:
        """parse_seed 用来实现从 request 中解析得到推荐的输入(种子) seed.

        一种常用的方式是使用 _parse_seed_from_query 解析出请求中的 ?query_name=seed.

        :return: seed: 如果返回必不为空！
        :raise: ValueError: 找不到会抛出一个 ValueError，描述返回给用户的提示
        """
        raise NotImplemented

    def _parse_seed_from_query(self, request: web.Request, query_name: str) -> Optional[str]:
        return request.query.get(query_name)

    def seed_to_str(self, seed):
        """为了打日志方便，可以自定义格式化 seed"""
        return str(seed)

    @abstractmethod
    def infer_emotion(self, seed, *args, **kwargs) -> Emotion:
        raise NotImplemented

    def recommend(self, seed, k=10, *args, **kwargs) -> EmotionalRecommendResult:
        e = self.infer_emotion(seed, *args, **kwargs)

        distances, indices = self.model.kneighbors([e], k)

        # indices => db_tracks => RecommendedTracks
        tracks = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            tid = self.data['ids'][idx]
            t = self.get_track_from_db(tid)  # DB Tracks
            rt = RecommendedTrack(
                track_id=f'ncm-{t.id}',
                track_name=t.name,
                artists=list(map(lambda a: a.name, t.artists_collection)))
            tracks.append(rt)

        result = EmotionalRecommendResult(
            seed_emotion=e,
            distances=distances.tolist(),  # ndarray
            recommended_tracks=tracks)
        return result

    async def handle_http(self, request: web.Request):
        try:
            seed = self.parse_seed(request)

            print(f"{time.ctime(time.time())} {request.method} {request.rel_url}: {self.seed_to_str(seed)}")  # log

            if not seed:
                raise ValueError('Unexpected empty seed.')

            k = int(request.query.get("k") or 10)

            result = self.recommend(seed, k=k)
            # print(f"RESP {result}")
            return web.json_response(result)
        except ValueError as e:
            raise web.HTTPBadRequest(text=str(e))


class EmoTextRecommendServer(EmotionalRecommendServer):
    """从 query 中获取 text 字段作为 seed。
    seed: str 是一段文本，从文本中用 emotext 推断心情。
    """

    def __init__(self, db, model_file, data_file):
        super().__init__(db, model_file, data_file)
        self.emotext = emotext.Emotions()

    def parse_seed(self, request: web.Request) -> any:
        seed = self._parse_seed_from_query(request, "text")
        if not seed:
            raise ValueError("required query: ?text=your_seed_text_here")
        return seed

    def infer_emotion(self, seed, *args, **kwargs):
        r = self.emotext.emotion_count(seed)
        r.emotions = softmax_dict(r.emotions)
        return Emotion(**r.emotions)

    def seed_to_str(self, seed):
        s = str(seed)
        return s.replace("\n", "  ").strip()


class EmoPicRecommendServer(EmotionalRecommendServer, ABC):
    """通过图像分析心情进而推荐音乐。

    因为 emotic 服务有两种请求方式 GET/POST 分别给图像路径和文件内容，
    所以做了这个抽象类，这个类已经在 infer_emotion 中实现公用的 emotic 结果的处理，
    但需要子类重载该类的 parse_seed 从请求中获取需要的东西，
    然后重载 request_emotic 来实现对 emotic 的请求。
    """

    def __init__(self, db, model_file, data_file, emopic_server):
        super().__init__(db, model_file, data_file)
        self.emopic_server = emopic_server

    @abstractmethod
    def request_emotic(self, seed) -> List[Dict]:
        """请求 emotic 服务，顺便要解析 JSON，获取下面这种响应：

        ```
         [{
           "bbox": [65, 15, 190, 186],
           "cat": {"Excitement": 0.16, "Peace": 0.11},
           "cont": [5.97, 6.06, 7.23]
         }]
         ```

        :param seed: parse_seed 得到的 seed
        :return: 解析出的响应
        """
        raise NotImplemented

    def infer_emotion(self, seed, *args, **kwargs) -> Emotion:
        emotic_result = self.request_emotic(seed)

        # 总的情感
        total_emotion = dict.fromkeys(emotext.emotions, 0.0)

        # 人在图片中占的面积
        area = lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        areas = [area(p['bbox']) for p in emotic_result]
        sum_area = sum(areas)

        # 转成 emotext 情感，面积大的人权重大
        for person, person_area in zip(emotic_result, areas):
            person_emo = emotic2dlut(person['cat'], person['cont'])
            weight = person_area / sum_area

            for emo, value in person_emo.items():
                total_emotion[emo] += value * weight

        sum_v = sum(v for v in total_emotion.values())
        for k in total_emotion:
            total_emotion[k] /= sum_v

        return Emotion(**total_emotion)


class GetEmoPicRecommendServer(EmoPicRecommendServer):
    """从本地图像分析心情。这个的局限性在于需要本地的文件，
    也就是说文件上传需要额外实现，再然后把路径传来这里。
    所以这个只适合本地测试使用。

    seed 是 query 中获取到的 img_path: 一个本地的文件路径
    """

    def __init__(self, db, model_file, data_file, emopic_server):
        super().__init__(db, model_file, data_file, emopic_server)

    def parse_seed(self, request: web.Request) -> any:
        seed = self._parse_seed_from_query(request, "img_path")
        if not seed:
            raise ValueError("required query: ?img_path=your_local_seed_img_path")
        return seed

    def request_emotic(self, seed) -> List[Dict]:
        img_path = seed
        try:
            resp = requests.post(self.emopic_server, files={
                'img': (img_path, open(img_path, 'rb')),
            })
        except FileNotFoundError as e:
            raise ValueError(str(e))
        if resp.status_code != 200:
            raise ValueError(resp.text)
        return resp.json()


class PostEmoPicRecommendServer(EmoPicRecommendServer):
    def __init__(self, db, model_file, data_file, emopic_server):
        super().__init__(db, model_file, data_file, emopic_server)

    def parse_seed(self, request: web.Request) -> any:
        nest_asyncio.apply()
        data = asyncio.get_event_loop().run_until_complete(request.post())  # avoid await and async
        img = data['img']
        if not img:
            raise ValueError(f"post data img is required")
        return img

    def request_emotic(self, seed) -> List[Dict]:
        img = seed
        resp = requests.post(self.emopic_server, files={
            'img': (img.filename, img.file),
        })
        if resp.status_code != 200:
            raise ValueError(resp.text)
        return resp.json()

    def seed_to_str(self, seed):
        try:
            return seed.filename
        except:
            pass
        return seed


# endregion server

# region cors


@web.middleware
async def cors_middleware(request, handler):
    """用来解决 cors

    `app = web.Application(middlewares=[cors_middleware])`
    """
    # if request.method == 'OPTIONS':
    #     response = web.Response(text="")
    # else:
    response = await handler(request)

    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = '*'
    response.headers['Access-Control-Allow-Headers'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'

    return response


async def empty_handler(request):
    """给每个 route 配上一个对应的 options empty_handler 即可解决 cors 问题:
    `web.options('...', empty_handler)`
    """
    return web.Response()


# endregion cors

# region cli

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("murecom-intro", description="A emotional music recommender")
    subparsers = parser.add_subparsers(help="sub commands")

    # service
    parser_service = subparsers.add_parser("service")
    parser_service.set_defaults(subcommand="service")

    parser_service.add_argument("--text", action='store_true', help="start emotext recommending service")
    parser_service.add_argument("--pic", action='store_true', help="start emopic recommending service")

    parser_service.add_argument("--db", type=str, help="path to SQLite database", required=True)
    parser_service.add_argument("--model", type=str, help="path to a trained model", required=True)
    parser_service.add_argument("--data", type=str, help="path to saved train-set data", required=True)
    parser_service.add_argument("--emopic-server", type=str, help="url to emopic (EMOTIC) server. required by --pic")

    parser_service.add_argument("--host", type=str, help="server host", default="localhost")
    parser_service.add_argument("--port", type=int, help="listen port", default=8080)

    # train: TODO
    parser_train = subparsers.add_parser("train", help="TODO")
    parser_train.set_defaults(subcommand="train")

    return parser


def run_service(args):
    app = web.Application(middlewares=[cors_middleware])
    app.add_routes([web.options('/', empty_handler)])

    if args.text:
        emotext_recommender = EmoTextRecommendServer(
            args.db, args.model, args.data)
        app.add_routes([
            web.get('/text', emotext_recommender.handle_http),
            web.options('/text', empty_handler)
        ])
    if args.pic:
        assert hasattr(args, 'emopic_server'), (
            '--emopic-server is required by EmoPicRecommendServer (--pic)')
        get_emopic_recommender = GetEmoPicRecommendServer(
            args.db, args.model, args.data, args.emopic_server)
        post_emopic_recommender = PostEmoPicRecommendServer(
            args.db, args.model, args.data, args.emopic_server)
        app.add_routes([
            web.get('/pic', get_emopic_recommender.handle_http),
            web.post('/pic', post_emopic_recommender.handle_http),
            web.options('/pic', empty_handler)
        ])

    web.run_app(app, host=args.host, port=args.port)


def run_train(args):
    raise NotImplementedError


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    # print(args)

    try:
        if args.subcommand == 'train':
            run_train(args)
        if args.subcommand == 'service':
            run_service(args)
    except AttributeError:  # 'Namespace' object has no attribute 'subcommand'
        parser.print_usage()

# endregion cli
