import argparse
import json
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from typing import List

import joblib
import requests
from aiohttp import web
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.ext.automap import automap_base

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

class EmotionalRecommenderBase:
    """EmotionalRecommenderBase 是 EmotionalRecommender Trainer/Server 的超累🥱

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


class EmotionalRecommenderServer(EmotionalRecommenderBase, ABC):
    """Abstract Emotional Recommender Server

    实现了通用的 recommend 流程和 http handler。
    重载 seed_name 和 recommender_name 来提供必要的名字:
        seed_name: http handler 接收 `?seed=...` 的具体 seed 名字，
                   如 "text" => `?text=...`
        recommender_name: http handler 注册的子路径名，
                   如 "emotext" => `localhost:8080/emotext`
    重载 infer_emotion 来实现具体的心情推断。
    """

    def __init__(self, db, model_file, data_file):
        super().__init__(db)

        with open(data_file) as df:
            self.data = json.load(df)
        self.model = joblib.load(model_file)

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
        seed = request.query.get(self.seed_name())

        print(f"GET {request.rel_url}: {seed}")

        if not seed:
            raise web.HTTPBadRequest(text=f"seed ({self.seed_name()}) required")
        k = int(request.query.get("k") or 10)

        result = self.recommend(seed, k=k)

        # print(f"RESP {result}")

        # return web.json_response(result, headers={"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "GET, POST, HEAD, OPTIONS"})
        return web.json_response(result)

    @abstractmethod
    def seed_name(self) -> str:
        return "seed"


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


class EmotionalTextRecommenderServer(EmotionalRecommenderServer):
    def seed_name(self) -> str:
        return "text"

    def __init__(self, db, model_file, data_file):
        super().__init__(db, model_file, data_file)
        self.emotext = emotext.Emotions()

    def infer_emotion(self, text, *args, **kwargs):
        r = self.emotext.emotion_count(text)
        r.emotions = softmax_dict(r.emotions)
        return Emotion(**r.emotions)


class EmotionalPictureRecommenderGetServer(EmotionalRecommenderServer):
    def seed_name(self) -> str:
        return "img_path"

    def __init__(self, db, model_file, data_file, emopic_server):
        super().__init__(db, model_file, data_file)
        self.emopic_server = emopic_server

    def infer_emotion(self, img_path, *args, **kwargs):
        """从图片中获取情感。

        请求 emopic/emotic 的 EMOPIC_SERVER 服务，获取给定图片的情感。
        会考虑图片中所有人物 (bbox from yolo)，以图片中人物所占面积比为权重，加权平均。

        :param img_path: 图片路径
        :return: Emotion 对象
        """
        resp = requests.post(self.emopic_server, files={
            'img': (img_path, open(img_path, 'rb')),
        })
        # [{
        #   "bbox": [65, 15, 190, 186],
        #   "cat": {"Excitement": 0.16, "Peace": 0.11},
        #   "cont": [5.97, 6.06, 7.23]
        # }]
        emotic_result = resp.json()

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

# TODO: EmotionalPictureRecommenderPostServer 和 EmotionalPictureRecommenderGetServer 存在许多重复代码
class EmotionalPictureRecommenderPostServer(EmotionalRecommenderServer):
    def seed_name(self) -> str:
        return 'post-data-img'

    def __init__(self, db, model_file, data_file, emopic_server):
        super().__init__(db, model_file, data_file)
        self.emopic_server = emopic_server

    def infer_emotion(self, img, *args, **kwargs):
        """从图片中获取情感。

        请求 emopic/emotic 的 EMOPIC_SERVER 服务，获取给定图片的情感。
        会考虑图片中所有人物 (bbox from yolo)，以图片中人物所占面积比为权重，加权平均。

        :param img: img.filename 图片名 , img.file 图片文件内容
        :return: Emotion 对象
        """
        resp = requests.post(self.emopic_server, files={
            'img': (img.filename, img.file),
        })
        # [{
        #   "bbox": [65, 15, 190, 186],
        #   "cat": {"Excitement": 0.16, "Peace": 0.11},
        #   "cont": [5.97, 6.06, 7.23]
        # }]
        emotic_result = resp.json()

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

    async def handle_http(self, request):
        data = await request.post()

        img = data['img']

        if not img:
            raise web.HTTPBadRequest(text=f"post data img is required")

        print(f"POST {request.rel_url}: {img.filename}")

        k = int(request.query.get("k") or 10)

        result = self.recommend(img, k=k)
        return web.json_response(result)


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


@web.middleware
async def cors_middleware(request, handler):
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
    return web.Response()


def run_service(args):
    app = web.Application(middlewares=[cors_middleware])
    app.add_routes([web.options('/', empty_handler)])

    if args.text:
        emotext_recommender = EmotionalTextRecommenderServer(
            args.db, args.model, args.data)
        app.add_routes([
            web.get('/text', emotext_recommender.handle_http),
            web.options('/text', empty_handler)
        ])
    if args.pic:
        assert hasattr(args, 'emopic_server'), (
            '--emopic-server is required by EmotionalPictureRecommenderServer (--pic)')
        get_emopic_recommender = EmotionalPictureRecommenderGetServer(
            args.db, args.model, args.data, args.emopic_server)
        post_emopic_recommender = EmotionalPictureRecommenderPostServer(
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
