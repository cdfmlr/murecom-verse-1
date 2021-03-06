import argparse
import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List, Optional, Dict, Iterable

import joblib
import numpy as np
import requests
from aiohttp import web
from aiohttp.web_exceptions import HTTPException
from sklearn.neighbors import NearestNeighbors
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.ext.automap import automap_base

import nest_asyncio

import emotext
from emopic.emotic2emotext import emotic2dlut

# region Emotion

Emotion = namedtuple('Emotion', emotext.emotions)


def keys(self):
    return self._fields


def values(self):
    return tuple(self)


Emotion.keys = keys
Emotion.values = values


def emotion_vector(emotions: List) -> Emotion:
    elems = dict.fromkeys(emotext.emotions, 0)
    elems.update({x.emotion: x.intensity for x in emotions})
    ev = Emotion(**elems)

    return ev


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

def RecommendedTrack(track_id: str, track_name: str, artists: List[str], album_cover: str = ''):
    return {
        "track_id": track_id,
        "track_name": track_name,
        "artists": artists,
        "album_cover": album_cover,
    }


def EmotionalRecommendResult(seed_emotion: Emotion, distances: List[float], recommended_tracks: List[RecommendedTrack]):
    return {
        "seed_emotion": seed_emotion,
        "distances": distances,
        "recommended_tracks": recommended_tracks,
    }


class Persistence(object):
    """Persistence with joblib
    """

    @staticmethod
    def save(obj, filename: str, compress=True):
        """Save object into filename
        :param obj: object to save
        :param filename: pah to saving file
        :param compress: Optional compression level for the data. 0 or False is
            no compression. Higher value means more compression, but also slower
            read and write times. Using a value of 3 is often a good compromise.
            If compress is True, the compression level used is 3.
        :return: The list of file names in which the data is stored.
        """
        names = joblib.dump(obj, filename, compress=compress)
        print("saved:", names)
        return names

    @staticmethod
    def load(filename: str):
        """load a saved object
        :param filename: the saved file
        :return: object
        """
        if filename.endswith('json'):  # legacy
            with open(filename) as df:
                return json.load(df)
        return joblib.load(filename)


# endregion data

class EmotionalRecommendBase:
    """EmotionalRecommendBase ??? EmotionalRecommender Trainer/Server ?????????????

    ???????????????????????????????????????????????????
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

class EmotionalRecommendTrainer(EmotionalRecommendBase, ABC):
    def __init__(self, db, datasize: int):
        super().__init__(db)

        self.data = {'ids': [], 'emo': []}
        self.model = None
        self.datasize = datasize

    @abstractmethod
    def query(self, *args, **kwargs) -> Iterable:
        """????????????????????????????????????????????????

        :return: Iterable[self.Track]
        """
        raise NotImplemented

    def make_data(self):
        for t in self.query():
            assert isinstance(t, self.Track)
            if not t.track_emotions_collection:
                continue
            self.data['ids'].append(t.id)
            self.data['emo'].append(emotion_vector(t.track_emotions_collection))

    def save_data(self, filename):
        Persistence.save(self.data, filename)

    def train(self, algorithm='ball_tree'):
        X = np.array(self.data['emo'])
        print(f'training on data {X.shape=}')
        nbrs = NearestNeighbors(n_neighbors=10, algorithm=algorithm).fit(X)
        self.model = nbrs

    def save_model(self, filename):
        Persistence.save(self.model, filename)


class GoodSongsTrainer(EmotionalRecommendTrainer):
    MIN_GOOD_SONG_PUBLISH_TIME = 0
    MAX_GOOD_SONG_ID = 10000000

    def __init__(self, db, datasize):
        super().__init__(db, datasize)

    def query(self, *args, **kwargs) -> Iterable:
        """
        select
            t.id tid, t.name, t.pop, t.publish_time,
            array(select a.name from artists a
                left join track_artists ta
                on ta.artist_id=a.id
                where ta.track_id=t.id) artist
        from tracks t
        where t.publish_time>0 and t.id<10000000
        order by t.pop desc, t.publish_time
        limit {datasize};
        """
        return self.db_session. \
                   query(self.Track). \
                   filter(self.Track.publish_time > self.MIN_GOOD_SONG_PUBLISH_TIME). \
                   filter(self.Track.id < self.MAX_GOOD_SONG_ID). \
                   order_by(self.Track.pop.desc(), self.Track.publish_time)[:self.datasize]


class JoinDatasTrainer(EmotionalRecommendTrainer):
    """??????????????????????????????????????????????????????

    ???????????????????????? CLI ????????????????????????????????????
    """

    def __init__(self, db, datasize):
        super().__init__(db, datasize)
        self.datasets = input(
            'Please enter datasets\' filenames (e.g. *.data.joblib) to join (sep by ",")\n> '
        ).split(',')
        self.datasets = list(map(lambda f: f.strip(), self.datasets))
        if len(self.datasets) < 1:
            raise ValueError('No datasets.')

    def query(self, *args, **kwargs) -> Iterable:
        pass

    def make_data(self):
        for filename in self.datasets:
            d = Persistence.load(filename)
            for tid, emo in zip(d['ids'], d['emo']):
                if tid in self.data['ids']:
                    continue
                self.data['ids'].append(tid)
                self.data['emo'].append(emo)


available_trainers = {c.__name__: c
                      for c in EmotionalRecommendTrainer.__subclasses__()}


# endregion trainer

# region server

class EmotionalRecommendServer(EmotionalRecommendBase, ABC):
    """Abstract Emotional Recommender Server

    ?????????????????? recommend ????????? http handler: `ANY /route/this[?k=10]`.

    - ?????? parse_seed ????????????????????????????????????????????????
         ?????? seed ????????????????????? infer_emotion ???????????????????????????
    - ?????? infer_emotion ?????????????????????????????????
    """

    def __init__(self, db, model_file, data_file):
        super().__init__(db)

        # with open(data_file) as df:
        #     self.data = json.load(df)
        # self.model = joblib.load(model_file)
        self.model = Persistence.load(model_file)
        self.data = Persistence.load(data_file)

    @abstractmethod
    def parse_seed(self, request: web.Request) -> any:
        """parse_seed ??????????????? request ??????????????????????????????(??????) seed.

        ?????????????????????????????? _parse_seed_from_query ????????????????????? ?query_name=seed.

        :return: seed: ???????????????????????????
        :raise: ValueError: ???????????????????????? ValueError?????????????????????????????????
        """
        raise NotImplemented

    def _parse_seed_from_query(self, request: web.Request, query_name: str) -> Optional[str]:
        return request.query.get(query_name)

    def seed_to_str(self, seed):
        """???????????????????????????????????????????????? seed"""
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

            artists = list(map(lambda a: a.name, t.artists_collection))
            albums = t.albums_collection
            cover = albums[0].pic_url if albums else ''

            rt = RecommendedTrack(
                track_id=f'ncm-{t.id}',
                track_name=t.name,
                artists=artists,
                album_cover=cover,
            )
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
            print(f"    400 Bad Request: {e}")
            raise web.HTTPBadRequest(text=str(e))


class EmoTextRecommendServer(EmotionalRecommendServer):
    """??? query ????????? text ???????????? seed???
    seed: str ????????????????????????????????? emotext ???????????????
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
    """?????????????????????????????????????????????

    ?????? emotic ??????????????????????????? GET/POST ???????????????????????????????????????
    ???????????????????????????????????????????????? infer_emotion ?????????????????? emotic ??????????????????
    ?????????????????????????????? parse_seed ????????????????????????????????????
    ???????????? request_emotic ???????????? emotic ????????????
    """

    def __init__(self, db, model_file, data_file, emopic_server):
        super().__init__(db, model_file, data_file)
        self.emopic_server = emopic_server

    @abstractmethod
    def request_emotic(self, seed) -> List[Dict]:
        """?????? emotic ???????????????????????? JSON??????????????????????????????

        ```
         [{
           "bbox": [65, 15, 190, 186],
           "cat": {"Excitement": 0.16, "Peace": 0.11},
           "cont": [5.97, 6.06, 7.23]
         }]
         ```

        :param seed: parse_seed ????????? seed
        :return: ??????????????????
        """
        raise NotImplemented

    def infer_emotion(self, seed, *args, **kwargs) -> Emotion:
        emotic_result = self.request_emotic(seed)

        # ????????????
        total_emotion = dict.fromkeys(emotext.emotions, 0.0)

        # ???????????????????????????
        area = lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        areas = [area(p['bbox']) for p in emotic_result]
        sum_area = sum(areas)

        # ?????? emotext ?????????????????????????????????
        for person, person_area in zip(emotic_result, areas):
            person_emo = emotic2dlut(person['cat'], person['cont'])
            weight = person_area / sum_area

            for emo, value in person_emo.items():
                total_emotion[emo] += value * weight

        sum_v = sum(v for v in total_emotion.values())
        if sum_v < 1e-8:
            raise ValueError('no emotion detected in the image')
        for k in total_emotion:
            total_emotion[k] /= sum_v

        return Emotion(**total_emotion)


class GetEmoPicRecommendServer(EmoPicRecommendServer):
    """??????????????????????????????????????????????????????????????????????????????
    ??????????????????????????????????????????????????????????????????????????????
    ??????????????????????????????????????????

    seed ??? query ??????????????? img_path: ???????????????????????????
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


CORS_HEADERS = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': '*',
    'Access-Control-Allow-Headers': '*',
    'Access-Control-Allow-Credentials': 'true',
}


@web.middleware
async def cors_middleware(request, handler):
    """???????????? cors

    `app = web.Application(middlewares=[cors_middleware])`
    """
    # if request.method == 'OPTIONS':
    #     response = web.Response(text="")
    # else:
    try:
        response = await handler(request)

        for k, v in CORS_HEADERS.items():
            response.headers[k] = v

        return response
    except HTTPException as e:
        for k, v in CORS_HEADERS.items():
            e.headers[k] = v
        raise e


async def empty_handler(request):
    """????????? route ????????????????????? options empty_handler ???????????? cors ??????:
    `web.options('...', empty_handler)`
    """
    return web.Response()


# endregion cors

# region cli

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("murecom-intro", description="A emotional music recommender")
    subparsers = parser.add_subparsers(help="sub commands")

    # service
    parser_service = subparsers.add_parser("service", help="start a inferring server on a trained model")
    parser_service.set_defaults(subcommand="service")

    parser_service.add_argument("--text", action='store_true', help="start emotext recommending service")
    parser_service.add_argument("--pic", action='store_true', help="start emopic recommending service")

    parser_service.add_argument("--db", type=str, help="path to PostgreSQL database", required=True)
    parser_service.add_argument("--model", type=str, help="path to a trained model", required=True)
    parser_service.add_argument("--data", type=str, help="path to saved train-set data", required=True)
    parser_service.add_argument("--emopic-server", type=str, help="url to emopic (EMOTIC) server. required by --pic")

    parser_service.add_argument("--host", type=str, help="server host", default="localhost")
    parser_service.add_argument("--port", type=int, help="listen port", default=8080)

    # train
    parser_train = subparsers.add_parser("train", help="train a new model")
    parser_train.set_defaults(subcommand="train")

    parser_train.add_argument("--db", type=str, help="path to PostgreSQL database", required=True)
    parser_train.add_argument("--datasize", type=int, help="training data count", required=True)

    parser_train.add_argument("--save-model", type=str, help="path to save trained model", required=True)
    parser_train.add_argument("--save-data", type=str, help="path to save train-set data", required=True)

    parser_train.add_argument("--trainer", type=str,
                              help=f"which trainer to use: {available_trainers.keys()}, default: GoodSongsTrainer",
                              default="GoodSongsTrainer")
    parser_train.add_argument("--algorithm", type=str, help="nearest neighbors algorithm, default: ball_tree",
                              default="ball_tree")

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
    assert args.trainer in available_trainers, f"no such trainer: {args.trainer}"
    trainer_class = available_trainers[args.trainer]
    assert issubclass(trainer_class, EmotionalRecommendTrainer)
    trainer = trainer_class(args.db, args.datasize)

    trainer.make_data()
    trainer.train(algorithm=args.algorithm)

    trainer.save_data(args.save_data)
    trainer.save_model(args.save_model)


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
