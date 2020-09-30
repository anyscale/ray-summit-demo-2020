import asyncio
import base64
from databases import Database
import os
import json
import sys
from typing import cast

import boto3

from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

import ray
from ray import serve


class B64Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (bytes)):
            return base64.b64encode(o).decode("utf-8")
        if isinstance(o, (memoryview)):
            return o.tobytes()
        return o


def _get_db_uri(db_name):
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager', region_name="us-west-2")
    get_secret_value_response = client.get_secret_value(
        SecretId="dev/demo/postgres")
    info = json.loads(get_secret_value_response['SecretString'])
    uri = f"postgresql://{info['username']}:{info['password']}@{info['host']}:{info['port']}/{db_name}"
    return uri


async def get_db_async_db(db_name="movies"):
    uri = _get_db_uri(db_name)
    database = Database(uri)
    await database.connect()
    return database


class MovieInfoService:
    """Return detailed information given a movie id

    Handle /info?mid=<int:id>
    Return {"title": ..., "rating": ..., "img_jpg_bytes": ...}
    """

    def __init__(self):
        self.async_conn = None
        self.get_conn_lock = asyncio.Lock()

    async def get_async_conn(self):
        async with self.get_conn_lock:
            if self.async_conn is None:
                self.async_conn = await get_db_async_db()
            return self.async_conn

    async def get_movie_from_id(self, movie_id: str):
        db_conn = await self.get_async_conn()
        cursor = await db_conn.fetch_one(
            """
            SELECT * FROM movies
            WHERE movies.id ~~ (:id)
            LIMIT 1
            """, {"id": movie_id})
        return dict(cursor)

    async def __call__(self, flask_request):
        movie_info = await self.get_movie_from_id(flask_request.args["mid"])
        return json.dumps(movie_info, cls=B64Encoder)


class RandomRecommender:
    """Return random set of movie ids

    Handle /rec/random?count=<int: count>
    Return [123, 1234, 1235]
    """

    def __init__(self):
        self.async_conn = None
        self.get_conn_lock = asyncio.Lock()

    async def get_async_conn(self):
        async with self.get_conn_lock:
            if self.async_conn is None:
                self.async_conn = await get_db_async_db()
            return self.async_conn

    async def __call__(self, request):
        num_returns = int(request.args.get("count", 20))

        db_conn = await self.get_async_conn()

        records = await db_conn.fetch_all(
            """
            SELECT movies.id
            FROM movies
            ORDER BY RANDOM()
            LIMIT (:num)
            """, {"num": num_returns})
        return [record['id'] for record in records]


client = serve.start(
    detached=True,
    http_host="0.0.0.0",
    http_middlewares=[Middleware(CORSMiddleware, allow_origins=["*"])])

client.create_backend("info:v0", MovieInfoService)
client.create_endpoint("info", backend="info:v0", route="/info")
client.create_backend("random:v0", RandomRecommender)
client.create_endpoint("random", backend="random:v0", route="/rec/random")
