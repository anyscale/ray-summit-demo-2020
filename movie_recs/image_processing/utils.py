import json
import io
from tqdm import tqdm

import boto3
from sqlalchemy import create_engine

import ray

with open("/tmp/all-image-ids.json") as f:
    IMAGE_IDS = json.load(f)


def get_image_ids():
    return IMAGE_IDS


def progress_bar(obj_refs):
    ready = []
    with tqdm(total=len(obj_refs)) as pbar:
        while len(obj_refs) > 0:
            new_ready, obj_refs = ray.wait(obj_refs, num_returns=10)
            pbar.update(len(new_ready))
            ready.extend(new_ready)
    return ready


def load_image(image_id):
    with open(f"/tmp/movie-cover-assets/{image_id}.jpg", 'rb') as f:
        return io.BytesIO(f.read())


def load_images(num=-1):
    if num == -1:
        num = len(IMAGE_IDS)

    images = []
    for i in range(num):
        images.append(load_image(IMAGE_IDS[i]))
    return images


def _get_db_uri(db_name):
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager', region_name="us-west-2")
    get_secret_value_response = client.get_secret_value(
        SecretId="dev/demo/postgres")
    info = json.loads(get_secret_value_response['SecretString'])
    uri = f"postgresql://{info['username']}:{info['password']}@{info['host']}:{info['port']}/{db_name}"
    return uri


def get_db_connection(db_name="movies"):
    uri = _get_db_uri(db_name)
    return create_engine(uri, connect_args={"connect_timeout": 3}).connect()


def upload_color_palettes(palettes):
    pairs = list(zip(IMAGE_IDS, palettes[:len(IMAGE_IDS)]))

    conn = get_db_connection()
    transaction = conn.begin()
    for mid, palette in tqdm(pairs):
        conn.execute("UPDATE movies SET palette_json = (%s) WHERE id ~~ (%s)",
                     (json.dumps(palette), mid[0]))
    transaction.commit()
