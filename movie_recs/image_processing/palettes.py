import json
import time

from colorthief import ColorThief

from movie_recs.image_processing.utils import get_image_ids, load_image, progress_bar

# If you want to run the code, you can define your
# own get_image_ids and load_image functions.
# def get_image_ids() -> List[str]
# def load_image(str) -> io.BytesIO

ids = get_image_ids()

import ray
ray.init()


@ray.remote
def get_palette(image_id):
    return ColorThief(load_image(image_id)).get_palette(color_count=6)


start = time.time()
palettes = []
for img_id in ids:
    palettes.append(get_palette.remote(img_id))

palettes = ray.get(progress_bar(palettes))
print("Finished {} images in {}s.".format(len(ids), time.time() - start))

with open("outputs/palettes.json", "w") as f:
    json.dump(palettes, f)
