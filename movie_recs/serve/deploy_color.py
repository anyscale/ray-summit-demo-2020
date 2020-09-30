from ray import serve

from movie_recs.serve import get_db_connection, create_faiss_index

class ColorRecommender:
    def __init__(self):
        # Create index of cover image colors.
        colors = get_db_connection().execute("SELECT id, palette_json FROM movies")
        self.color_index = create_faiss_index(colors)
    
    def __call__(self, request):
        # Perform KNN search for similar images.
        return self.color_index.search(request)

# Deploy the model.
client = serve.connect()
client.create_backend("color:v0", ColorRecommender)
client.create_endpoint("color", backend="color:v0", route="/rec/color")
