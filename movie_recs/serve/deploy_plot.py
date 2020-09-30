import faiss
from ray import serve

from movie_recs.serve import finetuned_bert_plot_index, LRMovieRanker, load_pretrained_lr_model


class PlotRecommender:
    def __init__(self, lr_model):
        # Load index of finetuned bert plots.
        self.index = finetuned_bert_plot_index(faiss.IndexFlatL2(768))
        # Load pretrained logistic regression ranking model.
        self.lr_model = LRMovieRanker(lr_model, self.index.features)

    def __call__(self, request):
        # Find k nearest movies with simliar plots.
        recommended_movies = self.index.search(request)

        # Rank them using logistic regression.
        return self.lr_model.rank_movies(recommended_movies)


if __name__ == "__main__":
    # Deploy the plot model.
    client = serve.connect()
    client.create_backend("plot:v0", PlotRecommender, load_pretrained_lr_model())
    client.create_endpoint("plot", backend="plot:v0", route="/rec/plot")
