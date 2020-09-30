import uuid

from ray import serve

from movie_recs.serve import ImpressionStore, choose_ensemble_results


class ComposedModel:
    def __init__(self):
        # Get handles to the two underlying models.
        client = serve.connect()
        self.color_handle = client.get_handle("color")
        self.plot_handle = client.get_handle("plot")

        # Store user click data in a detached actor.
        self.impressions = ImpressionStore.options(
            lifetime="detached", name="impressions").remote()

    async def __call__(self, request):
        session_key = request.args.get("session_key", str(uuid.uuid4()))

        # Call the two models and get their predictions.
        results = {
            "color": await self.color_handle.remote(request),
            "plot": await self.plot_handle.remote(request),
        }

        # Get the current model distribution.
        model_distribution = await self.impressions.model_distribution.remote(
            session_key, request.args["liked_id"])

        # Select which results to send to the user based on their clicks.
        distribution, impressions, chosen = choose_ensemble_results(
            model_distribution, results)

        # Record this click and these recommendations.
        await self.impressions.record_impressions.remote(
            session_key, impressions)

        return {
            "sessionKey": session_key,
            "dist": distribution,
            "ids": chosen,
            "sources": {
                i: source
                for source, impression in impressions.items()
                for i in impression
            }
        }


client = serve.connect()
client.create_backend("ensemble:v0", ComposedModel)
client.create_endpoint(
    "ensemble", backend="ensemble:v0", route="/rec/ensemble")
