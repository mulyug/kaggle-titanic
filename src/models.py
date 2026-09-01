from sklearn.pipeline import Pipeline


def create_pipeline(model, preprocessor=None) -> Pipeline:
    """Create a pipeline combining preprocessing and a model."""

    if preprocessor is None:
        return Pipeline([
            ("classifier", model),
        ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])