"""
Script to preload every .magnitude files in the directory tree app/embeddings/* by using most_similar() on them
"""

from pathlib import Path
import json

from pymagnitude import *

embeddings_metadata_path = Path("../")
embeddings_path = Path("embeddings")


def is_activated(embeddings, embeddings_metadata):
    for metadata in embeddings_metadata:
        if (
            str(Path(metadata["name"]).with_suffix(".magnitude"))
            == str(embeddings.name)
            and metadata["is_activated"]
        ):
            return True
    return False


# embeddings metadata
with open(embeddings_metadata_path / Path("embeddings_metadata.json")) as json_file:
    metadata = json.load(json_file)

# Looping on available .magnitude embeddings to preload them
for embeddings in list(embeddings_path.glob("**/*.magnitude")):
    if is_activated(embeddings, metadata):
        print("Preloading", embeddings)
        model = Magnitude(embeddings)
        model.most_similar("Chargement")
        model = None

