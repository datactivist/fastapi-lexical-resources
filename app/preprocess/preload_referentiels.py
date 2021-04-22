"""
Script to preload and save datasud keywords as vector for each embeddings of type .magnitude in the directory tree app/embeddings/*
"""

import json
import numpy as np
from pathlib import Path

from pymagnitude import *

embeddings_metadata_path = Path("")
referentiel_path = Path("referentiels")
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


def preload_datasud_vectors(
    model, referentiel_name, referentiel_keywords, embeddings_type, embeddings_name
):
    """
    Preload datasud vectors by saving them separately in "referentiels/vectors/embeddings_name/referentiel_name.npy"
    Input:  model: magnitude model
            referentiel_name: name of the referentiel
            referentiel_keywords: list of strings
            embeddings_type: type of the embeddings
            embeddings_name: name of the embeddings
    """

    referentiel_vectors_path = referentiel_path / "vectors"
    referentiel_vectors_path.mkdir(parents=False, exist_ok=True)

    type_dir_path = referentiel_vectors_path / embeddings_type
    type_dir_path.mkdir(parents=False, exist_ok=True)

    name_dir_path = type_dir_path / Path(embeddings_name).with_suffix("")
    name_dir_path.mkdir(parents=False, exist_ok=True)

    vectors_name = name_dir_path / Path(referentiel_name).with_suffix(".npy")

    if not vectors_name.is_file():

        print("Saving", vectors_name)
        vectors = model.query(referentiel_keywords)
        np.save(vectors_name, vectors)

    else:
        print(vectors_name, "already exist")

    model = None


# embeddings metadata
with open(embeddings_metadata_path / Path("embeddings_metadata.json")) as json_file:
    metadata = json.load(json_file)

# Looping on available .json referentiel to save them as vectors
referentiel_sources_path = referentiel_path / "sources"
for referentiel in list(referentiel_sources_path.glob("**/*.json")):
    # Load referentiels as a list of string
    with open(referentiel, encoding="utf-16") as json_file:
        referentiel_vectors = json.load(json_file,)["result"]

    # Looping on available .magnitude embeddings to preload the referentiel on them
    for embeddings in list(embeddings_path.glob("**/*.magnitude")):
        if is_activated(embeddings, metadata):
            preload_datasud_vectors(
                Magnitude(embeddings),
                referentiel.name,
                referentiel_vectors,
                embeddings.parent.name,
                embeddings.name,
            )
