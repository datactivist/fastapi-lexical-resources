"""
Script to download every missing embeddings found in the embeddings_metadata.json file
"""

import requests
import gzip
import shutil
import json
from pathlib import Path

root_path = Path('app')
embeddings_metadata_path = root_path / Path("")
embeddings_path = root_path / Path("embeddings")


def decompress_archive(archivename, filename):
    """
    decompress an archive of .gz, save it under filename name, and delete archivename
    """

    with gzip.open(archivename, "rb") as f_in:
        with open(filename, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    archivename.unlink()


def download_embeddings(embeddings):
    """
    download all the missing embeddings from the embeddings_metadata.json file
    """

    dir_path = embeddings_path / Path(embeddings["type"])
    embeddings_temp = dir_path / Path(embeddings["name"])
    embeddings_dl = dir_path / embeddings["dl_url"].rsplit("/", 1)[1]

    # if embeddings/embeddings_type directory doesn't exist, create it
    dir_path.mkdir(parents=False, exist_ok=True)

    embeddings_temp_path = embeddings_dl.with_suffix("")
    embeddings_extension = embeddings_dl.suffix

    # if the embeddings are not already downloaded and are activated, download them
    if (
        embeddings["is_activated"]
        and not embeddings_temp.with_suffix(".magnitude").exists()
        and not embeddings_temp.exists()
        and not embeddings_dl.exists()
    ):

        print(embeddings_temp_path, "does not exist, downloading it...")

        embed = requests.get(embeddings["dl_url"], allow_redirects=True)
        open(embeddings_dl, "wb").write(embed.content)

        # if the embeddings are compressed, extract it (for fasttext)
        if embeddings_extension == ".gz":
            decompress_archive(embeddings_dl, embeddings_temp)

    else:
        print(embeddings_temp_path, "already exist or is not activated")


# Open embeddings metadata to get urls
with open(embeddings_metadata_path / Path("embeddings_metadata.json")) as json_file:
    data = json.load(json_file)

# If embeddings directory doesn't exist, create it
embeddings_path.mkdir(parents=False, exist_ok=True)


# Loop on embeddings to download them
for embeddings in data:
    download_embeddings(embeddings)
