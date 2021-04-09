"""
Script to convert every downloaded files from dowload_embeddings.py in the directory tree app/embeddings/* into .magnitude files
"""

import os
import json
from pathlib import Path

embeddings_metadata_path = Path("../")
embeddings_path = Path("embeddings")


def convert_embeddings_to_magnitude(input_file):
    """
    convert the embeddings in parameter in a magnitude file, delete the old embeddings
    """

    input_file = embeddings_path / Path(input_file["type"]) / Path(input_file["name"])
    output_file = input_file.with_suffix(".magnitude")

    # if magnitude file does not exists already and is activated
    if not os.path.exists(output_file) and input_file["is_activated"]:
        print("converting", input_file)
        command = (
            "python3 -m pymagnitude.converter -i "
            + str(input_file)
            + " -e 'ignore' -o "
            + str(output_file)
        )
        os.system(command)
        os.remove(input_file)
    else:
        print(input_file, "is already converted / is not activated")


# Loop on embeddings to convert them
with open(embeddings_metadata_path / Path("embeddings_metadata.json")) as json_file:
    data = json.load(json_file)

for embeddings in data:
    convert_embeddings_to_magnitude(embeddings)
