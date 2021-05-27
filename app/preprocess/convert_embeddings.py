"""
Script to convert every downloaded files from dowload_embeddings.py in the directory tree app/embeddings/* into .magnitude files
"""

import json
from pathlib import Path
from subprocess import call


embeddings_metadata_path = Path("")  # local: in app / docker: in app
embeddings_path = Path("embeddings")


def convert_embeddings_to_magnitude(input_file):
    """
    convert the embeddings in parameter in a magnitude file, delete the old embeddings
    """
    fp_in = embeddings_path / Path(input_file["type"]) / Path(input_file["name"])
    output_file = fp_in.with_suffix(".magnitude")

    # if magnitude file does not exists already and is activated
    if not output_file.exists() and input_file["is_activated"]:
        print("converting", fp_in)
        command = (
            "python3 -m pymagnitude.converter -i "
            + str(fp_in)
            + " -e 'ignore' -o "
            + str(output_file)
        )
        call(command, shell=True)
        fp_in.unlink()
    else:
        print(fp_in, "is already converted / is not activated")


# Loop on embeddings to convert them
with open(embeddings_metadata_path / Path("embeddings_metadata.json")) as json_file:
    data = json.load(json_file)

for embeddings in data:
    convert_embeddings_to_magnitude(embeddings)
