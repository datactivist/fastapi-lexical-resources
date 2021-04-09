"""
Script to convert every downloaded files from dowload_embeddings.py in the directory tree app/embeddings/* into .magnitude files
"""

import json
from pathlib import Path
import sys

embeddings_metadata_path = Path("")
embeddings_path = Path("app/embeddings")

with open(embeddings_metadata_path / Path("embeddings_metadata.json")) as json_file:
    data = json.load(json_file)

if len(sys.argv) > 1:
    activation = sys.argv[1].split("|")
else:
    print("Usage: python3 updateEmbeddingsConfig.py <config-string>")
    exit(1)

flag_no_embed = True
dockerignorestr = ""
for i, embeddings in enumerate(data):
    if i < len(activation):
        if activation[i] == "activate":
            flag_no_embed = False
            embeddings["is_activated"] = True
        else:
            embeddings["is_activated"] = False
            embed_path = embeddings_path / Path(
                embeddings["type"] + "/" + embeddings["name"]
            )
            dockerignorestr += str(embed_path) + "\n"
            dockerignorestr += str(embed_path.with_suffix(".magnitude")) + "\n"

    else:
        embeddings["is_activated"] = False
        embed_path = embeddings_path / Path(
            embeddings["type"] + "/" + embeddings["name"]
        )
        dockerignorestr += str(embed_path) + "\n"
        dockerignorestr += str(embed_path.with_suffix(".magnitude")) + "\n"


if flag_no_embed:
    print(
        "Warning: No embeddings selected, please add at least one embedding in the app/api-config.config file"
    )

with open(
    embeddings_metadata_path / Path("embeddings_metadata.json"), "w"
) as json_file:
    json.dump(data, json_file)

with open(".dockerignore", "w") as dockerignorefile:
    dockerignorefile.write(dockerignorestr)
