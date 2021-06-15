"""
Script to download every missing referentiels found in the referentiel_metadata.json file
"""

from pathlib import Path
import json

root_path = Path('app')
referentiel_metadata_path = root_path / Path("")
referentiel_path = root_path / Path("referentiels")
referentiel_sources_path = referentiel_path / Path("sources")


def download_referentiels(referentiel):
    """
    download the referentiel given in argument
    """

    referentiel_output = referentiel_sources_path / Path(
        referentiel["name"]
    ).with_suffix(".json")

    # if the referentiels are not already downloaded and are activated, download them
    if referentiel["is_activated"] and not referentiel_output.exists():

        print(referentiel_output, "does not exist, downloading it...")

        """ TODO
        referentiel_dl = requests.get(referentiel["dl_url"], allow_redirects=True)
        """

        raise (
            "is missing, you need to download referentiels manually and put them in the following directory: app/referentiels/sources",
        )

    else:
        print(referentiel_output, "already exist or is not activated")


# Open embeddings metadata to get urls
with open(referentiel_metadata_path / Path("referentiels_metadata.json")) as json_file:
    data = json.load(json_file)

# If referentiels directories doesn't exist, create them
referentiel_path.mkdir(parents=False, exist_ok=True)
referentiel_sources_path.mkdir(parents=False, exist_ok=True)

# Loop on embeddings to download them
for referentiel in data:
    download_referentiels(referentiel)
