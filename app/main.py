import json
import numpy as np
from pathlib import Path
from enum import Enum

import embeddings_model

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional

model_list = {}
referentiel_list = {}


class Similarity_Query(BaseModel):

    keyword1: str
    keyword2: str
    embeddings_type: Optional[
        embeddings_model.EmbeddingsType
    ] = embeddings_model.EmbeddingsType.word2vec
    embeddings_name: Optional[
        str
    ] = "frWac_non_lem_no_postag_no_phrase_200_cbow_cut0.magnitude"

    class Config:
        schema_extra = {"example": {"keyword1": "chien", "keyword2": "chat",}}


class Most_Similar_Query(BaseModel):

    keyword: str
    embeddings_type: Optional[
        embeddings_model.EmbeddingsType
    ] = embeddings_model.EmbeddingsType.word2vec
    embeddings_name: Optional[
        str
    ] = "frWac_non_lem_no_postag_no_phrase_200_cbow_cut0.magnitude"
    topn: Optional[int] = 10
    slider: Optional[int] = 0

    only_vocabulary: Optional[bool] = False
    referentiel: Optional[str] = ""

    class Config:
        schema_extra = {"example": {"keyword": "barrage", "topn": 5,}}


class Get_Senses_Query(BaseModel):

    keyword: str
    embeddings_type: Optional[
        embeddings_model.EmbeddingsType
    ] = embeddings_model.EmbeddingsType.word2vec
    embeddings_name: Optional[
        str
    ] = "frWac_non_lem_no_postag_no_phrase_200_cbow_cut0.magnitude"

    class Config:
        schema_extra = {"example": {"keyword": "barrage"}}


class Tuple_Word_Sim(BaseModel):

    word: str
    similarity: float

    class Config:
        schema_extra = {
            "example": {"word": "hydroélectrique", "similarity": 0.6672828197479248,}
        }


class Most_Similar_Output(BaseModel):

    similar: List[Tuple_Word_Sim]
    synonym: List[Tuple_Word_Sim]
    hyponym: List[Tuple_Word_Sim]
    hypernym: List[Tuple_Word_Sim]

    class Config:
        schema_extra = {
            "example": {
                "similar": [
                    {"word": "barrages", "similarity": 0.7585024833679199},
                    {"word": "hydroélectrique", "similarity": 0.6672828197479248},
                    {"word": "digue", "similarity": 0.6171526312828064},
                    {"word": "génissiat", "similarity": 0.6053286790847778},
                    {"word": "rivière", "similarity": 0.6018728017807007},
                ],
                "synonym": [],
                "hyponym": [],
                "hypernym": [],
            }
        }


class Most_Similar_From_Referentiel_Query(BaseModel):

    keyword: str
    referentiel: str
    ref_type: str
    embeddings_type: Optional[
        embeddings_model.EmbeddingsType
    ] = embeddings_model.EmbeddingsType.word2vec
    embeddings_name: Optional[
        str
    ] = "frWac_non_lem_no_postag_no_phrase_200_cbow_cut0.magnitude"
    topn: Optional[int] = 10
    only_vocabulary: Optional[bool] = False
    slider: Optional[int] = 0

    class Config:
        schema_extra = {
            "example": {"keyword": "barrage", "referentiel": "datasud", "topn": 5,}
        }


embeddings_path = Path("embeddings")
# Looping on available .magnitude embeddings to save them in dictionnary
print("Starting saving embeddings in dictionnary")
for embeddings in list(embeddings_path.glob("**/*.magnitude")):
    model_list[embeddings.name] = embeddings_model.MagnitudeModel(
        embeddings.parent.name, embeddings.name
    )
print("Saving done")

referentiel_path = Path("referentiels/vectors")
# Looping on available .npy referentiel to save them in dictionnary
print("Starting saving referentiels in dictionnary")
for referentiel in list(referentiel_path.glob("**/*.npy")):
    key = str(referentiel.parent.name) + "/" + str(referentiel.with_suffix("").name)
    referentiel_list[key] = referentiel
print("Saving done")

# Launch API
app = FastAPI()


@app.get("/help_embeddings/{embeddings_type}")
async def get_embeddings_names(embeddings_type: embeddings_model.EmbeddingsType):
    """
    ## Function
    Return every variant embeddings available for the embeddings_type given in parameter
    ## Parameter
    ### Required
    - **embeddings_type**: Type of the embeddings
    ## Output
    List of embeddings variant available
    """
    if embeddings_type != embeddings_model.EmbeddingsType.wordnet:
        path = Path("./embeddings") / Path(embeddings_type.value)
        results = []
        for filename in list(path.glob("**/*.magnitude")):
            results.append(filename.name)
        return results
    else:
        return [
            "This API uses the library NLTK, which uses the Open Multilingual Wordnet, which uses the wolf-1.0b4 embeddings, therefore you don't have to specify an embeddings name."
        ]


@app.get("/help_referentiels")
async def get_referentiels_names():
    """
    ## Function
    Return every referentiel available for the most_similar_frm_referentiel requests
    """

    result = []
    for keys in referentiel_list.keys():
        ref_name = str(Path(keys).with_suffix("").name)
        if ref_name not in result:
            result.append(ref_name)

    return ref_name


@app.post("/similarity", response_model=float)
async def get_similarity(similarity_query: Similarity_Query):
    """
    ## Function
    Return similarity between two keywords
    ## Parameter
    ### Required
    - **keyword1**: a word of type string
    - **keyword2**: a word of type string
    ### Optional
    - **embeddings_type**: Type of the embeddings | default value: word2vec
    - **embeddings_name**: Variant of the embeddings | default value: frWac_non_lem_no_postag_no_phrase_200_cbow_cut0.magnitude
    ## Output
    An int between 0 and 1
    """

    if similarity_query.embeddings_type == embeddings_model.EmbeddingsType.bert:
        raise HTTPException(
            status_code=501, detail="Bert embeddings not implemented yet"
        )

    if similarity_query.embeddings_type != embeddings_model.EmbeddingsType.wordnet:
        try:
            model = model_list[similarity_query.embeddings_name]
        except:
            raise HTTPException(
                status_code=404,
                detail="This embeddings_name doesn't exist for this embeddings_type. You can request the list of embeddings available for a particular embeddings type by requesting get(http/[...]/help_embeddings/{embeddings_type}",
            )
    else:
        model = embeddings_model.WordnetModel()
        # TODO mettre dans le dict?

    return model.similarity(similarity_query.keyword1, similarity_query.keyword2)


@app.post("/most_similar", response_model=Most_Similar_Output)
async def get_most_similar(most_similar_query: Most_Similar_Query):
    """
    ## Function
    Return the topn most similar words from the keyword in parameter
    ## Parameter
    ### Required
    - **keyword**: a word of type string
    ### Optional
    - **embeddings_type**: Type of the embeddings | default value: word2vec
    - **embeddings_name**: Variant of the embeddings | default value: frWac_non_lem_no_postag_no_phrase_200_cbow_cut0.magnitude
    - **topn**: number of neighbors to get | default value: 10
    - **only_vocabulary**: only take words part of the vocabulary if available
    - **slider**: slide the results | default value: 0  - (i.e topn=10 and slider = 1 -> [1-11]) to avoid looping when depth>1

    ## Output
    A list of tuple of type (word, similary_type) with the topn most similar words
    """

    if most_similar_query.embeddings_type == embeddings_model.EmbeddingsType.bert:
        raise HTTPException(
            status_code=501, detail="Bert embeddings not implemented yet"
        )

    if most_similar_query.embeddings_type != embeddings_model.EmbeddingsType.wordnet:
        try:
            model = model_list[most_similar_query.embeddings_name]
        except:
            raise HTTPException(
                status_code=404,
                detail="This embeddings_name doesn't exist for this embeddings_type. You can request the list of embeddings available for a particular embeddings type by requesting get(http/[...]/help_embeddings/{embeddings_type}",
            )
    else:
        model = embeddings_model.WordnetModel()
        # TODO mettre dans le dict?

    return model.most_similar(
        most_similar_query.keyword,
        most_similar_query.topn,
        most_similar_query.slider,
        most_similar_query.only_vocabulary,
        most_similar_query.referentiel,
    )


@app.post("/most_similar_referentiel", response_model=List[Tuple_Word_Sim])
async def get_most_similar_from_referenciel(
    most_similar_from_ref_query: Most_Similar_From_Referentiel_Query,
):
    """
    ## Function
    Return the topn most similar words to keyword from the selected referentiel
    ## Parameter
    ### Required
    - **keyword**: a word of type string
    - **referentiel**: a referentiel pointing to a list of words - see /help_referentiels requests to see what is available | default value: Datasud
    ### Optional
    - **embeddings_type**: Type of the embeddings | default value: word2vec
    - **embeddings_name**: Variant of the embeddings | default value: frWac_non_lem_no_postag_no_phrase_200_cbow_cut0.magnitude
    - **topn**: number of neighbors to get | default value: 10
    - **only_vocabulary**: only take words part of the vocabulary if available
    - **slider**: slide the results | default value: 0  - (i.e topn=10 and slider = 1 -> [1-11]) to avoid looping when depth>1

    ## Output
    A list of tuple of type (word, similary_type) with the topn most similar words from the referentiel
    """

    if (
        most_similar_from_ref_query.embeddings_type
        == embeddings_model.EmbeddingsType.bert
    ):
        raise HTTPException(
            status_code=501, detail="Bert embeddings not implemented yet"
        )

    if (
        most_similar_from_ref_query.embeddings_type
        != embeddings_model.EmbeddingsType.wordnet
    ):
        try:
            model = model_list[most_similar_from_ref_query.embeddings_name]
        except:
            raise HTTPException(
                status_code=404,
                detail="This embeddings_name doesn't exist for this embeddings_type. You can request the list of embeddings available for a particular embeddings type by requesting get(http/[...]/help_embeddings/{embeddings_type}",
            )
    else:
        model = embeddings_model.WordnetModel()
        # TODO mettre dans le dict?

    most_similar_ref = model.most_similar_from_referentiel(
        most_similar_from_ref_query.keyword,
        referentiel,
        most_similar_from_ref_query.ref_type,
        most_similar_from_ref_query.topn,
        most_similar_from_ref_query.only_vocabulary,
        most_similar_from_ref_query.slider,
    )
    return most_similar_ref
