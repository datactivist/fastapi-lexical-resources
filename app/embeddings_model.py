import json
import numpy as np
from enum import Enum
from pathlib import Path

from nltk.corpus import wordnet as wn
import nltk

from pymagnitude import *

embeddings_path = Path("embeddings")
referentiels_sources_path = Path("referentiels/sources")


def second_key_from_tuple(tuple):
    """
    Return second value of a tuple, used for sorting array of dimension [n, 2] on the second value
    """

    return tuple[1]


def sort_array_of_tuple_with_second_value(array):
    """
    Return an array of tuple sorted by second key values
    """

    array.sort(key=second_key_from_tuple, reverse=True)

    return array


def remove_second_key_from_array_of_tuple(array):

    return [array[i][0] for i in range(len(array))]


def word_is_in_vocabulary(word, referentiel):
    """
    Return whether or not a word is part of the vocabulary
    If no vocabulary avalaible, return True by default
    """

    word = word.lower()
    vocab = referentiels_sources_path / Path(f"vocabulary_{referentiel}.json")
    if not vocab.is_file():
        return True

    with open(vocab, encoding="utf-8",) as json_file:
        vocabulary = json.load(json_file)

    for vocab_word in vocabulary:
        if vocab_word == word:
            return True
    print(word, "not part of vocabulary")
    return False


class EmbeddingsType(str, Enum):
    word2vec = "word2vec"
    wordnet = "wordnet"
    fasttext = "fasttext"
    bert = "bert"
    elmo = "elmo"


class SimilarityType(str, Enum):
    synonym = "synonym"
    hyponym = "hyponym"
    hypernym = "hypernym"
    holonym = "holonym"
    similar = "similar"


class MagnitudeModel:
    """
    Class representing a Magnitude Model
    Input: Type of the embeddings and name of the embeddings
    """

    def __init__(self, embeddings_type, embeddings_name):
        self.embeddings_type = embeddings_type
        self.embeddings_name = embeddings_name
        self.model = self.load_model()

    def load_model(self):
        """
        load model using Magnitude
        """

        return Magnitude(
            embeddings_path / Path(self.embeddings_type + "/" + self.embeddings_name)
        )

    def similarity(self, keyword1, keyword2):
        """
        Compute and return similarity between two words

        Input: Two words of type string
        Output: Similarity between the two keywords
        """

        return self.model.similarity(keyword1, keyword2)

    def most_similar(
        self, keyword, topn=10, slider=0, only_vocabulary=False, referentiel=""
    ):
        """
        Return the nearest neighbors of keyword

        Input:  keyword: a word of type string
                referentiel: a referentiel pointing to a list of words
                topn: number of neighbors to get (default: 10)
                only_vocabulary: Only output words that are part of the vocabulary if available
                slider: slide the results (default: 0)  - (i.e topn=10 and slider = 2 -> [2-12]) to avoid looping when depth>1
        Output: Return the topn closest words and their similarity with keyword
        """

        similar_words = {}
        for sim_type in SimilarityType:
            similar_words[sim_type.value] = []

        similar_words[SimilarityType.similar.value] = []
        for i, word in enumerate(
            self.model.most_similar(keyword, topn=(topn + slider) * 2)[slider:]
        ):
            if i >= topn + slider:
                break
            if (
                only_vocabulary
                and word_is_in_vocabulary(word[0], referentiel)
                or not only_vocabulary
            ):
                similar_words[SimilarityType.similar.value].append(
                    {"word": word[0], "similarity": word[1],}
                )
            else:
                slider += 1
        return similar_words

    def most_similar_from_referentiel(
        self, keyword, referentiel, ref_type, topn=10, slider=0
    ):
        """
        Return the nearest neighbors of keyword taken from the referentiel

        Input:  keyword: a word of type string
                referentiel: a referentiel pointing to a list of words
                topn: number of neighbors to get (default: 10)
                slider: slide the results (default: 0)  - (i.e topn=10 and slider = 2 -> [2-12]) to avoid looping when depth>1
        Output: Return the topn closest words to keyword from the referentiel
        """

        if ref_type == "tags":
            # Load ref_words
            with open(
                referentiels_sources_path / referentiel.with_suffix(".json").name,
                encoding="utf-16",
            ) as json_file:
                ref_keywords_strings = json.load(json_file,)["names"]

            # Load their vectors
            ref_keywords_vectors = np.load(referentiel)

            # Calculate every keyword / ref_word similarities
            sim_list = []
            for keyword_str, keyword_vect in zip(
                ref_keywords_strings, ref_keywords_vectors
            ):
                sim_list.append((keyword_str, self.similarity(keyword, keyword_vect)))

            # Sort them by similarity
            sim_list = sort_array_of_tuple_with_second_value(sim_list)

            dict_list = []
            for word in sim_list[slider : topn + slider]:
                dict_list.append({"word": word[0], "similarity": word[1]})
            return dict_list

        elif ref_type == "geoloc":

            # load geoloc ref
            with open(
                referentiels_sources_path / referentiel.with_suffix(".json").name,
                encoding="utf-16",
            ) as json_file:
                ref_geoloc_json = json.load(json_file,)

            # load their vectors
            id = ref_geoloc_json["names"].index(keyword)
            parent_id = ref_geoloc_json["code_postal"].index(
                ref_geoloc_json["parent"][id]
            )
            print("lexic", ref_geoloc_json["names"][parent_id])
            return [{"word": ref_geoloc_json["names"][parent_id], "similarity": 1}]

        else:
            return []


class WordnetModel:
    """
    Class representing a WordNet model
    """

    def __init__(self):
        self.embeddings_type = EmbeddingsType.wordnet
        self.embeddings_name = "wolf-b-04.xml"

    def similarity(self, keyword1, keyword2):
        """
        Compute and return similarity between two object

        Input: Two keywords of type string, synset, or list of synset
        Output: Similiarity between the two objets using path_similarity
        """

        if type(keyword1) == nltk.corpus.reader.wordnet.Synset:
            keyword1 = [keyword1]
        elif type(str):
            keyword1 = wn.synsets(keyword1, lang="fra")

        if type(keyword2) == nltk.corpus.reader.wordnet.Synset:
            keyword2 = [keyword2]
        elif type(str):
            keyword2 = wn.synsets(keyword2, lang="fra")

        sim = 0
        for k1 in keyword1:
            for k2 in keyword2:
                new_sim = k1.path_similarity(k2)
                if new_sim != None and sim < new_sim:
                    sim = new_sim
        return sim

    def most_similar(self, keyword, topn=10, slider=0):
        """
        Return a dictionnary with the list of similar words and their similarityType

        Input: keyword, synset or list of synset
        Output: List of synonyms, hyponyms, hypernyms and holonyms in a dictionnary
        """

        if type(keyword) == nltk.corpus.reader.wordnet.Synset:
            synsets = [keyword]
        elif type(str):
            synsets = wn.synsets(keyword, lang="fra")

        similar_words = {}
        for sim_type in SimilarityType:
            similar_words[sim_type] = []

        for synset in synsets:
            if type(synset) == nltk.corpus.reader.wordnet.Synset:

                for synonym in synset.lemma_names("fra"):
                    similar_words[SimilarityType.synonym].append(
                        {"word": synonym, "similarity": -1}
                    )

                for hyponym in synset.hyponyms():
                    for word in hyponym.lemma_names("fra"):
                        similar_words[SimilarityType.hyponym].append(
                            {"word": word, "similarity": -1}
                        )

                for hypernym in synset.hypernyms():
                    for word in hypernym.lemma_names("fra"):
                        similar_words[SimilarityType.hypernym].append(
                            {"word": word, "similarity": -1}
                        )

                for holonym in synset.member_holonyms():
                    for word in holonym.lemma_names("fra"):
                        similar_words[SimilarityType.holonym].append(
                            {"word": word, "similarity": -1}
                        )

        with open("simwords.json", "w") as outfile:
            json.dump(similar_words, outfile)

        return similar_words

    def most_similar_from_referentiel(self, keyword, referentiel, topn=10, slider=0):
        return []
