import expansion
from pathlib import Path
import json
import numpy as np


def test_magnitudeModel():

    model = None
    for embeddings in list(Path("embeddings").glob("**/*.magnitude")):
        model = expansion.MagnitudeModel(embeddings.parent.name, embeddings.name)
        break

    if model != None:

        sim = model.similarity("chien", "chat")
        assert sim <= 1 and sim >= 0

        most_sim = model.most_similar("barrage", 10)
        for sim_type in expansion.SimilarityType:
            if sim_type == expansion.SimilarityType.similar:
                assert len(most_sim[sim_type]) == 10
            else:
                assert len(most_sim[sim_type]) == 0

        dtsud_strings, dtsud_keywords = model.load_datasud_keywords()
        assert len(dtsud_keywords) > 0
        assert len(dtsud_strings) > 0


def test_get_senses_from_keyword():

    keyword = "barrage"

    # Wordnet Test
    wordnet_model = expansion.WordnetModel()
    senses = expansion.get_senses_from_keyword(wordnet_model, keyword)
    assert len(senses) > 2

    # Magnitude embeddings Test
    for embeddings in list(Path("embeddings").glob("**/*.magnitude")):
        basic_model = expansion.MagnitudeModel(embeddings.parent.name, embeddings.name)
        break

    if basic_model != None:
        senses = expansion.get_senses_from_keyword(basic_model, keyword)
        assert len(senses) == 1
        assert senses[0] == keyword


def test_wordnetModel():

    model = expansion.WordnetModel()

    sim = model.similarity("chien", "chat")
    assert sim <= 1 and sim >= 0

    senses = expansion.get_senses_from_keyword(model, "chien")
    most_sim = model.most_similar(senses[0])

    for sim_type in expansion.SimilarityType:
        if sim_type != expansion.SimilarityType.similar:
            assert len(most_sim[sim_type]) > 1


def test_split_keywords():

    keywords = "barrage électrique"

    assert expansion.split_user_entry(keywords) == ["barrage", "électrique"]


def test_second_key_from_tuple():

    mytuple = ["hydrauélectrique", 0.8974]

    assert expansion.second_key_from_tuple(mytuple) == mytuple[1]


def test_sort_array_of_tuple_with_second_value():

    myarray = [["chat", 0.2345], ["hydrauélectrique", 0.8974], ["chien", 0.340]]

    expansion.sort_array_of_tuple_with_second_value(myarray)

    for i in range(len(myarray) - 1):
        assert myarray[i][1] > myarray[i + 1][1]


def test_get_datasud_similarities():

    min_threshold = 0.2

    model = None
    for embeddings in list(Path("embeddings").glob("**/*.magnitude")):
        model = expansion.MagnitudeModel(embeddings.parent.name, embeddings.name)
        break

    if model != None:
        sim_list = expansion.get_datasud_similarities(model, "barrage", min_threshold)
        assert len(sim_list) > 0
        for sim in sim_list:
            assert sim[1] > min_threshold


def test_get_datasud_keywords():

    max_k = 5

    model = None
    for embeddings in list(Path("embeddings").glob("**/*.magnitude")):
        model = expansion.MagnitudeModel(embeddings.parent.name, embeddings.name)
        break

    if model != None:

        keywords_list = expansion.get_datasud_keywords(model, "barrage", max_k)

        assert len(keywords_list) == max_k


def test_build_tree():

    keyword = "barrage"
    max_width = 3
    max_depth = 1
    max_dtsud_keywords = 5

    # Wordnet Test
    wordnet_model = expansion.WordnetModel()
    tree = expansion.build_tree(
        wordnet_model, keyword, max_width, max_depth, max_dtsud_keywords
    )
    assert tree["original_keyword"] == keyword
    assert len(tree["tree"]) > 1  # wordnet so more than 1 sense per keyword

    # Test Cluster en amont pour wordnet
    assert len(tree["tree"][0]["similar_senses"]) > 1  # Ignored max_width
    assert len(tree["tree"][0]["similar_senses"][0]) == 2  # tuple (sense, synonymtype)

    # Magnitude embeddings Test
    for embeddings in list(Path("embeddings").glob("**/*.magnitude")):
        basic_model = expansion.MagnitudeModel(embeddings.parent.name, embeddings.name)
        break

    if basic_model != None:

        tree = expansion.build_tree(
            basic_model, keyword, max_width, max_depth, max_dtsud_keywords
        )

        assert tree["original_keyword"] == keyword
        assert len(tree["tree"]) == 1  # not wordnet, so only 1 sense


def test_get_cluster():
    keyword = "barrage"
    max_width = 3
    max_depth = 1

    # Magnitude embeddings Test
    for embeddings in list(Path("embeddings").glob("**/*.magnitude")):
        basic_model = expansion.MagnitudeModel(embeddings.parent.name, embeddings.name)
        break

    if basic_model != None:

        current_depth = 1
        cluster = expansion.get_cluster(
            basic_model, keyword, max_width, max_depth, current_depth
        )
        assert cluster["sense"] == keyword
        assert cluster["similar_senses"] == None  # max_depth

        current_depth = 0
        cluster = expansion.get_cluster(
            basic_model, keyword, max_width, max_depth, current_depth
        )
        assert cluster["sense"] == keyword
        assert len(cluster["similar_senses"]) == max_width
        assert len(cluster["similar_senses"][0]) == 2  # tuple (sense, synonym_type)
        assert cluster["similar_senses"][0][0]["similar_senses"] == None  # max_depth
