import gensim.downloader as api

model = api.load("glove-wiki-gigaword-100")

def explore_word_relationships():
    print("Exploring word relationships through arithmetic operations:")

    equation_1 = model.most_similar(positive=["king", "woman"], negative=["man"], topn=1)
    print(f"king - man + woman = {equation_1[0][0]}")

    equation_2 = model.most_similar(positive=["ocean", "rain"], negative=["water"], topn=1)
    print(f"ocean - water + rain = {equation_2[0][0]}")

    equation_3 = model.most_similar(positive=["teacher", "knowledge"], negative=["educator"], topn=1)
    print(f"teacher - educator + knowledge = {equation_3[0][0]}")

explore_word_relationships()
