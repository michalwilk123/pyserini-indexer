from pyserini.search.faiss import AutoQueryEncoder, FaissSearcher

encoder = AutoQueryEncoder("sentence-transformers/all-MiniLM-L6-v2")
searcher = FaissSearcher("tmp_index", encoder)
hits = searcher.search("what is a lobster roll")

for i in range(0, 10):
    print(f"{i+1:2} {hits[i].docid:7} {hits[i].score:.5f}")
