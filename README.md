# pyserini-indexer

Wrapper around pyserini for building various search indexes from the command line

### File format
All data is stored under JSONL format. 
Each line contains a object with below schema:
```javascript
{
    "id": 1234,
    "contents": "The aegis of Athena is referred to in several places in the Iliad. \"It produced a sound as from myriad roaring dragons (Iliad, 4.17) and was borne by Athena in battle ... and among them went bright-eyed Athene, holding the precious aegis which is ageless and immortal: a hundred tassels of pure gold hang fluttering from it, tight-woven each of them, and each the worth of a hundred oxen."
}
```

Command automatically does splitting, chooses appropriate model links and generates index files.

### Google Colab
If you want to test out package, you can try hosted notebook:

https://colab.research.google.com/drive/1OUKKtnHpI6NQ-EhY31F34t7C7stMY8U0?usp=sharing


### Prerequisites
* python 3.9+ : currently there is an issue with the pytorch, so that newer version of python brake
* if you would want to create/search using learnable 
Sparse indexes (UNICOIL), you would need CUDA support (pyserini requires here CUDA for backend 😞)

### Installation
```
pip install pyserini-indexer
```

<br/>

### Indexing

### Indexing methods:
* Sparse:
    * BM25 - traditional not-learnable indexing technique
    * UNICOIL - uses LuceneImpact underneath
* Dense:
    * MINILM_V2 - smallest model available
    * ANCE
    * TCT_COLBERT - biggest (0.5 gb) and provides best results

### Backend support
By default **all** indexing methods use CPU. If you would want to change this behaviour, set environment variable **USE_GPU** to some value.

Example usage
1. Create BM25 sparse index directory
```python
pyserini-indexer BM25 index_input -o tests/indexes/bm25
```

2. Create dense index directory
```python
pyserini-indexer MINILM_V2 index_input -o tests/indexes/minilm_v2
```

---

### Searching

Use also can use the package to perform searching through the index like for example:

```python
import pyserini_indexer

searcher = pyserini_indexer.get_searcher("index_dir", dense_model="ANCE")

hits = searcher.search("Some test query")
```

##### Result:

```bash
[DenseSearchResult(docid='p34ezktf', score=39.51640439033508),
 DenseSearchResult(docid='t40ybhgb', score=39.21655011177063),
 DenseSearchResult(docid='e62cfqt7', score=39.122857332229614),
 DenseSearchResult(docid='tvxpckxo', score=39.01145672798157),
 DenseSearchResult(docid='s64v656n', score=38.794487071037295),
 DenseSearchResult(docid='58czem0j', score=38.787667059898375),
 DenseSearchResult(docid='5dk231qs', score=38.733979558944704),
 DenseSearchResult(docid='ajlctjeb', score=38.71289944648743),
 DenseSearchResult(docid='kvhoa2se', score=38.7067670583725),
 DenseSearchResult(docid='vw8xjo9t', score=38.684127068519594)]
```

#### Available searchers:

* sparse searcher: uses pyserini.search.lucene.LuceneSearcher or pyserini.search.lucene.LuceneImpactSearcher
* dense searcher: uses pyserini.search.faiss.FaissSearcher
* hybrid searcher: uses pyserini.search.hybrid.HybridSearcher

The searcher is chosen based on existence of model name (sparse_model or dense_model). If both are provided, than hybrid searcher gets created. For this kind of searcher, both index directories should be provided. The directory for dense searcher should be set in __second_index_dir__ argument
