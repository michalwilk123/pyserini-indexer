import functools
from typing import Optional

import pyserini.search.faiss as faiss
import pyserini.search.hybrid as hybrid
import pyserini.search.lucene as lucene

import utils.models as models


@functools.lru_cache
def get_searcher(
    index_dir: str,
    *,
    dense_model: Optional[str] = None,
    sparse_model: Optional[str] = None,
):
    dense_searcher = None
    sparse_searcher = None

    if dense_model == models.DenseIndexes.ANCE.value:
        encoder = faiss.AnceQueryEncoder(models.ANCE_MODEL)
        dense_searcher = faiss.FaissSearcher(index_dir, encoder)
    elif dense_model == models.DenseIndexes.TCT_COLBERT.value:
        encoder = faiss.TctColBertQueryEncoder(models.DenseIndexes.TCT_COLBERT)
        dense_searcher = faiss.FaissSearcher(index_dir, encoder)
    elif dense_model == models.DenseIndexes.MINILM_V2.value:
        encoder = faiss.AutoQueryEncoder(models.AUTO_MODEL)
        dense_searcher = faiss.FaissSearcher(index_dir, encoder)

    if sparse_model == models.SparseIndexes.BM25.value:
        sparse_searcher = lucene.LuceneSearcher(index_dir)
    elif sparse_model == models.SparseIndexes.UNICOIL.value:
        sparse_searcher = lucene.LuceneImpactSearcher(index_dir, models.UNICOIL_MODEL)

    if sparse_searcher and dense_searcher:
        return hybrid.HybridSearcher(dense_searcher, sparse_model)
    elif sparse_searcher:
        return sparse_searcher
    elif dense_searcher:
        return dense_searcher

    raise RuntimeError(
        "Unknown models: \nsparse{} \ndense{}".format(sparse_model, dense_model)
    )
