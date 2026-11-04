from archivarius.bm25 import BM25Index

DOCS = [
    "Оконные функции SQL позволяют считать накопительные итоги",
    "Docker контейнер запускается из образа",
    "Виртуальное окружение Python создаётся через venv",
    "Функция ROW_NUMBER нумерует строки окна",
]


def test_bm25_ranks_relevant_first():
    index = BM25Index().fit(DOCS)
    top = index.top("оконная функция", 2)
    assert top[0][0] in (0, 3)
    assert all(score > 0 for _, score in top)


def test_bm25_morphology():
    index = BM25Index().fit(DOCS)
    assert index.top("контейнеры", 1)[0][0] == 1


def test_bm25_unknown_query_returns_nothing():
    index = BM25Index().fit(DOCS)
    assert index.top("квантовая хромодинамика", 3) == []


def test_bm25_empty_corpus():
    index = BM25Index().fit([])
    assert index.size == 0
    assert index.top("что угодно", 3) == []


def test_bm25_idf_prefers_rare_terms():
    docs = ["данные данные отчёт", "данные витрина", "данные метрика"]
    index = BM25Index().fit(docs)
    assert index.top("данные витрина", 1)[0][0] == 1
