"""
build_index.py — Скрипт построения векторного индекса FAISS для RAG-бота QuantumForge.

Архитектурный выбор:
- RecursiveCharacterTextSplitter: иерархическое разбиение по разделителям
  (параграфы → предложения → слова), что сохраняет семантическую целостность
  энциклопедических текстов лучше, чем фиксированное разбиение по символам.
- chunk_size=800 / chunk_overlap=150: ~200 wordpiece-токенов на чанк
  (с запасом до лимита all-MiniLM-L6-v2 в 256 токенов). Перекрытие 150 символов
  (~19%) гарантирует, что граничные факты не потеряются при разбиении.
- HuggingFaceEmbeddings через `langchain_huggingface`: нативный импорт без deprecated
  обёрток из `langchain_community.embeddings`.
- FAISS: максимально производительный in-memory поиск с сериализацией на диск.
"""

from __future__ import annotations

import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge_base"
INDEX_DIR = PROJECT_ROOT / "vector_index"

CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 150

EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"


def load_markdown_documents(directory: Path) -> list[Document]:
    """Загружает все .md файлы из директории как список Document.

    Каждый документ снабжается метаданными с именем исходного файла —
    это позволит в будущем цитировать источник ответа.

    Args:
        directory: Путь к папке с Markdown-файлами.

    Returns:
        Список объектов Document с заполненными page_content и metadata.
    """
    docs: list[Document] = []

    if not directory.exists():
        raise FileNotFoundError(f"Директория базы знаний не найдена: {directory}")

    for md_path in sorted(directory.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        docs.append(
            Document(
                page_content=text,
                metadata={"source": md_path.name},
            )
        )

    return docs


def split_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Разбивает документы на чанки с помощью RecursiveCharacterTextSplitter.

    Args:
        documents: Исходные документы.
        chunk_size: Максимальный размер чанка в символах.
        chunk_overlap: Перекрытие между соседними чанками в символах.

    Returns:
        Список чанков (Document), где каждый чанк наследует metadata от родителя.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)


def build_and_save_index(
    chunks: list[Document],
    embedding_model: str,
    index_dir: Path,
) -> FAISS:
    """Строит индекс FAISS из чанков и сохраняет его на диск.

    Args:
        chunks: Чанки для индексации.
        embedding_model: Идентификатор модели эмбеддингов (HuggingFace).
        index_dir: Путь для сохранения индекса.

    Returns:
        Созданный объект FAISS.
    """
    print(f"Инициализация эмбеддинг-модели: {embedding_model} ...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    print(f"Построение FAISS-индекса из {len(chunks)} чанков ...")
    start = time.perf_counter()
    vector_store = FAISS.from_documents(chunks, embeddings)
    elapsed = time.perf_counter() - start

    index_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_dir))

    print(f"Индекс сохранён в: {index_dir} (время: {elapsed:.2f} сек)")
    return vector_store


def main() -> None:
    """Основной пайплайн: загрузка → чанкование → эмбеддинги → индекс."""
    print("=" * 60)
    print("QuantumForge RAG — Построение векторного индекса")
    print("=" * 60)

    # 1. Загрузка документов
    print(f"\n[1/4] Загрузка документов из {KNOWLEDGE_DIR} ...")
    documents = load_markdown_documents(KNOWLEDGE_DIR)
    print(f"      Загружено документов: {len(documents)}")

    # 2. Чанкование
    print(
        f"\n[2/4] Чанкование (chunk_size={CHUNK_SIZE}, "
        f"chunk_overlap={CHUNK_OVERLAP}) ..."
    )
    chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"      Получено чанков: {len(chunks)}")

    # 3. Эмбеддинги + индекс
    print("\n[3/4] Генерация эмбеддингов и построение FAISS-индекса ...")
    build_and_save_index(chunks, EMBEDDING_MODEL, INDEX_DIR)

    # 4. Smoke-test
    print("\n[4/4] Дымовой тест: поиск по запросу ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    query = "Кто такой Xarn Velgor и как он погиб?"
    results = vector_store.similarity_search(query, k=3)

    print(f'  Запрос: "{query}"')
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"  [{i}] {source} | {preview}...")

    print("\n" + "=" * 60)
    print("Готово! Индекс успешно создан и протестирован.")
    print("=" * 60)


if __name__ == "__main__":
    main()