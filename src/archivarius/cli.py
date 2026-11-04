"""Интерфейс командной строки."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from archivarius import __version__
from archivarius.config import SEARCH_MODES, Settings
from archivarius.engine import Engine
from archivarius.ollama_client import OllamaClient, OllamaError
from archivarius.store import EmbeddingMismatchError

app = typer.Typer(
    help="Archivarius: вопросно-ответный поиск по локальным документам через Ollama.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

ConfigOption = typer.Option(None, "--config", "-c", help="Путь к YAML-файлу настроек")
KOption = typer.Option(None, "--k", "-k", help="Число фрагментов")
ModeOption = typer.Option(None, "--mode", "-m", help="bm25 | dense | hybrid")


def _engine(config: Path | None) -> Engine:
    try:
        return Engine(Settings.load(config))
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]Ошибка конфигурации:[/red] {exc}")
        raise typer.Exit(2) from exc


def _fail(exc: Exception) -> None:
    console.print(f"[red]{escape(str(exc))}[/red]")
    raise typer.Exit(1) from exc


def _check_mode(mode: str | None) -> None:
    if mode is not None and mode not in SEARCH_MODES:
        console.print(f"[red]Режим должен быть одним из: {', '.join(SEARCH_MODES)}[/red]")
        raise typer.Exit(2)


@app.command()
def version() -> None:
    """Показать версию."""
    console.print(f"archivarius {__version__}")


@app.command()
def init(path: Path = typer.Argument(Path("archivarius.yaml"))) -> None:
    """Создать файл настроек со значениями по умолчанию."""
    import yaml

    if path.exists():
        console.print(f"[yellow]Файл уже существует:[/yellow] {path}")
        raise typer.Exit(1)
    path.write_text(
        yaml.safe_dump(Settings().to_dict(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    console.print(f"Создан {path}")


@app.command()
def doctor(config: Path | None = ConfigOption) -> None:
    """Проверить доступность Ollama и наличие нужных моделей."""
    settings = Settings.load(config)
    with OllamaClient(settings.ollama_url, timeout=10) as client:
        try:
            console.print(f"Ollama {client.version()} доступна: {settings.ollama_url}")
            models = client.list_models()
        except OllamaError as exc:
            _fail(exc)
    ok = True
    for role, name in (("эмбеддинги", settings.embed_model), ("генерация", settings.llm_model)):
        present = any(m == name or m == f"{name}:latest" for m in models)
        mark = "[green]есть[/green]" if present else f"[red]нет[/red], выполните ollama pull {name}"
        console.print(f"  {role}: {name} ... {mark}")
        ok &= present
    raise typer.Exit(0 if ok else 1)


@app.command()
def index(
    path: Path = typer.Argument(..., help="Файл или каталог с документами"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Очистить индекс перед загрузкой"),
    config: Path | None = ConfigOption,
) -> None:
    """Проиндексировать документы (повторный запуск обрабатывает только изменения)."""
    labels = {
        "added": "[green]+[/green]",
        "updated": "[cyan]~[/cyan]",
        "removed": "[red]-[/red]",
        "failed": "[red]![/red]",
        "empty": "[yellow]0[/yellow]",
    }

    def progress(status: str, file: Path) -> None:
        if status in labels:
            console.print(f" {labels[status]} {escape(str(file))}")

    with _engine(config) as engine:
        try:
            with console.status("Индексация…"):
                report = engine.index(path, rebuild=rebuild, progress=progress)
        except (OllamaError, EmbeddingMismatchError, FileNotFoundError) as exc:
            _fail(exc)
        console.print(
            f"Готово: добавлено {report.added}, обновлено {report.updated}, "
            f"без изменений {report.unchanged}, удалено {report.removed}, "
            f"пустых {report.empty}, ошибок {len(report.failed)}; новых фрагментов {report.chunks}"
        )
        for file, error in report.failed:
            console.print(f"  [red]{escape(file)}[/red]: {escape(error)}")


@app.command()
def search(
    query: str,
    k: int | None = KOption,
    mode: str | None = ModeOption,
    as_json: bool = typer.Option(False, "--json", help="Вывести результат в JSON"),
    config: Path | None = ConfigOption,
) -> None:
    """Найти фрагменты без генерации ответа."""
    _check_mode(mode)
    with _engine(config) as engine:
        try:
            hits = engine.search(query, k, mode)
        except OllamaError as exc:
            _fail(exc)
    if as_json:
        print(json.dumps([h.to_dict() for h in hits], ensure_ascii=False, indent=2))
        return
    if not hits:
        console.print("Ничего не найдено")
        return
    table = Table(show_lines=True)
    for column in ("#", "Документ", "Место", "Оценка", "Фрагмент"):
        table.add_column(column)
    for i, hit in enumerate(hits, start=1):
        preview = hit.chunk.text[:220].replace("\n", " ")
        table.add_row(
            str(i), hit.chunk.title, hit.chunk.location, f"{hit.score:.4f}", escape(preview)
        )
    console.print(table)


def _print_sources(sources: list[dict], citations: list[int]) -> None:
    for i, source in enumerate(sources, start=1):
        mark = "[bold yellow]*[/bold yellow]" if i in citations else " "
        place = f", {source['location']}" if source["location"] else ""
        console.print(f" {mark} [{i}] {escape(source['title'])}{escape(place)}")


def _ask_once(engine: Engine, question: str, k: int | None, mode: str | None) -> None:
    sources: list[dict] = []
    for event in engine.ask_stream(question, k, mode):
        if event["type"] == "sources":
            sources = event["sources"]
        elif event["type"] == "token":
            console.print(event["text"], end="", markup=False, highlight=False)
        elif event["type"] == "done":
            console.print()
            console.print("\n[dim]Источники:[/dim]")
            _print_sources(sources, event["citations"])
            t = event["timings"]
            console.print(
                f"[dim]поиск {t['retrieval_ms']} мс, генерация {t['generation_ms']} мс[/dim]"
            )


@app.command()
def ask(
    question: str,
    k: int | None = KOption,
    mode: str | None = ModeOption,
    config: Path | None = ConfigOption,
) -> None:
    """Получить ответ на вопрос со ссылками на источники."""
    _check_mode(mode)
    with _engine(config) as engine:
        try:
            _ask_once(engine, question, k, mode)
        except OllamaError as exc:
            _fail(exc)


@app.command()
def chat(
    k: int | None = KOption,
    mode: str | None = ModeOption,
    config: Path | None = ConfigOption,
) -> None:
    """Интерактивный режим вопросов и ответов. Выход: пустая строка или Ctrl+C."""
    _check_mode(mode)
    with _engine(config) as engine:
        console.print(Panel.fit("Archivarius: задавайте вопросы, пустая строка завершает работу"))
        while True:
            try:
                question = console.input("[bold]> [/bold]").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                break
            try:
                _ask_once(engine, question, k, mode)
            except OllamaError as exc:
                console.print(f"[red]{escape(str(exc))}[/red]")
            console.print()


@app.command()
def stats(config: Path | None = ConfigOption) -> None:
    """Показать состояние индекса."""
    with _engine(config) as engine:
        data = engine.stats()
        documents = engine.store.list_documents()
    table = Table(show_header=False)
    for key, value in data.items():
        table.add_row(key, str(value))
    console.print(table)
    for doc in documents[:50]:
        console.print(f"  {escape(doc.path)}")
    if len(documents) > 50:
        console.print(f"  … и ещё {len(documents) - 50}")


@app.command("eval")
def evaluate_command(
    dataset: Path = typer.Argument(..., help="JSONL с полями question и relevant"),
    k: int = typer.Option(5, "--k", "-k"),
    modes: str = typer.Option("bm25,dense,hybrid", "--modes", help="Режимы через запятую"),
    report: Path | None = typer.Option(None, "--report", help="Сохранить отчёт в Markdown"),
    details: Path | None = typer.Option(None, "--details", help="Сохранить детали в JSON"),
    config: Path | None = ConfigOption,
) -> None:
    """Сравнить режимы поиска по метрикам Hit@k, Recall@k, MRR и nDCG."""
    from archivarius.evaluation import evaluate, load_dataset, questions_label, render_markdown

    mode_list = [m.strip() for m in modes.split(",") if m.strip()]
    for mode in mode_list:
        _check_mode(mode)
    items = load_dataset(dataset)
    with _engine(config) as engine:
        try:
            with console.status("Оценка…"):
                results = evaluate(engine.retriever, items, mode_list, k)
        except OllamaError as exc:
            _fail(exc)

    table = Table(title=f"{questions_label(len(items))}, k = {k}")
    for column in ("Режим", f"Hit@{k}", f"Recall@{k}", f"MRR@{k}", f"nDCG@{k}", "мс"):
        table.add_column(column, justify="right")
    best = max(r.mrr for r in results)
    for r in results:
        style = "bold green" if r.mrr == best else ""
        table.add_row(
            r.mode,
            f"{r.hit_rate:.3f}",
            f"{r.recall:.3f}",
            f"{r.mrr:.3f}",
            f"{r.ndcg:.3f}",
            f"{r.latency_ms:.1f}",
            style=style,
        )
    console.print(table)
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(render_markdown(results, len(items)), encoding="utf-8")
        console.print(f"Отчёт: {report}")
    if details:
        details.parent.mkdir(parents=True, exist_ok=True)
        payload = [{**r.summary(), "questions": r.per_question} for r in results]
        details.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"Детали: {details}")


@app.command("eval-generate")
def eval_generate(
    output: Path = typer.Argument(..., help="Куда сохранить JSONL"),
    n: int = typer.Option(30, "--n", "-n", help="Число вопросов"),
    seed: int = typer.Option(42, "--seed"),
    config: Path | None = ConfigOption,
) -> None:
    """Сгенерировать синтетический набор вопросов по проиндексированным документам."""
    from archivarius.evaluation import generate_synthetic, save_dataset

    with _engine(config) as engine:
        try:
            with console.status(f"Генерация {n} вопросов…"):
                items = generate_synthetic(engine.store, engine.llm, n=n, seed=seed)
        except (OllamaError, ValueError) as exc:
            _fail(exc)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(items, output)
    console.print(
        f"Сохранено {len(items)} вопросов в {output}. Проверьте их вручную перед оценкой."
    )


def _port_is_free(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    config: Path | None = ConfigOption,
) -> None:
    """Запустить веб-интерфейс и HTTP API."""
    import uvicorn

    from archivarius.api import create_app

    if not _port_is_free(host, port):
        console.print(
            f"[red]Порт {port} занят.[/red] Возможно, сервер уже запущен в другом окне: "
            f"остановите его через Ctrl+C или укажите другой порт, например --port {port + 1}"
        )
        raise typer.Exit(1)
    engine = _engine(config)
    console.print(f"Веб-интерфейс: http://{host}:{port}  ·  API: http://{host}:{port}/docs")
    try:
        uvicorn.run(create_app(engine), host=host, port=port, log_level="warning")
    finally:
        engine.close()


if __name__ == "__main__":
    app()
