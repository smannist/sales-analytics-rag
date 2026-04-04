import pandas as pd
import typer
from langchain_chroma import Chroma
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text


import factories as _  # noqa: F401 -- current decorator pattern requires importing the factories too
from config import CliMessage
from console import console
from dataset import download_dataset, load_dataset
from llm import retrieve, generate_answer, determine_retrieval_plan
from registry import DOCUMENT_FACTORY_REGISTRY
from vectorstore import (
    get_vectorstore,
    populate_vectorstore,
    is_vectorstore_empty
)


app = typer.Typer(rich_markup_mode="rich")


@app.command()
def run(ctx: typer.Context) -> None:
    """The main CLI for the app."""
    vectorstore = ctx.obj

    console.print()
    console.print(
        Panel(
            CliMessage.WELCOME,
            style="bold cyan",
            padding=(1, 2)
        )
    )
    console.print()
    console.print(CliMessage.QUERY_HINT, style="bold")

    while True:
        question = console.input(
            Text(
                CliMessage.INPUT_PROMPT,
                style="bold cyan"
            )
        ).strip()

        if not question:
            continue

        with console.status(
            Text(
                CliMessage.SEARCHING,
                style="bold green"
            )
        ):
            plan = determine_retrieval_plan(question)      # type: ignore[arg-type]
            documents = retrieve(vectorstore, plan)        # type: ignore[arg-type]
            answer = generate_answer(question, documents)  # type: ignore[arg-type]

        console.print()
        console.print(
            Panel(
                Markdown(answer),
                title="Answer",
                border_style="green",
                padding=(1, 2)
            )
        )

        console.print()
        console.print(CliMessage.NEW_QUERY, style="dim")
        console.print()


@app.callback(invoke_without_command=True)
def setup(ctx: typer.Context) -> None:
    """Load dataset and build vectorstore, storing it in context."""
    df = _get_dataset()
    vectorstore = get_vectorstore()

    if is_vectorstore_empty(vectorstore):
        vectorstore = _build_vectorstore(df, vectorstore)
    else:
        console.print(CliMessage.ALREADY_POPULATED, style="dim")

    ctx.obj = vectorstore


def _get_dataset() -> pd.DataFrame:
    """Loads the dataset, if not present, then downloads it.

    Returns:
        The dataset as a pandas DataFrame.
    """
    try:
        df = load_dataset()
        console.print(CliMessage.DATASET_EXISTS, style="dim")
    except FileNotFoundError:
        console.print(CliMessage.DATASET_DOWNLOADING, style="yellow")
        download_dataset()
        df = load_dataset()
        console.print(CliMessage.DATASET_DOWNLOADED, style="green")
    return df


def _build_vectorstore(
        df: pd.DataFrame,
        vectorstore: Chroma
) -> Chroma:
    """Builds the vectorstore.

    Args:
        df: The pandas DataFrame.
        vectorstore: The Chroma vectorstore.

    Returns:
        The Chroma vectorstore instance.
    """
    documents = [
         doc
         for document_factory in DOCUMENT_FACTORY_REGISTRY
         for doc in document_factory(df)
    ]

    with console.status(
        Text(
            CliMessage.INSERTING,
            style="bold green"
        )
    ):
        for inserted, total in populate_vectorstore(vectorstore, documents):
            console.print(
                f"{inserted}/{total} inserted.",
                style="dim"
            )
        console.print(CliMessage.INSERTED, style="green")

    return vectorstore
