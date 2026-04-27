from dataclasses import dataclass

import pandas as pd
import tiktoken
import typer
from langchain_chroma import Chroma
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, trim_messages
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text


from config import ChatHistoryConfig, CliMessage
from console import console
from dataset import download_dataset, load_dataset
from llm import (
    AnswerStrategy,
    determine_retrieval_plan,
    generate_answer,
    generate_followup_answer,
    retrieve,
)
from registry import load_document_factories
from vectorstore import (
    get_vectorstore,
    populate_vectorstore,
    is_vectorstore_empty
)


app = typer.Typer(rich_markup_mode="rich")
_encoding = tiktoken.get_encoding("cl100k_base")


@dataclass(frozen=True)
class AppContext:
    """Stores CLI app context."""
    vectorstore: Chroma
    chat_history: InMemoryChatMessageHistory


@app.command()
def run(ctx: typer.Context) -> None:
    """The main CLI for the app."""
    vectorstore = ctx.obj.vectorstore
    chat_history = ctx.obj.chat_history

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

        history = trim_messages(
            chat_history.messages,
            max_tokens=ChatHistoryConfig.MAX_HISTORY_TOKENS,
            strategy="last",
            token_counter=_count_tokens,
            start_on="human",
            include_system=False,
        )

        plan = determine_retrieval_plan(question, history)  # type: ignore[arg-type]

        status_message = (
            CliMessage.FOLLOWING_UP
            if plan.strategy == AnswerStrategy.FOLLOW_UP
            else CliMessage.SEARCHING
        )

        with console.status(
            Text(
                status_message,
                style="bold green"
            )
        ):
            if plan.strategy == AnswerStrategy.FOLLOW_UP:
                answer = generate_followup_answer(question, history)    # type: ignore[arg-type]
            else:
                documents = retrieve(vectorstore, plan)                 # type: ignore[invalid-argument-type]
                answer = generate_answer(question, documents, history)  # type: ignore[arg-type]
            chat_history.add_user_message(question)
            chat_history.add_ai_message(answer)

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
    """Load dataset, build vectorstore, and prepare chat history."""
    df = _get_dataset()
    vectorstore = get_vectorstore()

    if is_vectorstore_empty(vectorstore):
        vectorstore = _build_vectorstore(df, vectorstore)
    else:
        console.print(CliMessage.ALREADY_POPULATED, style="dim")

    ctx.obj = AppContext(
        vectorstore=vectorstore,
        chat_history=InMemoryChatMessageHistory(),
    )


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
         for document_factory in load_document_factories()
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


def _count_tokens(messages: list[BaseMessage]) -> int:
    """Count tokens, works for Groq, too.

    Returns:
        The total token count across all messages.
    """
    return sum(len(_encoding.encode(m.text)) for m in messages)
