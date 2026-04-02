import typer
from rich.panel import Panel
from rich.text import Text

from config import CliMessage
from console import console
from dataset import load_dataset
from vectorstore import get_vectorstore


app = typer.Typer(rich_markup_mode="rich")


@app.command()
def run(
    ctx: typer.Context,
    results: int = typer.Option(
        1, "--results", "-k", help="Number of results to return."
    )
) -> None:
    """The main CLI for the app."""
    vectorstore = ctx.obj

    console.print()
    console.print(CliMessage.QUERY_HINT)

    while True:
        question = console.input(CliMessage.USER_QUESTION).strip()

        if not question:
            continue

        with console.status(CliMessage.SEARCHING):
            documents = vectorstore.similarity_search(question, k=results)

        console.print()
        for index, document in enumerate(documents, start=1):
            title = Text(f"Result {index}/{len(documents)}", style="bold cyan")
            console.print(
                Panel(
                document.page_content,
                title=title,
                border_style="cyan",
                padding=(1, 2),
            )
        )

        console.print(CliMessage.NEW_QUERY)
        console.print("\n")


@app.callback(invoke_without_command=True)
def setup(ctx: typer.Context) -> None:
    """Load dataset and build vectorstore, storing it in context."""
    df = load_dataset()
    ctx.obj = get_vectorstore(df)
