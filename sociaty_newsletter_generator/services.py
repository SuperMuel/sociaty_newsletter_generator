from textwrap import dedent
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.pydantic_v1 import BaseModel, Field


from sociaty_newsletter_generator.models import SetOfUniqueArticles

Language = Literal["fr", "en"]


class SummaryOutput(BaseModel):
    """Output for the summarization process. Every field is mandatory."""

    scratchpad: str = Field(
        ...,
        description="Very dense set of facts and ideas separated by comas that should cover a large amount of the content.",
    )
    missing_entities: str = Field(
        ...,
        description="Mandatory list of ideas or facts that you forgot to include in the scratchpad.",
    )
    final_summary: str = Field(
        ...,
        description="Final comprehensive summary of the content including the scratchpad and forgotten entities.",
    )


type ArticlesSummarizerChain = Runnable[SetOfUniqueArticles, str]


def create_articles_summarizer_chain(
    llm: BaseChatModel | str = "gpt-4o-mini",
    language: Language = "en",
    articles_limit: int | None = 100,
) -> ArticlesSummarizerChain:
    if isinstance(llm, str):
        llm = init_chat_model(llm)
    structured_llm = llm.with_structured_output(SummaryOutput)

    def _format_articles(articles: SetOfUniqueArticles):
        return "\n\n".join(
            f"{article.title}\n{article.date}\n{article.url}\n{article.body}"
            for article in articles.limit(articles_limit)
        )

    prompt = PromptTemplate.from_template(
        dedent("""Here is a set of news articles to summarize : 

                                <articles>
                                {formatted_articles}
                                </articles>

                                Write a comprehensive summary of this content in {language} language.
                                Include as much facts and ideas as possible in the summary.""")
    )

    return (
        {
            "formatted_articles": RunnableLambda(_format_articles),
            "language": RunnableLambda(lambda _: language),
        }
        | prompt
        | structured_llm
        | RunnableLambda(lambda x: x.final_summary)
    ).with_config(run_name="articles_summarizer_chain")
