import asyncio
from typing import Literal
from crewai.crews.crew_output import CrewOutput
import logging

from sociaty_newsletter_generator.db import init_db
from sociaty_newsletter_generator.models import (
    Cluster,
    ClusteringSession,
    SetOfUniqueArticles,
)

from crewai import Agent, Task, Crew, Process
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv


from crewai.telemetry import Telemetry

load_dotenv()


def noop(*args, **kwargs):
    pass


def disable_crewai_telemetry():
    for attr in dir(Telemetry):
        if callable(getattr(Telemetry, attr)) and not attr.startswith("__"):
            setattr(Telemetry, attr, noop)


disable_crewai_telemetry()

logger = logging.getLogger(__name__)

llm = init_chat_model("gpt-4o-mini", temperature=0)

# TODO : add a research agent that will be responsible for consulting the websites already found in cluster, and finding more information
# TODO : add a questionner agent that will be responsible for asking questions about the current newsletter

writer = Agent(
    role="Writer",
    goal="Write a comprehensive and engaging newsletter from the provided clusters of news articles.",
    backstory="You are a talented writer with a knack for crafting detailed summaries and sections of the newsletter. You are a native speaker of the '{language}' language",
    instructions=(
        "1. Write detailed summaries and sections for each cluster."
        "2. Ensure the content is engaging, informative, and cohesive."
        "3. Compile the sections into a full newsletter."
        "4. Format the newsletter appropriately, including titles, summaries, and links."
    ),
    llm=llm,
)

writing_task = Task(
    description=(
        "Write detailed summaries and sections for the newsletter based on the research and summaries provided. Ensure that the content is engaging and informative."
        "Language for the newsletter : '{language}'"
        "Here are the researched subjects : \n\n{clusters}"
    ),
    expected_output="A full newsletter compiled from the clusters, formatted with titles, summaries, and links. Language : '{language}'",
    agent=writer,
    llm=llm,
)

# Define crew
crew = Crew(
    agents=[writer],
    tasks=[writing_task],
    process=Process.sequential,
    share_crew=False,
)


def format_cluster(cluster: Cluster, articles: SetOfUniqueArticles) -> str:
    assert cluster.title and cluster.summary, "Cluster title and summary are required"
    assert articles

    articles_str = "\n\n".join(
        [f"{article.title}\n{article.body}\n{article.url}" for article in articles]
    )

    return f"{cluster.title}\n{cluster.summary}\n\n{articles_str}"


def generate_newsletter(
    formatted_clusters: str,
    language: Literal["en", "fr"] = "en",
    # TODO : add llm here
) -> CrewOutput:
    return crew.kickoff(
        inputs={
            "clusters": formatted_clusters,
            "language": language,
        }
    )


async def main():
    await init_db()
    logger.info("Connected to MongoDB")
    session_id = "669a345da02feb1858343077"
    session = await ClusteringSession.get(session_id)
    logger.info("Session found")
    assert session

    clusters = await Cluster.find_many({"session.$id": session.id}).to_list()

    # order clusters by size
    clusters = sorted(clusters, key=lambda x: x.articles_count, reverse=True)

    # limit to 5 clusters
    clusters = clusters[:5]

    clusters_str = ""
    for i, cluster in enumerate(clusters, start=1):
        articles = SetOfUniqueArticles(await cluster.get_articles()).limit(20)
        clusters_str += f"<subject_{i}>\n"
        clusters_str += format_cluster(cluster, articles)
        clusters_str += f"</subject_{i}>\n"

    print(generate_newsletter(clusters_str))


if __name__ == "__main__":
    asyncio.run(main())
