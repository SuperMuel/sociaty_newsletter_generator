import asyncio
from crewai_tools import tool
from typing import Literal
from crewai.crews.crew_output import CrewOutput
import logging
from langchain.chat_models.base import BaseChatModel

from sociaty_newsletter_generator.db import init_db
from sociaty_newsletter_generator.models import (
    Article,
    Cluster,
    ClusteringSession,
    SetOfUniqueArticles,
)
from make_it_sync import make_sync

from crewai import Agent, Task, Crew, Process
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv


from crewai.telemetry import Telemetry
from pydantic import BaseModel, Field

load_dotenv()


def noop(*args, **kwargs):
    pass


def disable_crewai_telemetry():
    for attr in dir(Telemetry):
        if callable(getattr(Telemetry, attr)) and not attr.startswith("__"):
            setattr(Telemetry, attr, noop)


disable_crewai_telemetry()

logger = logging.getLogger(__name__)

# TODO : add a research agent that will be responsible for consulting the websites already found in cluster, and finding more information
# TODO : add a questionner agent that will be responsible for asking questions about the current newsletter


class Subject(BaseModel):
    subject_id: str
    title: str
    summary: str
    formatted_material: str


class NewsletterRequest(BaseModel):
    main_subjects: list[Subject]
    secondary_subjects: list[Subject]
    # context: str | None = Field(
    #     default=None,
    #     description="Context provided by the user, explaining their company, audience, use case, etc.",
    # )
    language: Literal["en", "fr"] = "en"


class SubjectOutput(BaseModel):
    subject_id: str
    newsletter_title: str
    newsletter_body: str


def format_cluster(cluster: Cluster, articles: SetOfUniqueArticles) -> str:
    assert cluster.title and cluster.summary, "Cluster title and summary are required"
    assert articles

    articles_str = "\n\n".join(
        [f"{article.title}\n{article.body}\n{article.url}" for article in articles]
    )

    return f"{cluster.title}\n{cluster.summary}\n\n{articles_str}"


def dict_to_yaml(json: dict) -> str:
    yaml = ""
    for key, value in json.items():
        yaml += f"{key}: {value}\n"
    return yaml


class NewsletterCrew:
    def __init__(self, llm: BaseChatModel | str = "gpt-4o-mini"):
        if isinstance(llm, str):
            self.llm = init_chat_model(llm)
        else:
            self.llm = llm

    def generate_newsletter(self, request: NewsletterRequest):
        main_subject_writer = Agent(
            role="Main Subject Writer",
            goal="Write a 1 paragraph section for each main subject of the newsletter.",
            backstory="You are a skilled writer who excels in creating detailed and engaging content.",
            instructions=(
                "1. Analyse the material provided."
                "2. Write a 1 paragraph section based on the gathered information."
                "3. Ensure the content is informative and engaging."
            ),
            llm=self.llm,
        )

        secondary_subject_writer = Agent(
            role="Secondary Subject Writer",
            goal="Write a 2 sentences section for each secondary subject of the newsletter.",
            backstory="You are a concise writer who can create engaging content in a short format.",
            instructions=(
                "1. Analyse the material provided."
                "2. Write a 2 sentences section based on the gathered information."
                "3. Ensure the content is concise and engaging."
            ),
            llm=self.llm,
        )

        # Markdown formatter agent
        formatter_agent = Agent(
            role="Formatter",
            goal="Format the newsletter in Markdown in the `{language}` language.",
            backstory="You are an expert in Markdown formatting, ensuring that the newsletter is well-structured and visually appealing.",
            instructions=(
                "1. Compile the sections into a full newsletter."
                "2. Format the newsletter appropriately in Markdown, including titles, summaries, and links."
            ),
            llm=self.llm,
        )

        main_subject_tasks = [
            Task(
                description=(
                    f"Write a detailed section for the main subject '{subject.title}' based on the research provided. "
                    "Ensure that the content is engaging and informative. "
                    "Language for the newsletter : '{language}'\n"
                    f"Here is an overview of the subject : \n {dict_to_yaml(subject.model_dump())}\n"
                ),
                expected_output=f"A detailed newsletter section for one of the main subject '{subject.title}' composed of a title and a body.",
                output_pydantic=SubjectOutput,
                agent=main_subject_writer,
                llm=self.llm,
            )
            for subject in request.main_subjects
        ]

        secondary_subject_tasks = [
            Task(
                description=(
                    f"Write a brief section for the secondary subject '{subject.title}' based on the research provided."
                    " Ensure that the content is engaging and informative."
                    "Language for the newsletter : '{language}'"
                    f"Here is an overview of the subject : \n {dict_to_yaml(subject.model_dump())}\n"
                ),
                expected_output=f"A brief newsletter section for one of the secondary subject '{subject.title}' composed of a title and a 1 or 2 sentences body.",
                output_pydantic=SubjectOutput,
                agent=secondary_subject_writer,
                llm=self.llm,
            )
            for subject in request.secondary_subjects
        ]

        format_task = Task(
            description="Compile all sections into a full newsletter and format it in Markdown.",
            expected_output="A fully formatted newsletter in Markdown, in the `{language}` language.",
            agent=formatter_agent,
            llm=self.llm,
            context=main_subject_tasks + secondary_subject_tasks,
        )

        crew = Crew(
            agents=[
                main_subject_writer,
                secondary_subject_writer,
                formatter_agent,
            ],
            tasks=main_subject_tasks + secondary_subject_tasks + [format_task],
            process=Process.sequential,
            share_crew=False,
        )

        return crew.kickoff({"language": request.language})


async def main():
    await init_db()
    crew = NewsletterCrew()

    request = NewsletterRequest.model_validate(
        {
            "main_subjects": [
                {
                    "subject_id": "6697a91d6720e5b25ef7946f",
                    "title": "SK Hynix's $75 Billion Investment Focuses on AI-Enhanced HBM Processors",
                    "summary": "SK Hynix plans to invest $74.8 billion by 2028, with 80% of the emphasis on high-bandwidth memory chips for artificial intelligence applications. The parent company, SK Group, will allocate an additional $58 billion specifically for AI-related technologies and shareholder returns.",
                },
                {
                    "subject_id": "6697a91e6720e5b25ef7947c",
                    "title": "Controversy Surrounding AI Training on Web Content Sparks Legal Battles",
                    "summary": "Microsoft and OpenAI face lawsuits for using copyrighted online content to train AI models, sparking debates on fair use and intellectual property rights. Microsoft's AI CEO claims that content on the open web is considered 'freeware' for AI training, leading to backlash from content creators and legal challenges.",
                },
                {
                    "subject_id": "6697a91d6720e5b25ef7946e",
                    "title": "Morgan Freeman Condemns Unauthorized AI Voice Imitations",
                    "summary": "Morgan Freeman publicly criticizes the unauthorized use of AI technology to replicate his distinctive voice, thanking fans for alerting him to these imitations. The actor expresses his disapproval of AI-generated voice scams and deepfakes, joining other celebrities targeted by similar practices.",
                },
                {
                    "subject_id": "6697a91f6720e5b25ef7948f",
                    "title": "Apple Working to Integrate AI Features into Vision Pro Headsets",
                    "summary": "Apple is reportedly planning to bring its AI technology, Apple Intelligence, to the Vision Pro headset, expanding beyond its initial launch on other Apple devices. The integration may not happen this year, but is expected in the future to enhance the capabilities of the Vision Pro.",
                },
                {
                    "subject_id": "6697a91e6720e5b25ef79477",
                    "title": "States Addressing AI Skills Gap in Workforce",
                    "summary": "States are taking steps to help workers update their skills for an AI-driven workplace, with concerns about job obsolescence and the need for AI expertise. Employers are also seeking AI-savvy talent, while organizations globally are recognizing the opportunities AI presents despite lingering job loss concerns.",
                },
            ],
            "secondary_subjects": [
                {
                    "subject_id": "6697a91e6720e5b25ef7947b",
                    "title": "Apple Plans to Monetize Advanced AI Features with Subscription Model",
                    "summary": "Apple is considering introducing a paid subscription model for advanced features of its Apple Intelligence AI service, which currently offers a range of AI capabilities for free. The company aims to diversify its revenue sources beyond hardware sales by potentially launching a premium AI subscription service in the future.",
                },
                {
                    "subject_id": "6697a91f6720e5b25ef7948e",
                    "title": "Apple to Integrate Google's Gemini AI into iPhone for Enhanced AI Capabilities",
                    "summary": "Apple is set to partner with Google to integrate Gemini AI into iPhones, Macs, and iPads, expanding AI options for users. The announcement is expected this fall, coinciding with the iPhone 16 launch, as Apple aims to make AI a direct revenue stream.",
                },
                {
                    "subject_id": "6697a91f6720e5b25ef79494",
                    "title": "Chinese AI Startups Flock to Singapore for Global Expansion Amid US Restrictions",
                    "summary": "Chinese AI startups are relocating to Singapore to access capital and global markets due to challenges in China, including US regulatory hurdles and restrictions on AI investments. OpenAI's move to restrict API access in China further drives this trend.",
                },
                {
                    "subject_id": "6697a91f6720e5b25ef7948b",
                    "title": "Top Artificial Intelligence (AI) Stocks for Long-Term Investment",
                    "summary": "Leading AI companies like Nvidia, Amazon, and Alphabet are recommended for long-term investment due to their advancements in AI technology, cloud computing services, and AI chip manufacturing. Analysts predict strong growth and upside potential for these AI stocks in the coming years.",
                },
                {
                    "subject_id": "6697a9236720e5b25ef794cf",
                    "title": "Robinhood Enhances Investing App with AI Tools Through Acquisition of Pluto Capital",
                    "summary": "Robinhood has acquired Pluto Capital, an AI-powered research platform, to integrate advanced AI features into its investing app. This acquisition will enable Robinhood to provide tools for quicker identification of trends and investment opportunities, enhancing the overall user experience.",
                },
            ],
            "language": "fr",
        }
    )

    output = crew.generate_newsletter(request)
    print(output)


if __name__ == "__main__":
    asyncio.run(main())
