from dataclasses import dataclass
from datetime import UTC, date, datetime
from math import floor
from typing import Any, Dict, Iterator, List, Tuple

from beanie import Document, Link, PydanticObjectId
from pydantic import Field
from sociaty_newsletter_generator.settings import Settings

DateOrDatetime = date | datetime


@dataclass(frozen=True)
class TitleBodyDate:
    """
    A data class that represents an article with a title, body, and date.

    Attributes:
        title (str): The title of the article.
        body (str): The body of the article.
        date (DateOrDatetime): The date of the article.
    """

    title: str
    body: str
    date: DateOrDatetime

    @staticmethod
    def from_dict(data: dict) -> "TitleBodyDate":
        """
        Create a TitleBodyDate instance from a dictionary.

        Args:
            data (dict): A dictionary containing the title, body, and date of the article.

        Returns:
            TitleBodyDate: A TitleBodyDate instance created from the dictionary.
        """
        return TitleBodyDate(data["title"], data["body"], data["date"])


class SetOfUniqueArticles:
    """
    A class designed to reduce the amount of duplicated or similar articles passed to LLMs.

    Articles are considered the same if:
    - They have the same title (case insensitive, ignoring leading/trailing whitespaces)
    - They have the same body (case insensitive, ignoring leading/trailing whitespaces)
    - They have the same date (ignoring the time part if it exists)

    If one article's body is a prefix of another's, the article with the longer body is kept.
    """

    def __init__(self, input_data: TitleBodyDate | List[TitleBodyDate] | None = None):
        """
        Initialize the set of unique articles.

        Args:
            articles (List[TitleBodyDate], optional): A list of initial articles to add.
        """
        self.articles: Dict[Tuple[str, date], TitleBodyDate] = {}
        if input_data:
            if isinstance(input_data, TitleBodyDate):
                input_data = [input_data]
            for article in input_data:
                self.add_article(article)

    def _normalize_string(self, s: str) -> str:
        """
        Normalize a string by trimming whitespace and converting to lowercase.

        Args:
            s (str): The string to normalize.

        Returns:
            str: The normalized string.
        """
        return s.strip().lower()

    def _normalize_date(self, d: DateOrDatetime) -> date:
        """
        Normalize a date or datetime by extracting the date part.

        Args:
            d (DateOrDatetime): The date or datetime to normalize.

        Returns:
            date: The normalized date.
        """
        if isinstance(d, datetime):
            return d.date()
        return d

    def _generate_key(self, article: TitleBodyDate) -> Tuple[str, date]:
        """
        Generate a key for an article based on its normalized title and date.

        Args:
            article (TitleBodyDate): The article to generate the key for.

        Returns:
            Tuple[str, date]: The generated key.
        """
        normalized_title = self._normalize_string(article.title)
        normalized_date = self._normalize_date(article.date)
        return (normalized_title, normalized_date)

    def add_article(self, article: TitleBodyDate):
        """
        Add an article to the set of unique articles.

        Args:
            article (TitleBodyDate): The article to add.
        """
        key = self._generate_key(article)
        normalized_body = self._normalize_string(article.body)

        if key not in self.articles:
            self.articles[key] = article
            return

        existing_article = self.articles[key]
        existing_body = self._normalize_string(existing_article.body)

        if (
            normalized_body != existing_body
            and normalized_body.startswith(
                existing_body[
                    : floor(
                        len(existing_body) * 0.9
                    )  # fix if for instance, the shorter existing body ends with an extra dot
                ]
            )
        ):
            self.articles[key] = article

    def get_articles(self) -> List[TitleBodyDate]:
        """
        Get the list of unique articles.

        Returns:
            List[TitleBodyDate]: The list of unique articles.
        """
        return list(self.articles.values())

    def __iter__(self) -> Iterator[TitleBodyDate]:
        """
        Return an iterator over the unique articles.

        Returns:
            Iterator[TitleBodyDate]: An iterator over the unique articles.
        """
        return iter(self.get_articles())

    def __repr__(self) -> str:
        """
        Return a string representation of the set of unique articles.

        Returns:
            str: A string representation of the set of unique articles.
        """

        articles = [article.__repr__() for article in self.get_articles()]
        class_name = self.__class__.__name__
        return f"{class_name}([{',\n    '.join(articles)}])"

    def __bool__(self) -> bool:
        """
        Return True if the set of unique articles is not empty, False otherwise.

        Returns:
            bool: True if the set of unique articles is not empty, False otherwise.
        """
        return bool(self.articles)

    def limit(self, n: int) -> "SetOfUniqueArticles":
        """
        Limit the number of articles to be returned.

        Args:
            n (int): The number of articles to return.

        Returns:
            SetOfUniqueArticles: A new SetOfUniqueArticles instance with the limited number of articles.
        """
        return SetOfUniqueArticles(list(self.get_articles())[:n])

    def __len__(self) -> int:
        """
        Return the number of unique articles.

        Returns:
            int: The number of unique articles.
        """
        return len(self.articles)


class ClusteringSession(Document):
    session_start: datetime = Field(default_factory=lambda: datetime.now(UTC))
    session_end: datetime | None = None

    data_start: datetime
    data_end: datetime

    metadata: Dict[str, Any]

    articles_count: int = Field(
        ...,
        description="Number of articles on which the clustering was performed, including noise.",
    )
    clusters_count: int

    noise_articles_ids: List[PydanticObjectId]
    noise_articles_count: int
    clustered_articles_count: int = Field(
        ...,
        description="Number of articles in clusters, excluding noise.",
    )

    class Settings:
        name = Settings().mongodb_clusterings_sessions_collection  # type:ignore


class Cluster(Document):
    session: Link[ClusteringSession]
    articles_count: int = Field(
        ...,
        description="Number of articles in the cluster.",
    )
    articles_ids: List[PydanticObjectId] = Field(
        ...,
        description="IDs of articles in the cluster, sorted by their distance to the cluster center",
    )

    title: str | None = Field(
        default=None, description="AI generated title of the cluster"
    )
    summary: str | None = Field(
        default=None, description="AI generated summary of the cluster"
    )

    overview_generation_error: str | None = Field(
        default=None, description="Error message if the overview generation failed"
    )

    class Settings:
        name = Settings().mongodb_clusters_collection  # type:ignore
