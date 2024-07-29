from datetime import UTC, date, datetime
from enum import Enum
from math import floor
from typing import Annotated, Any, Dict, Iterator, List, Tuple

from beanie import Document, Link, PydanticObjectId
from pydantic import Field, PastDatetime, UrlConstraints, field_validator
from pydantic_core import Url
from pymongo import IndexModel

from sociaty_newsletter_generator.settings import Settings

DateOrDatetime = date | datetime

HttpUrl = Annotated[
    Url,
    UrlConstraints(max_length=2083, allowed_schemes=["http", "https"]),
]


def utc_datetime_factory():
    return datetime.now(UTC)


class Region(str, Enum):
    ARABIA = "xa-ar"
    ARABIA_EN = "xa-en"
    ARGENTINA = "ar-es"
    AUSTRALIA = "au-en"
    AUSTRIA = "at-de"
    BELGIUM_FR = "be-fr"
    BELGIUM_NL = "be-nl"
    BRAZIL = "br-pt"
    BULGARIA = "bg-bg"
    CANADA = "ca-en"
    CANADA_FR = "ca-fr"
    CATALAN = "ct-ca"
    CHILE = "cl-es"
    CHINA = "cn-zh"
    COLOMBIA = "co-es"
    CROATIA = "hr-hr"
    CZECH_REPUBLIC = "cz-cs"
    DENMARK = "dk-da"
    ESTONIA = "ee-et"
    FINLAND = "fi-fi"
    FRANCE = "fr-fr"
    GERMANY = "de-de"
    GREECE = "gr-el"
    HONG_KONG = "hk-tzh"
    HUNGARY = "hu-hu"
    INDIA = "in-en"
    INDONESIA = "id-id"
    INDONESIA_EN = "id-en"
    IRELAND = "ie-en"
    ISRAEL = "il-he"
    ITALY = "it-it"
    JAPAN = "jp-jp"
    KOREA = "kr-kr"
    LATVIA = "lv-lv"
    LITHUANIA = "lt-lt"
    LATIN_AMERICA = "xl-es"
    MALAYSIA = "my-ms"
    MALAYSIA_EN = "my-en"
    MEXICO = "mx-es"
    NETHERLANDS = "nl-nl"
    NEW_ZEALAND = "nz-en"
    NORWAY = "no-no"
    PERU = "pe-es"
    PHILIPPINES = "ph-en"
    PHILIPPINES_TL = "ph-tl"
    POLAND = "pl-pl"
    PORTUGAL = "pt-pt"
    ROMANIA = "ro-ro"
    RUSSIA = "ru-ru"
    SINGAPORE = "sg-en"
    SLOVAK_REPUBLIC = "sk-sk"
    SLOVENIA = "sl-sl"
    SOUTH_AFRICA = "za-en"
    SPAIN = "es-es"
    SWEDEN = "se-sv"
    SWITZERLAND_DE = "ch-de"
    SWITZERLAND_FR = "ch-fr"
    SWITZERLAND_IT = "ch-it"
    TAIWAN = "tw-tzh"
    THAILAND = "th-th"
    TURKEY = "tr-tr"
    UKRAINE = "ua-uk"
    UNITED_KINGDOM = "uk-en"
    UNITED_STATES = "us-en"
    UNITED_STATES_ES = "ue-es"
    VENEZUELA = "ve-es"
    VIETNAM = "vn-vi"
    NO_REGION = "wt-wt"


class Article(Document):
    title: str = Field(..., min_length=1, max_length=200)
    url: HttpUrl
    body: str = Field(default="", max_length=1000)
    found_at: PastDatetime = Field(default_factory=utc_datetime_factory)
    date: PastDatetime
    region: Region | None = None
    image: HttpUrl | None = None
    source: str | None = Field(default=None, max_length=100)
    vector_indexed: bool = False

    @field_validator("title", mode="before")
    @classmethod
    def truncate_title(cls, v: str) -> str:
        return v[:200] if len(v) > 200 else v

    @field_validator("body", mode="before")
    @classmethod
    def truncate_body(cls, v: str) -> str:
        return v[:1000] if len(v) > 1000 else v

    class Settings:
        name = "ai_news"  # TODO : change the name
        indexes = [
            IndexModel("url", unique=True),
            IndexModel("vector_indexed"),
            IndexModel("date"),
        ]


DateOrDatetime = date | datetime


class SetOfUniqueArticles:
    """
    A class designed to reduce the amount of duplicated or similar articles passed to LLMs.

    Articles are considered the same if any of the following conditions are met:
    - They have the same ID
    - They have the same URL
    - They have the same title (case insensitive, ignoring leading/trailing whitespaces)
      AND the same date (ignoring the time part if it exists)

    If articles have the same title and date but different bodies, the article with the longer body is kept.
    """

    def __init__(self, input_data: Article | List[Article] | None = None):
        """
        Initialize the set of unique articles.

        Args:
            input_data (Article | List[Article], optional): An initial article or list of articles to add.
        """
        self.articles_by_id: Dict[str, Article] = {}
        self.articles_by_url: Dict[Url, Article] = {}
        self.articles_by_title_date: Dict[Tuple[str, date], Article] = {}

        if not input_data:
            return

        if isinstance(input_data, Article):
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

    def _normalize_date(self, d: datetime) -> date:
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

    def _generate_title_date_key(self, article: Article) -> Tuple[str, date]:
        """
        Generate a key for an article based on its normalized title and date.

        Args:
            article (Article): The article to generate the key for.

        Returns:
            Tuple[str, date]: The generated key.
        """
        normalized_title = self._normalize_string(article.title)
        normalized_date = self._normalize_date(article.date)
        return (normalized_title, normalized_date)

    def add_article(self, article: Article):
        """
        Add an article to the set of unique articles.

        Args:
            article (Article): The article to add.
        """
        # Check for ID uniqueness
        if article.id and str(article.id) in self.articles_by_id:
            return

        # Check for URL uniqueness
        if article.url in self.articles_by_url:
            return

        title_date_key = self._generate_title_date_key(article)
        normalized_body = self._normalize_string(article.body)

        if title_date_key not in self.articles_by_title_date:
            self._add_article_to_all_dicts(article, title_date_key)
            return

        existing_article = self.articles_by_title_date[title_date_key]
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
            self._remove_article_from_all_dicts(existing_article)
            self._add_article_to_all_dicts(article, title_date_key)

    def _add_article_to_all_dicts(
        self, article: Article, title_date_key: Tuple[str, date]
    ):
        """
        Add an article to all internal dictionaries.

        Args:
            article (Article): The article to add.
            title_date_key (Tuple[str, date]): The key for the title-date dictionary.
        """
        if article.id:
            self.articles_by_id[str(article.id)] = article
        self.articles_by_url[article.url] = article
        self.articles_by_title_date[title_date_key] = article

    def _remove_article_from_all_dicts(self, article: Article):
        """
        Remove an article from all internal dictionaries.

        Args:
            article (Article): The article to remove.
        """
        if article.id:
            self.articles_by_id.pop(str(article.id), None)
        self.articles_by_url.pop(article.url, None)
        title_date_key = self._generate_title_date_key(article)
        self.articles_by_title_date.pop(title_date_key, None)

    def get_articles(self) -> List[Article]:
        """
        Get the list of unique articles.

        Returns:
            List[Article]: The list of unique articles.
        """
        return list(self.articles_by_title_date.values())

    def __iter__(self) -> Iterator[Article]:
        """
        Return an iterator over the unique articles.

        Returns:
            Iterator[Article]: An iterator over the unique articles.
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
        return bool(self.articles_by_title_date)

    def limit(self, n: int) -> "SetOfUniqueArticles":
        """
        Limit the number of articles to be returned. Note: subsequent calls to `add_article` will not be limited.

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
        return len(self.articles_by_title_date)


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

    async def get_articles(self) -> List[Article]:
        return await Article.find_many({"_id": {"$in": self.articles_ids}}).to_list()
