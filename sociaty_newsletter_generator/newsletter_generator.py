import asyncio
from datetime import date, datetime

from langchain import hub
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
)
from langsmith import traceable
from pydantic import BaseModel, Field, HttpUrl

from sociaty_newsletter_generator.db import init_db
from sociaty_newsletter_generator.models import (
    Cluster,
    ClusteringSession,
    SetOfUniqueArticles,
)
from sociaty_newsletter_generator.services import (
    ArticlesSummarizerChain,
    Language,
    create_articles_summarizer_chain,
)


class _TopicMaterial(BaseModel):
    comprehensive_summary: str
    url: HttpUrl
    image_url: HttpUrl | None

    def format(self) -> str:
        return f"{self.comprehensive_summary}\nSource URL : {self.url}\nImage URL : {self.image_url}"


class NewsletterGenerationRequest(BaseModel):
    template: str = Field(
        ..., description="Template showing the base structure of the newsletter"
    )
    clustering_session: ClusteringSession = Field(
        ...,
        description="The session that detected the clusters of articles (thus the topics) that will be used to generate the newsletter",
    )
    nb_main_topics: int = 5
    nb_secondary_topics: int = 3


def create_cluster_to_topic_chain(
    articles_summarizer: ArticlesSummarizerChain,
) -> Runnable[Cluster, _TopicMaterial]:
    def get_first_image_url(articles: SetOfUniqueArticles) -> HttpUrl | None:
        return next((article.image for article in articles if article.image), None)

    return (
        RunnableLambda(Cluster.get_articles)
        | RunnableLambda(SetOfUniqueArticles)
        | RunnableParallel(
            comprehensive_summary=articles_summarizer,
            url=RunnableLambda(lambda x: x[0].url),  # type:ignore # Cannot access attribute "url" for class "list[Article]" Attribute "url" is unknown
            image_url=RunnableLambda(
                get_first_image_url
            ),  # TODO : move this before the SetOfUniqueArticle, because we might loose information during the deduplication process
        )
        | RunnableLambda(_TopicMaterial.model_validate)
    )


sociaty_newsletter_template = """# SocIAty Intelligence - <date>

*Votre source d'insights sur l'intelligence artificielle*

<introduction>

---

## 🚀 L'IA en Action
<!-- Commentaire : cette section contient les sujets principaux. Il y a deux sujets à titre d'exemple, mais ça peut être plus. -->

### <sujet1_titre>
![sujet1_image](sujet1_image_url) <!-- Remplacer par l'URL de l'image -->

**<sujet1_resumé>[Lorem ipsum dolor sit amet](https://lien_vers_la_source_officielle), consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. </sujet1_résumé>**

<sujet1_détails>

- Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo.
- Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem.
- Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur?
- Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur?

</sujet1_détails>

### <sujet2_titre>
![sujet2_image](sujet1_image_url) <!-- Remplacer par l'URL de l'image -->

**<sujet2_resumé>[Lorem ipsum dolor sit amet](https://lien_vers_la_source_officielle2), consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. </sujet2_résumé>**

<sujet2_détails>

- Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo.
- Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem.
- Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur?
- Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur?

</sujet2_détails>

<!-- Commentaire : S'il y en a, insérer ici les autres sujets principaux, sous le même format-->


## 📰 Actualités et Tendances
<!-- Commentaire : cette section contient les sujets secondaires. Il y a trois sujets à titre d'exemple, mais ça peut être plus. -->

**XXX** a [verbe](https://lien_vers_la_source_officielle) ipsum dolor sit amet, consectetur adipiscing elit. Etiam venenatis felis id ligula porttitor sollicitudin. Nulla facilisi. Morbi commodo arcu eu varius condimentum. Aenean felis eros, pharetra quis mi eu, sagittis dapibus ante. 

**YYY** [verbe](https://lien_vers_la_source_officielle) Duis finibus dolor elementum ex rhoncus interdum. Aliquam tincidunt rhoncus bibendum. Fusce scelerisque est vel tempus volutpat. Sed nec auctor est, in sollicitudin ante. Nunc viverra, quam in pharetra gravida, orci tellus auctor elit, et tincidunt turpis augue in nisi. Integer ac nulla id urna pulvinar imperdiet.

**ZZZ** [verbe](https:///lien_vers_la_source_officielle)  usce tempus faucibus vestibulum. Phasellus sem mi, facilisis pulvinar nisl a, consectetur posuere turpis. Curabitur commodo ante eu sapien tristique, sit amet ultricies diam convallis. Nulla nec ligula consectetur, imperdiet ipsum non, rutrum urna. Vivamus imperdiet sagittis mi vel dapibus.

</autres_sujets_secondaires>

---

<phrase_de_conclusion>

https://www.linkedin.com/company/sociaty-io/

Nos offres https://sociaty.io/#offres

Contactez-nous https://sociaty.io/#projet
"""


@traceable
async def generate_newsletter(
    session: ClusteringSession, llm: BaseChatModel, language: Language = "fr"
) -> str:
    clusters = await session.get_included_sorted_clusters()

    articles_summarizer = create_articles_summarizer_chain(language="fr")
    cluster_to_topic_chain = create_cluster_to_topic_chain(articles_summarizer)

    topics = await cluster_to_topic_chain.abatch(clusters[:10])

    main_topics_str = "\n\n".join(topic.format() for topic in topics[:5])
    secondary_topics_str = "\n\n".join(topic.format() for topic in topics[5:10])

    prompt = hub.pull("generate_newsletter")

    chain = prompt | llm | StrOutputParser()

    newsletter = chain.invoke(
        {
            "newsletter_template": sociaty_newsletter_template,
            "main_topics": main_topics_str,
            "secondary_topics": secondary_topics_str,
            "language": language,
            "writing_date": date.today().strftime("%d %B %Y"),
            "data_start": session.data_start.strftime("%d %B %Y"),
            "data_end": session.data_end.strftime("%d %B %Y"),
        }
    )
    return newsletter


async def main():
    await init_db()

    llm = init_chat_model("gpt-4o", temperature=0)

    output_folder = "output"

    for session in await ClusteringSession.find_all().to_list():
        data_start, data_end = session.data_start, session.data_end

        print(f"Session : {session.id}")
        print(
            f"- From {data_start.strftime('%d %B %Y')} to {data_end.strftime('%d %B %Y')}"
        )

        newsletter = await generate_newsletter(session=session, llm=llm, language="fr")

        file_name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{session.id}.md"
        print(newsletter)

        with open(f"{output_folder}/{file_name}", "w", encoding="utf-8") as f:
            f.write(newsletter)

        print(f"Newsletter saved at {output_folder}/{file_name}")


if __name__ == "__main__":
    asyncio.run(main())
