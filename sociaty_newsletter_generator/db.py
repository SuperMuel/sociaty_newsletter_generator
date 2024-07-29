from sociaty_newsletter_generator.models import Cluster, ClusteringSession
from sociaty_newsletter_generator.settings import Settings
import motor.motor_asyncio
from beanie import init_beanie


async def init_db():
    settings = Settings()  # type:ignore
    client = motor.motor_asyncio.AsyncIOMotorClient(settings.mongodb_uri)
    database = client[settings.mongodb_database]
    await init_beanie(database, document_models=[Cluster, ClusteringSession])
