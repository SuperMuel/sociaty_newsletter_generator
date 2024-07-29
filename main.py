import asyncio

from sociaty_newsletter_generator.db import init_db
from sociaty_newsletter_generator.models import Cluster


async def main():
    await init_db()
    print(await Cluster.find_one())


if __name__ == "__main__":
    asyncio.run(main())
