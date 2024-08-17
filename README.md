# artifactsmmo
Bot automation game to practice asynchronous dispatch of tasks (and have fun while learning :p)


First content is run on notebook to be triggered and monitored easily through smartphone

Next step will be to get it hosted on NAS


As for now:

- Create an account on https://artifactsmmo.com/
- Paste notebook_content.py in a notebook (ex: google colab)
- Update your token

Then run 

async def main():
  async with aiohttp.ClientSession() as session:

    map_object = Map(session)

    await asyncio.gather(
      run_bot(session, 'Character_name1', 'mining', map_object),
      run_bot(session, 'Character_name2', 'woodcutting', map_object)
    )

asyncio.run(main())
