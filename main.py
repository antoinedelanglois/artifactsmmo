import asyncio
import logging
from logging.handlers import RotatingFileHandler
from aiohttp import ClientSession
from character import Character
from api import (get_all_maps, get_all_items, get_all_monsters, get_all_resources, get_all_status,
                 get_all_items_quantities, get_all_bank_details)
from models import Environment, Task


async def run_bot(character_object: Character):
    while True:

        await character_object.deposit_items_at_bank()

        await character_object.manage_task()

        priority_target_code = ""

        if character_object.name == "Brubu":
            priority_target_code = "cultist_acolyte"

        await character_object.set_initial_task(priority_target_code)

        await character_object.execute_task()

        # Reinitialize task
        character_object.task = Task()
        await character_object.initialize()


# class Market:
#     last_update: int
#     prices: List[Any]
#     api: ArtifactAPI
#
#     def __init__(self, api: ArtifactAPI):
#         self.api = api
#         self.prices = []
#         self.last_update = 0  # Initialize last_update with 0
#
#     def get_prices(self):
#         self.refresh()
#
#         items = [MarketItem(i) for i in self.prices]
#         return items
#
#     def get_by_code(self, code):
#         self.refresh()
#
#         for i in self.prices:
#             if i['code'] == code:
#                 return MarketItem(i)
#
#     def update_price(self, data):
#         for i, item  in enumerate(self.prices.copy()):
#             if item['code'] == data['code']:
#                 self.prices[i] = data
#
#     def refresh(self):
#         current_time = time.time()
#
#         # Update only if 60 seconds have passed since last update
#         if current_time - self.last_update < 60:
#             return
#
#         prices_page_1 = self.api.get_ge({'page': 1, 'size': 100})
#         prices_page_2 = self.api.get_ge({'page': 2, 'size': 100})
#
#         self.prices = prices_page_1 + prices_page_2
#         self.last_update = current_time  # Update the timestamp


# FIXME consumable should be of the correct level to be equipped
# FIXME if there is a task implying fight, make it priority

async def main():

    try:
        async with ClientSession() as session:

            async def safe_request(coro):
                try:
                    return await coro
                except Exception as error:
                    logging.error(f"Error in task: {str(error)}")
                    return None  # or handle it differently

        # Parallelization of initial API calls
            items, monsters, resources_data, maps_data, status, bank_details, items_quantities = await asyncio.gather(*[
                safe_request(get_all_items(session)),
                safe_request(get_all_monsters(session)),
                safe_request(get_all_resources(session)),
                safe_request(get_all_maps(session)),
                safe_request(get_all_status(session)),
                safe_request(get_all_bank_details(session)),
                safe_request(get_all_items_quantities(session))
            ])

            environment = Environment(
                items=items,
                monsters=monsters,
                resource_locations=resources_data,
                maps=maps_data,
                status=status,
                bank_details=bank_details
            )

            obsolete_equipments = environment.get_obsolete_equipments(items_quantities)

            # LOCAL_BANK = await get_bank_item_codes2qty(session)

            # Lich 96% > cursed_specter, gold_shield, cursed_hat, malefic_armor, piggy_pants, gold_boots, ruby_ring,
            # ruby_ring, ruby_amulet
            # Lich 100% > cursed_specter, gold_shield, cursed_hat, malefic_armor, piggy_pants, gold_boots, ruby_ring,
            # ruby_ring, magic_stone_amulet
            # Lich > gold_sword, gold_shield, lich_crown, obsidian_armor, gold_platelegs, lizard_boots, dreadful_ring,
            # topaz_ring, topaz_amulet

            characters_ = [
                Character(
                    session=session,
                    environment=environment,
                    obsolete_equipments=obsolete_equipments,
                    name='Kersh',
                    max_fight_level=33,
                    skills=['weaponcrafting', 'cooking', 'mining', 'woodcutting']
                ),  # 'weaponcrafting', 'mining', 'woodcutting'
                Character(
                    session=session,
                    environment=environment,
                    obsolete_equipments=obsolete_equipments,
                    name='Capu',
                    max_fight_level=33,
                    skills=['gearcrafting', 'woodcutting', 'mining']
                ),  # 'gearcrafting',
                Character(
                    session=session,
                    environment=environment,
                    obsolete_equipments=obsolete_equipments,
                    name='Brubu',
                    max_fight_level=33,
                    skills=['cooking', 'woodcutting', 'mining']
                ),  # , 'fishing', 'mining', 'woodcutting'
                Character(
                    session=session,
                    environment=environment,
                    obsolete_equipments=obsolete_equipments,
                    name='Crabex',
                    max_fight_level=33,
                    skills=['jewelrycrafting', 'mining', 'woodcutting']
                ),  # 'jewelrycrafting', 'woodcutting', 'mining'
                Character(
                    session=session,
                    environment=environment,
                    obsolete_equipments=obsolete_equipments,
                    name='JeaGa',
                    max_fight_level=33,
                    skills=['fishing', 'mining', 'woodcutting']
                ),  # 'cooking', 'fishing'
            ]

            # Initialize all characters asynchronously
            await asyncio.gather(*[character.initialize() for character in characters_])

            # Add stocking objectives to characters depending on their crafting objectives

            # FIXME Define a stock Task, where the gathering / crafting is made until bank stock is reached

            # Start the bot for all characters
            await asyncio.gather(*[run_bot(character) for character in characters_])

    except Exception as e:
        logging.exception(f"An unexpected error occurred in the main function. Error: {e}")


if __name__ == '__main__':

    logging.basicConfig(
        force=True,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - [%(levelname)s] %(message)s",
        handlers=[
            RotatingFileHandler("logs/output.log", maxBytes=5*1024*1024, backupCount=5),
            logging.StreamHandler()
        ]
    )

    logging.info("Bot started.")
    asyncio.run(main())
    logging.info("Bot finished.")
