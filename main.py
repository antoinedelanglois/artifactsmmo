import asyncio
import logging
from logging.handlers import RotatingFileHandler
from aiohttp import ClientSession
from character import Character
from api import (get_bank_item_qty, get_all_maps, get_all_items, get_all_monsters, get_all_resources, get_all_status,
                 get_all_items_quantities, get_all_bank_details)
from models import Environment, Task, TaskType


async def run_bot(character_object: Character):
    while True:

        character_infos = await character_object.get_infos()

        await character_object.deposit_items_at_bank()

        await character_object.manage_task(character_object.session)

        # Check if game task is feasible, assign if it is / necessarily existing
        event_task = await character_object.get_event_task()
        # event_task = None
        game_task = await character_object.get_game_task()
        recycling_task = await character_object.get_recycling_task()
        craft_for_equipping_task = await character_object.get_craft_for_equipping_task()
        fight_for_leveling_up_task = await character_object.get_fight_for_leveling_up_task()
        if event_task.type != TaskType.IDLE:
            character_object.task = event_task
        # No need to do game tasks if already a lot of task coins
        elif ((game_task.is_feasible(character_infos, character_object.max_fight_level)
              and (await get_bank_item_qty(character_object.session, "tasks_coin") < 100))
              or len(character_object.objectives) == 0):
            character_object.task = game_task
        elif recycling_task.type != TaskType.IDLE:
            character_object.task = recycling_task
        # TODO get a task of leveling up on gathering if craftable items without autonomy
        elif craft_for_equipping_task.type != TaskType.IDLE:
            character_object.task = craft_for_equipping_task
        elif fight_for_leveling_up_task.type != TaskType.IDLE and await character_infos.got_enough_consumables(1):
            character_object.task = fight_for_leveling_up_task
        elif character_object.task.type == TaskType.IDLE:
            # find and assign a valid task
            character_object.task = await character_object.get_task()     # From a list?

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
            # Parallelization of initial API calls
            items, monsters, resources_data, maps_data, status, bank_details = await asyncio.gather(*[
                asyncio.create_task(get_all_items(session)),
                asyncio.create_task(get_all_monsters(session)),
                asyncio.create_task(get_all_resources(session)),
                asyncio.create_task(get_all_maps(session)),
                asyncio.create_task(get_all_status(session)),
                asyncio.create_task(get_all_bank_details(session))
            ])

            environment = Environment(
                items=items,
                monsters=monsters,
                resource_locations=resources_data,
                maps=maps_data,
                status=status,
                bank_details=bank_details
            )

            all_items_quantities = await get_all_items_quantities(session)
            obsolete_equipments = environment.get_obsolete_equipments(all_items_quantities)

            # LOCAL_BANK = await get_bank_items(session)

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
                    max_fight_level=30,
                    skills=['weaponcrafting', 'cooking', 'mining', 'woodcutting']
                ),  # 'weaponcrafting', 'mining', 'woodcutting'
                Character(
                    session=session,
                    environment=environment,
                    obsolete_equipments=obsolete_equipments,
                    name='Capu',
                    max_fight_level=30,
                    skills=['gearcrafting', 'woodcutting', 'mining']
                ),  # 'gearcrafting',
                Character(
                    session=session,
                    environment=environment,
                    obsolete_equipments=obsolete_equipments,
                    name='Brubu',
                    max_fight_level=30,
                    skills=['cooking', 'woodcutting', 'mining']
                ),  # , 'fishing', 'mining', 'woodcutting'
                Character(
                    session=session,
                    environment=environment,
                    obsolete_equipments=obsolete_equipments,
                    name='Crabex',
                    max_fight_level=30,
                    skills=['jewelrycrafting', 'mining', 'woodcutting']
                ),  # 'jewelrycrafting', 'woodcutting', 'mining'
                Character(
                    session=session,
                    environment=environment,
                    obsolete_equipments=obsolete_equipments,
                    name='JeaGa',
                    max_fight_level=30,
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
