from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
from aiohttp import ClientSession
from models import Environment, Task, Item, Monster, TaskType, CharacterInfos
from constants import (STOCK_QTY_OBJECTIVE, EXCLUDED_MONSTERS, SERVER, SPAWN_COORDINATES, BANK_COORDINATES,
                       EQUIPMENTS_SLOTS, SLOT_TYPE_MAPPING)
from utils import select_best_equipment, select_best_equipment_set
from api import (get_bank_item_qty, make_request, get_place_name, get_all_maps, get_all_events,
                 get_all_items_quantities, needs_stock, get_all_map_item_qty, get_bank_items, get_all_infos,
                 get_character_move, get_character_exchange_tasks_coins, get_character_accept_new_task,
                 get_character_cancel_task, get_character_complete_task)
import asyncio
import logging


class Character(BaseModel):
    session: ClientSession
    environment: Environment
    obsolete_equipments: dict[str, Item]
    name: str
    skills: list[str]
    max_fight_level: int = 0
    stock_qty_objective: int = STOCK_QTY_OBJECTIVE
    task: Task = Field(default_factory=Task)
    gatherable_items: list[Item] = Field(default_factory=list)
    craftable_items: list[Item] = Field(default_factory=list)
    fightable_monsters: list[Monster] = Field(default_factory=list)
    fightable_materials: list[Item] = Field(default_factory=list)
    objectives: list[Item] = Field(default_factory=list)
    craft_objectives: list[Item] = Field(default_factory=list)
    gather_objectives: list[Item] = Field(default_factory=list)
    fight_objectives: list[Monster] = Field(default_factory=list)
    # infos: dict = Field(default_factory=dict)

    _logger: logging.Logger = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(**data)
        self._logger = logging.getLogger(self.name)

    async def initialize(self):
        """
        This method handles the async initialization of gatherable resources, craftable items,
        and fightable monsters. It should be called explicitly after character creation.
        """
        # TODO add also check on inventory and on task status?
        infos = await get_all_infos(self.session, self.name)
        await asyncio.gather(
            self.set_gatherable_items(infos),
            self.set_craftable_items(infos),
            self.set_fightable_monsters()
        )
        await self.set_objectives()

    async def get_infos(self) -> CharacterInfos:
        return await get_all_infos(self.session, self.name)

    async def set_gatherable_items(self, character_infos: CharacterInfos):
        """
        Fetch and set gatherable resources based on the character's collect skill and level
        """
        gatherable_items = [
            item
            for item in self.environment.items.values()
            if (item.is_gatherable()
                and item.level <= character_infos.get_skill_level(item.subtype)
                and self.environment.get_item_dropping_max_rate(item.code) <= 100)
        ]

        self.gatherable_items = gatherable_items
        self._logger.info(f"Gatherable items for {self.name}: {[r.code for r in self.gatherable_items]}")

    async def set_craftable_items(self, character_infos: CharacterInfos):
        """
        Fetch and set craftable items based on the character's craft skill and level
        """
        skill_craftable_items = self.environment.get_craftable_items(character_infos)
        craftable_items = skill_craftable_items[::-1]
        # TODO exclude protected items (such as the one using jasper_crystal)

        excluded_item_codes = []
        for item_code in Item.get_event_craft_items():
            for material_code in self.environment.items[item_code].get_craft_recipee():
                if material_code in Item.get_event_gather_items():
                    if await get_bank_item_qty(self.session, material_code) < self.stock_qty_objective:
                        excluded_item_codes.append(item_code)
                        continue

        filtered_craftable_items = [
            item
            for item in craftable_items
            if item.code not in excluded_item_codes
        ]
        self.craftable_items = filtered_craftable_items
        self._logger.info(f"Craftable items for {self.name}: {[i.code for i in self.craftable_items]}")

    async def set_fightable_monsters(self):
        """
        Fetch and set fightable monsters based on the character's maximum fight level
        """
        fightable_monsters = [
            monster
            for monster in self.environment.monsters.values()
            if monster.level <= self.max_fight_level    # TODO create method is_fightable
        ]
        # TODO make it dynamic based on real fight capacity
        self.fightable_monsters = [m for m in fightable_monsters if m.code not in EXCLUDED_MONSTERS]
        self.fightable_materials = [
            {item.code: item for item in self.environment.items.values() if item.type == "resource"}[drop.code]
            for monster in self.fightable_monsters
            for drop in monster.drops
            if drop.rate <= 50
        ]
        self._logger.debug(f"Fightable monsters for {self.name}: {[m.code for m in self.fightable_monsters]}")
        self._logger.debug(f"Fightable materials for {self.name}: {[m.code for m in self.fightable_materials]}")

    async def can_be_home_made(self, item: Item) -> bool:
        if item.is_gatherable():
            return item.code in [i.code for i in self.gatherable_items]
        elif item.is_dropped():
            return item.code in [i.code for i in self.fightable_materials]
        elif item.is_crafted():
            return all([
                await self.can_be_home_made(self.environment.items[material_code])
                for material_code in list(item.get_craft_recipee().keys())
            ])
        elif item.is_given():
            return False
        else:
            self._logger.warning(f' {item.code} to categorize')

    async def set_objectives(self):

        xp_item_codes = [
            item.code
            for item in self.craftable_items + self.gatherable_items
            if item.does_provide_xp(await self.get_infos(), self.environment.status.max_level)
        ]

        home_made_item_codes = [
            item.code
            for item in self.craftable_items + self.gatherable_items
            if await self.can_be_home_made(item)
        ]

        # Out of craftable items, which one can be handled autonomously
        craft_objectives = [
            item
            for item in self.craftable_items
            if item.code in home_made_item_codes and item.code in xp_item_codes
        ]

        too_high_level_items = [
            item
            for item in self.craftable_items
            if item.code not in home_made_item_codes and item.code in xp_item_codes
        ]
        self._logger.info(f' NEED LEVELING UP OR SPECIAL MATERIALS TO CRAFT: {[o.code for o in too_high_level_items]}')

        # Sort items by their rarity in the bank (to prioritize items that are rarer)
        items2bank_qty = {
            craftable_item.code: await get_bank_item_qty(self.session, craftable_item.code)
            for craftable_item in craft_objectives
        }

        craft_objectives = sorted(craft_objectives, key=lambda x: items2bank_qty.get(x.code, 0), reverse=False)

        gather_objectives = [
            resource
            for resource in self.gatherable_items
            if (resource.code in home_made_item_codes and resource.code in xp_item_codes
                and not resource.is_from_event())
        ][::-1]

        self._logger.info(f' GATHER OBJECTIVES: {[o.code for o in gather_objectives]}')

        # Filter according to defined craft skills
        self.craft_objectives = [
            item
            for item in craft_objectives
            if item.is_skill_compliant(self.skills)
        ]

        self.gather_objectives = [
            item
            for item in gather_objectives
            if item.is_skill_compliant(self.skills)
        ]

        self.fight_objectives = [
            monster
            for monster in self.fightable_monsters
            if (await self.can_be_vanquished(monster)
                and monster.does_provide_xp(await self.get_infos(),
                                            self.environment.status.max_level))
        ][::-1]

        self.objectives = self.craft_objectives + self.gather_objectives
        self._logger.info(f' CAN GET XP WITH: {[o.code for o in self.objectives]}')

    async def can_be_vanquished(self, monster: Monster) -> bool:
        return monster.code in [m.code for m in self.fightable_monsters]

    async def move_to_bank(self):
        bank_coords = await self.get_nearest_coords(
            content_type='bank',
            content_code='bank'
        )
        cooldown_ = await self.move(*bank_coords)
        await asyncio.sleep(cooldown_)

    async def deposit_items_at_bank(self, _items_details: dict[str, int] = None):
        infos = await self.get_infos()
        if infos.get_inventory_occupied_slots_nb() > 0:
            # Go to bank and deposit all objects
            # Move to the bank
            await self.move_to_bank()

            if _items_details is None:
                _items_details = infos.get_inventory_items()
                gold_amount = infos.get_gold_amount()
                if gold_amount > 0:
                    _items_details['money'] = gold_amount

            self._logger.debug(f'depositing at bank: {_items_details} ...')
            for item_code, item_qty in _items_details.items():
                cooldown_ = await self.bank_deposit(item_code, item_qty)
                await asyncio.sleep(cooldown_)

    async def withdraw_items_from_bank(self, _items_details: dict[str, int]):
        # Move to the bank
        await self.move_to_bank()

        self._logger.debug(f'collecting at bank: {_items_details} ...')

        infos = await self.get_infos()
        nb_free_slots = infos.get_inventory_free_slots_nb()
        if sum([qty for qty in _items_details.values()]) > nb_free_slots:
            self._logger.error(f'Not enough room left in inventory')

        for item_code, item_qty in _items_details.items():
            cooldown_ = await self.bank_withdraw(item_code, item_qty)
            await asyncio.sleep(cooldown_)

    async def gather_material(self, material_code: str, quantity: int):

        resource_location = self.environment.get_item_dropping_location(material_code)
        if not resource_location:
            return
        location_coords = await self.get_nearest_coords('resource', resource_location.code)

        cooldown_ = await self.move(*location_coords)
        await asyncio.sleep(cooldown_)

        self._logger.info(f'gathering {quantity} {material_code} ...')
        while await self.is_up_to_gather(material_code, quantity):
            cooldown_ = await self.perform_gathering()
            await asyncio.sleep(cooldown_)

    async def is_up_to_gather(self, _material_code: str, target_qty: int):
        infos = await self.get_infos()
        inventory_items = infos.get_inventory_items()
        gathered_qty = inventory_items.get(_material_code, 0)
        return not infos.is_inventory_full() and gathered_qty < target_qty

    async def is_up_to_fight(self, is_event: bool = False) -> bool:
        (infos, is_event_still_on,
         is_at_spawn_place, is_task_completed) = await asyncio.gather(
            get_all_infos(self.session, self.name),
            self.is_event_still_on(),
            self.is_at_spawn_place(),
            self.is_task_completed()
        )
        got_enough_consumables = infos.got_enough_consumables(-1)
        is_inventory_full = infos.is_inventory_full()
        if is_event:
            up_to_fight = got_enough_consumables and is_event_still_on and not (is_inventory_full or is_at_spawn_place)
        else:
            up_to_fight = got_enough_consumables and not (is_inventory_full or is_task_completed or is_at_spawn_place)
        self._logger.debug(f' up to fight? {up_to_fight}: consumables: {got_enough_consumables}'
                           f' is_inventory_full: {is_inventory_full} / is_event_still_on: {is_event_still_on} / '
                           f'is_task_completed: {is_task_completed} / is_at_spawn_place: {is_at_spawn_place}')
        return up_to_fight

    async def is_at_spawn_place(self) -> bool:
        infos = await self.get_infos()
        current_location = infos.get_current_location()
        if current_location == SPAWN_COORDINATES:
            self._logger.debug(f'is already at spawn place - likely killed by a monster')
            return True
        return False

    async def go_and_fight_to_collect(self, material_code: str, quantity_to_get: int):
        # Identify the monster and go to its location if fightable
        if not await self.is_fightable(material_code):
            return

        monster = self.environment.get_item_dropping_monster(material_code)
        if not monster:
            return
        await self.move_to_monster(monster.code)

        infos = await self.get_infos()
        while not infos.is_inventory_full() and infos.get_inventory_item_quantity(material_code) < quantity_to_get:
            infos = await self.get_infos()
            cooldown_ = await self.perform_fighting()
            await asyncio.sleep(cooldown_)

    async def complete_task(self, session: ClientSession, name: str):
        data = await get_character_complete_task(self.session, self.name)
        if data:
            _cooldown = data["data"]["cooldown"]["total_seconds"]
            await asyncio.sleep(_cooldown)
        else:
            self._logger.error(f'failed to complete task.')

    async def cancel_task(self):
        data = await get_character_cancel_task(self.session, self.name)
        if data:
            _cooldown = data["data"]["cooldown"]["total_seconds"]
            await asyncio.sleep(_cooldown)
        else:
            self._logger.error(f'failed to cancel task.')

    async def accept_new_task(self):
        data = await get_character_accept_new_task(self.session, self.name)
        if data:
            _cooldown = data["data"]["cooldown"]["total_seconds"]
            await asyncio.sleep(_cooldown)
        else:
            self._logger.error(f'failed to get new task.')

    async def exchange_tasks_coins(self):
        data = await get_character_exchange_tasks_coins(self.session, self.name)
        if data:
            _cooldown = data["data"]["cooldown"]["total_seconds"]
            await asyncio.sleep(_cooldown)
        else:
            self._logger.error(f'failed to get new task.')

    async def move(self, x, y) -> int:
        infos = await self.get_infos()
        current_location = infos.get_current_location()
        if current_location == (x, y):
            self._logger.debug(f'is already at the location ({x}, {y})')
            return 0

        data = await get_character_move(self.session, self.name, x, y)
        if data:
            _cooldown = data["data"]["cooldown"]["total_seconds"]
            self._logger.debug(f'moved to ({await get_place_name(self.session, x, y)}). Cooldown: {_cooldown} seconds')
            return _cooldown
        else:
            self._logger.error(f'failed to move to ({x}, {y})')
            return 0

    async def get_nearest_coords(self, content_type: str, content_code: str) -> tuple[int, int]:

        infos = await self.get_infos()

        if content_type == 'workshop' and content_code == 'fishing':
            content_code = 'cooking'
        resource_locations = await get_all_maps(
            session=self.session,
            params={
                'content_type': content_type,
                'content_code': content_code
            }
        )
        nearest_resource = {'x': BANK_COORDINATES[0], 'y': BANK_COORDINATES[1]}     # Default to bank
        if len(resource_locations) == 0:
            self._logger.warning(f'No resource {content_code} on this map')
            return nearest_resource['x'], nearest_resource['y']

        if len(resource_locations) == 1:
            return int(resource_locations[0]['x']), int(resource_locations[0]['y'])

        min_dist = 999999
        for resource_loc in resource_locations:
            res_x, res_y = int(resource_loc['x']), int(resource_loc['y'])
            character_location_x, character_location_y = infos.get_current_location()
            dist_to_loc = (res_x - character_location_x) ** 2 + (res_y - character_location_y) ** 2
            if dist_to_loc < min_dist:
                min_dist = dist_to_loc
                nearest_resource = resource_loc
        return nearest_resource['x'], nearest_resource['y']

    async def get_nb_craftable_items(self, _item: Item, from_inventory: bool = False) -> int:

        craft_recipee = _item.get_craft_recipee()
        self._logger.debug(f' recipee for {_item.code} is {craft_recipee}')

        infos = await self.get_infos()

        nb_craftable_items = _item.get_max_taskable_quantity(infos.get_inventory_max_size())

        if from_inventory:  # Taking into account drops > update qty available in inventory
            for material_code, qty in craft_recipee.items():
                material_inventory_qty = infos.get_inventory_item_quantity(material_code)
                nb_craftable_items = min(material_inventory_qty//qty, nb_craftable_items)

        self._logger.debug(f' nb of craftable items {"from inventory" if from_inventory else ""} '
                           f'for {_item.code} is {nb_craftable_items}')

        return nb_craftable_items

    async def bank_deposit(self, item_code: str, quantity: int) -> int:
        url = f"{SERVER}/my/{self.name}/action/bank/deposit"
        if item_code == 'money':
            url += '/gold'
            payload = {
                "quantity": quantity
            }
        else:
            payload = {
                "code": item_code,
                "quantity": quantity
            }
        data = await make_request(session=self.session, method='POST', url=url, payload=payload)
        if data:
            _cooldown = data["data"]["cooldown"]["total_seconds"]
            self._logger.debug(f'{quantity} {item_code} deposited. Cooldown: {_cooldown} seconds')
            return _cooldown
        else:
            self._logger.error(f'Failed to deposit {quantity} {item_code}')
            return 0

    async def bank_withdraw(self, item_code: str, quantity: int) -> int:
        url = f"{SERVER}/my/{self.name}/action/bank/withdraw"
        payload = {
            "code": item_code,
            "quantity": quantity
        }
        data = await make_request(session=self.session, method='POST', url=url, payload=payload)
        if data:
            _cooldown = data["data"]["cooldown"]["total_seconds"]
            self._logger.debug(f'{quantity} {item_code} withdrawn. Cooldown: {_cooldown} seconds')
            return _cooldown
        else:
            self._logger.error(f'Failed to withdraw {quantity} {item_code}')
            return 0

    async def perform_crafting(self, item_code: str, qte: int) -> int:
        url = f"{SERVER}/my/{self.name}/action/crafting"
        payload = {
            "code": item_code,
            "quantity": qte
        }
        data = await make_request(session=self.session, method='POST', url=url, payload=payload)
        if data:
            _cooldown = data["data"]["cooldown"]["total_seconds"]
            gained_xp = data["data"]["details"]["xp"]
            self._logger.info(f'{qte} {item_code} crafted. XP gained: {gained_xp}. Cooldown: {_cooldown} seconds')
            return _cooldown
        else:
            self._logger.error(f'Failed to craft {qte} {item_code}')
            return 0

    async def perform_fighting(self) -> int:
        url = f"{SERVER}/my/{self.name}/action/fight"
        data = await make_request(session=self.session, method='POST', url=url)
        if data:
            _cooldown = data["data"]["cooldown"]["total_seconds"]
            return _cooldown
        else:
            self._logger.error(f'failed to perform fighting action.')
            return 0

    async def perform_gathering(self) -> int:
        url = f"{SERVER}/my/{self.name}/action/gathering"
        data = await make_request(session=self.session, method='POST', url=url)
        if data:
            _cooldown = data["data"]["cooldown"]["total_seconds"]
            return _cooldown
        else:
            self._logger.error(f'failed to gather resources.')
            return 0

    async def perform_recycling(self, _item: Item, qte: int) -> int:
        url = f"{SERVER}/my/{self.name}/action/recycling"
        payload = {
            "code": _item.code,
            "quantity": qte
        }

        # SECURITY CHECK ON RARE ITEMS
        if _item.has_protected_ingredients():
            self._logger.warning(f' Item {_item.code} is rare so better not to recycle it.')
            return 0

        data = await make_request(session=self.session, method='POST', url=url, payload=payload)
        if data:
            _cooldown = data["data"]["cooldown"]["total_seconds"]
            self._logger.info(f'{qte} {_item.code} recycled. Cooldown: {_cooldown} seconds')
            return _cooldown
        else:
            self._logger.error(f'failed to recycle {qte} {_item.code}.')
            return 0

    async def is_gatherable(self, resource_code) -> bool:
        return resource_code in [item.code for item in self.gatherable_items]

    async def is_fightable(self, material_code) -> bool:
        return material_code in [item.code for item in self.fightable_materials]

    async def is_craftable(self, item_code) -> bool:
        return item_code in [item.code for item in self.craftable_items]

    async def is_collectable_at_bank(self, item_code, quantity) -> bool:
        qty_at_bank = await get_bank_item_qty(self.session, item_code)
        return qty_at_bank > 3 * quantity

    async def move_to_workshop(self):
        # get the skill out of item
        skill_name = self.task.details.get_skill_name()
        coords = await self.get_nearest_coords(content_type='workshop', content_code=skill_name)
        self._logger.debug(f'{self.name} > moving to workshop at {coords}')
        cooldown_ = await self.move(*coords)
        await asyncio.sleep(cooldown_)

    async def move_to_monster(self, monster_code: str = ""):
        monster_code = self.task.code if monster_code == "" else monster_code
        if self.task.is_event:
            coords = (self.task.x, self.task.y)
        else:
            coords = await self.get_nearest_coords(content_type='monster', content_code=monster_code)
        self._logger.debug(f'{self.name} > moving to monster {monster_code} at {coords}')
        cooldown_ = await self.move(*coords)
        await asyncio.sleep(cooldown_)

    async def move_to_task_master(self):
        coords = await self.get_nearest_coords(content_type='tasks_master', content_code='monsters')
        self._logger.debug(f'{self.name} > moving to tasks master at {coords}')
        cooldown_ = await self.move(*coords)
        await asyncio.sleep(cooldown_)

    async def get_current_equipments(self, _items: dict[str, Item]) -> dict[str, Item]:
        infos = await self.get_infos()
        return {
            equipment_slot: _items[infos.get_slot_content(equipment_slot)]
            if infos.get_slot_content(equipment_slot) != "" else None
            for equipment_slot in EQUIPMENTS_SLOTS
        }

    async def go_and_equip(self, _equipment_slot: str, _equipment_code: str):
        infos = await self.get_infos()
        current_equipment_code = infos.get_equipment_code(_equipment_slot)
        if current_equipment_code != _equipment_code:
            self._logger.debug(f' will change equipment for {_equipment_slot} '
                               f'from {current_equipment_code} to {_equipment_code}')
            await self.withdraw_items_from_bank({_equipment_code: 1})
            if current_equipment_code != "":
                await self.unequip(_equipment_slot)
                await self.deposit_items_at_bank({current_equipment_code: 1})
            await self.equip(_equipment_code, _equipment_slot)

    async def equip_for_gathering(self, gathering_skill: str):
        """
        Equip the best available equipment for gathering based on the gathering skill.

        Args:
            gathering_skill (str): The gathering skill
        """

        infos = await self.get_infos()

        # Get the list of equipments from the bank for the 'weapon' slot
        bank_equipments = await self.get_bank_equipments_for_slot('weapon')

        # Filter equipments that have effects matching the gathering skill
        valid_equipments = []
        for equipment in bank_equipments:
            for effect in equipment.effects:
                if effect.name == gathering_skill:
                    valid_equipments.append(equipment)
                    break  # No need to check other effects

        if not valid_equipments:
            self._logger.info(f"No valid equipment found for gathering skill {gathering_skill}")
            return

        # Ensure equipments are valid for the character's level
        valid_equipments = [
            equipment for equipment in valid_equipments
            if equipment.is_valid_equipment(infos.get_level())
        ]

        if not valid_equipments:
            self._logger.info(f"No valid equipment found for gathering skill {gathering_skill} at character's level")
            return

        # Sort equipments based on the effect value for the gathering skill
        def get_effect_value(_equipment: Item, effect_name: str):
            for _effect in _equipment.effects:
                if _effect.name == effect_name:
                    return _effect.value
            return 0

        # Adjusted sorting: sort in ascending order since more negative is better
        valid_equipments.sort(
            key=lambda eq: get_effect_value(eq, gathering_skill),
            reverse=False
        )

        # Select the best equipment
        best_equipment = valid_equipments[0]
        self._logger.debug(f"Selected best equipment {best_equipment.code} for gathering skill {gathering_skill}")

        # Equip the best equipment
        await self.go_and_equip("weapon", best_equipment.code)

    async def gather_and_collect(self, _craft_details: dict[str, dict[str, int]]):
        fight_details = _craft_details['fight']
        if sum([qty for _, qty in fight_details.items()]) > 0:
            self._logger.debug(f'going to fight: {fight_details} ...')
            for item_code, qty in fight_details.items():
                monster = self.environment.get_item_dropping_monster(item_code)
                if not monster:
                    return
                await self.equip_for_fight(monster)
                await self.go_and_fight_to_collect(item_code, qty)
        for item_code, qty in _craft_details['gather'].items():
            await self.equip_for_gathering(self.environment.items[item_code].subtype)
            await self.gather_material(item_code, qty)
        collect_details = _craft_details['collect']
        if sum([qty for _, qty in collect_details.items()]) > 0:
            # need to deposit what's not useful for the crafting
            deposit_details = await self.identify_unnecessary_items(_craft_details)
            await self.deposit_items_at_bank(deposit_details)
            await self.withdraw_items_from_bank(collect_details)

    async def identify_unnecessary_items(self, craft_details: dict[str, dict[str, int]]) -> dict[str, int]:
        material2craft_qty = {}
        for _, materials_details in craft_details.items():
            for material_code, material_qty in materials_details.items():
                material2craft_qty[material_code] = material2craft_qty.get(material_code, 0) + material_qty

        material2deposit_qty = {}
        infos = await self.get_infos()
        for inventory_item_code, inventory_qty in infos.get_inventory_items().items():
            if inventory_item_code not in material2craft_qty:
                material2deposit_qty[inventory_item_code] = inventory_qty
            else:
                exceeding_qty = inventory_qty - material2craft_qty[inventory_item_code]
                if exceeding_qty > 0:
                    material2deposit_qty[inventory_item_code] = exceeding_qty

        return material2deposit_qty

    async def select_eligible_targets(self) -> list[str]:
        """
        Select eligible targets (tasks) for the character, ensuring that they will gain XP from them.
        Includes crafting, gathering, and fishing tasks. Returns a list of eligible items that provide XP.
        """
        self._logger.debug(f"Selecting eligible targets for {self.name}")

        # Filter items that are valid and provide XP (this includes craftable, gatherable, and fishable items)
        valid_craftable_items = self.gatherable_items

        # Log eligible items
        self._logger.debug(f'Eligible items for XP: {[item.code for item in valid_craftable_items]}')

        if not valid_craftable_items:
            self._logger.warning(f'No valid craftable/gatherable items found for {self.name} that provide XP.')
            return []

        # Sort items by their rarity in the bank (to prioritize items that are rarer)
        items2bank_qty = {
            craftable_item.code: await get_bank_item_qty(self.session, craftable_item.code)
            for craftable_item in valid_craftable_items
        }

        self.craftable_items = sorted(valid_craftable_items, key=lambda x: items2bank_qty.get(x.code, 0), reverse=False)

        return [item.code for item in self.craftable_items]

    async def equip_for_fight(self, _monster: Monster = None):

        if _monster is None:
            _monster = self.task.details

        # Identify vulnerability
        vulnerabilities = _monster.get_vulnerabilities()
        self._logger.debug(f' monster {_monster.code} vulnerabilities are {vulnerabilities}')

        current_equipments = await self.get_current_equipments(self.environment.items)
        sorted_valid_bank_equipments = await self.get_sorted_valid_bank_equipments()

        selected_equipments = await select_best_equipment_set(
            current_equipments,
            sorted_valid_bank_equipments,
            vulnerabilities
        )
        for equipment_slot, equipment in selected_equipments.items():
            if equipment is not None:
                await self.go_and_equip(equipment_slot, equipment.code)
            else:
                self._logger.debug(f"No equipment selected for slot {equipment_slot}")

        # Manage consumables
        await self.equip_best_consumables()

    async def get_eligible_bank_consumables(self) -> list[Item]:
        infos = await self.infos()
        return [
            consumable
            for consumable in self.environment.consumables.values()
            if await get_bank_item_qty(self.session, consumable.code) > 0 and consumable.level <= infos.get_level()
        ]

    async def get_2_best_consumables_including_equipped(self) -> list[Item]:
        """
        Fetches the two best consumables, including currently equipped ones, and ranks them.
        """
        # Fetch all consumables from the bank (TODO and inventory)
        valid_consumables = await self.get_eligible_bank_consumables()

        # Add the currently equipped consumables to the list of valid ones (if they are equipped)
        valid_consumables_codes = [c.code for c in valid_consumables]
        ordered_current_consumables = await self.get_ordered_current_consumables()
        for current_consumable in ordered_current_consumables:
            if current_consumable and current_consumable.code not in valid_consumables_codes:
                valid_consumables.append(current_consumable)

        valid_consumables = [
            consumable for consumable in valid_consumables
            if not consumable.is_protected_consumable()
        ]

        self._logger.debug(f' eligible consumables are {valid_consumables}')
        self._logger.debug(f' ordered current consumables are {ordered_current_consumables}')

        # Sort all consumables by level (higher is better)
        sorted_two_best_consumables = sorted(valid_consumables, key=lambda x: x['level'], reverse=True)[:2]
        self._logger.debug(f' two best consumables are {sorted_two_best_consumables}')

        # Initialize result as None placeholders for two consumables
        two_best_consumables = [None, None]

        # Single loop to determine the best consumables for each slot
        for consumable in sorted_two_best_consumables:
            if ordered_current_consumables[0] and consumable.code == ordered_current_consumables[0].code:
                # Keep in the same slot if it's already consumable1
                two_best_consumables[0] = consumable
            elif ordered_current_consumables[1] and consumable.code == ordered_current_consumables[1].code:
                # Keep in the same slot if it's already consumable2
                two_best_consumables[1] = consumable
            else:
                # Assign the consumable to the first available slot
                if two_best_consumables[0] is None:
                    two_best_consumables[0] = consumable
                elif two_best_consumables[1] is None:
                    two_best_consumables[1] = consumable

        # Ensure exactly 2 elements in the result, filling with None if necessary
        return two_best_consumables

    async def equip_best_consumables(self):
        """
        Equip the best available consumables in the consumable1 and consumable2 slots.
        This function avoids un-equipping if the consumable is already equipped and fully stocked.
        """
        # Fetch the two best consumables, ensuring they match the current slot order
        new_consumables = await self.get_2_best_consumables_including_equipped()

        for i, slot in enumerate(["consumable1", "consumable2"]):
            new_consumable = new_consumables[i]

            if new_consumable is None:
                self._logger.debug(f"No valid consumable available for {slot}. Skipping.")
                continue

            # Get current consumable details for the slot
            character_infos = await self.get_infos()
            current_code = character_infos.get_slot_content(slot)
            current_qty = character_infos.get_slot_quantity(slot)
            new_code = new_consumable.code

            # Check if the new consumable is the same as the current one
            if current_code == new_code:
                # Refill the consumable if not fully stocked
                if current_qty < 100:
                    additional_qty_needed = 100 - current_qty
                    bank_qty = await get_bank_item_qty(self.session, new_code)
                    withdraw_qty = min(additional_qty_needed, bank_qty)

                    if withdraw_qty > 0:
                        await self.withdraw_items_from_bank({new_code: withdraw_qty})
                        await self.equip(new_code, slot, withdraw_qty)
                else:
                    self._logger.debug(f"{slot} already equipped with {new_code} and fully stocked.")
            else:
                # Equip the new consumable if it's different from the current one
                if current_code:
                    await self.unequip(slot, current_qty)
                    await self.deposit_items_at_bank({current_code: current_qty})

                # Withdraw and equip the new consumable
                new_qty = min(100, await get_bank_item_qty(self.session, new_code))
                if new_qty > 0:
                    await self.withdraw_items_from_bank({new_code: new_qty})
                    await self.equip(new_code, slot, new_qty)

    async def get_ordered_current_consumables(self) -> list[Item]:
        character_infos = await self.get_infos()
        currently_equipped_consumables = [character_infos.consumable1_slot, character_infos.consumable2_slot]
        self._logger.debug(f' Currently equipped consumables: {currently_equipped_consumables}')
        return [self.environment.consumables[c] if c else None for c in currently_equipped_consumables]

    async def equip_best_equipment(self, _equipment_slot: str, vulnerabilities: dict[str, int]):
        infos = await self.get_infos()
        available_equipments = await self.get_bank_equipments_for_slot(_equipment_slot)
        self._logger.debug(f'available equipment at bank {[e.code for e in available_equipments]}')
        sorted_valid_equipments = sorted([
            equipment
            for equipment in available_equipments
            if equipment.is_valid_equipment(infos.get_level())
        ], key=lambda x: x.level, reverse=True)

        self._logger.debug(f'may be equipped with {[e.code for e in sorted_valid_equipments]}')

        current_equipment_code = infos.get_equipment_code(_equipment_slot)
        if len(sorted_valid_equipments) == 0:
            return
        current_equipment_infos = self.environment.equipments.get(current_equipment_code, {})
        new_equipment_details = await select_best_equipment(
            current_equipment_infos,
            sorted_valid_equipments,
            vulnerabilities
        )
        self._logger.debug(f' has been assigned {new_equipment_details.get("code", "")} '
                           f'for slot {_equipment_slot} instead of {current_equipment_infos.get("code", "")}')
        await self.go_and_equip(_equipment_slot, new_equipment_details.get('code', ""))

    async def perform_unequip(self, slot_code: str, qte: int) -> int:
        url = f"{SERVER}/my/{self.name}/action/unequip"
        payload = {
            "slot": slot_code,
            "quantity": qte
        }
        data = await make_request(session=self.session, method='POST', url=url, payload=payload)
        if data:
            _cooldown = data["data"]["cooldown"]["total_seconds"]
            self._logger.debug(f'{qte} {slot_code} unequipped. Cooldown: {_cooldown} seconds')
            return _cooldown
        else:
            self._logger.error(f'Failed to unequip {qte} {slot_code}')
            return 0

    async def perform_equip(self, item_code: str, slot_code: str, qte: int) -> int:
        url = f"{SERVER}/my/{self.name}/action/equip"
        payload = {
            "code": item_code,
            "slot": slot_code,
            "quantity": qte
        }
        data = await make_request(session=self.session, method='POST', url=url, payload=payload)
        if data:
            _cooldown = data["data"]["cooldown"]["total_seconds"]
            self._logger.debug(f'{qte} {item_code} equipped at slot {slot_code}. Cooldown: {_cooldown} seconds')
            return _cooldown
        else:
            self._logger.error(f'Failed to equip {qte} {item_code} at slot {slot_code}')
            return 0

    async def unequip(self, slot_code: str, qte: int = 1):
        cooldown_ = await self.perform_unequip(slot_code, qte)
        await asyncio.sleep(cooldown_)

    async def equip(self, item_code: str, slot_code: str, qte: int = 1):
        cooldown_ = await self.perform_equip(item_code, slot_code, qte)
        await asyncio.sleep(cooldown_)

    async def get_bank_equipments_for_slot(self, equipment_slot: str) -> list[Item]:
        """
        Retrieve a list of equipment items from the bank suitable for a specific equipment slot.
        Args:
            equipment_slot (str): The equipment slot to match (e.g., 'ring1', 'weapon').
        Returns:
            list[Item]: A list of equipment items that can be equipped in the specified slot.
        """
        bank_items = await get_bank_items(self.session)
        matching_equipments = []

        for item_code in bank_items.keys():
            equipment = self.environment.equipments.get(item_code)
            if equipment:
                # Check if the equipment type matches the slot (e.g., 'ring' in 'ring1')
                if equipment.type in equipment_slot:
                    matching_equipments.append(equipment)

        return matching_equipments

    async def get_sorted_valid_bank_equipments(self) -> dict[str, list[Item]]:
        """
        Retrieves a dictionary mapping equipment slots to sorted lists of valid equipment items from the bank.

        Returns:
            dict[str, list[Item]]: A dictionary where keys are equipment slots and values are lists of Items.
        """
        infos = await self.get_infos()
        bank_items = await get_bank_items(self.session)
        level = infos.get_level()

        # Create a mapping from equipment slots to lists of equipments
        bank_equipments = {slot: [] for slot in EQUIPMENTS_SLOTS}

        # Iterate over bank items and categorize them
        for item_code in bank_items.keys():
            equipment = self.environment.equipments.get(item_code)
            if equipment and equipment.is_valid_equipment(level):
                equipment_type = equipment.type
                slots = SLOT_TYPE_MAPPING.get(equipment_type, [])
                for slot in slots:
                    if slot in EQUIPMENTS_SLOTS:
                        bank_equipments[slot].append(equipment)

        # Sort equipments for each slot
        for slot in bank_equipments:
            bank_equipments[slot] = sorted(
                bank_equipments[slot],
                key=lambda x: x.level,
                reverse=True
            )

        return bank_equipments

    async def manage_task(self, session: ClientSession):
        game_task = await self.get_game_task()
        # if task completed (or none assigned yet), go to get rewards and renew task

        nb_tasks_coins = await get_bank_item_qty(session, "tasks_coin")
        nb_tasks_coins_lots = (nb_tasks_coins - 100)//6
        if nb_tasks_coins_lots > 0:
            await self.withdraw_items_from_bank({"tasks_coin": nb_tasks_coins_lots * 6})
            await self.move_to_task_master()

            for _ in range(nb_tasks_coins_lots):
                await self.exchange_tasks_coins()

        if await self.is_task_completed():
            # go to task master
            await self.move_to_task_master()
            # if task, get reward
            if game_task.code != "":
                await self.complete_task(session, self.name)
            # ask for new task
            await self.accept_new_task()
            game_task = await self.get_game_task()

        # If task is too difficult, change
        while game_task.code in EXCLUDED_MONSTERS:
            infos = await self.get_infos()
            if infos.get_inventory_item_quantity("tasks_coin") == 0:
                await self.withdraw_items_from_bank({"tasks_coin": 1})
            await self.move_to_task_master()
            await self.cancel_task()
            await self.accept_new_task()
            game_task = await self.get_game_task()

    async def is_task_completed(self) -> bool:

        # TODO can also be a personal task (amount of collectibles) - allow for some materials to be collected by others

        character_infos = await self.get_infos()
        return character_infos.task_progress == character_infos.task_total

    async def get_unnecessary_equipments(self) -> dict[str, int]:
        recycle_details = {}
        # Apply only on those that can be crafted again
        for item_code, item_qty in (await get_bank_items(self.session)).items():
            # No recycling for planks and ores and cooking
            item = self.environment.items[item_code]
            if item_qty == 0 or item.is_not_recyclable():
                continue
            min_qty = item.get_min_stock_qty()
            if item.is_crafted() and item_qty > min_qty:
                recycle_details[item_code] = item_qty - min_qty
        return recycle_details

    async def is_worth_selling(self, _item: Item) -> bool:
        item_gold_value = _item.get_sell_price()   # FIXME this could be moving // need to be up to date
        materials = [self.environment.items[material] for material in _item.get_craft_recipee().keys()]
        if any([material.is_protected() for material in materials]):
            return False
        gold_value_sorted_materials = sorted(materials, key=lambda x: x["ge"]["buy_price"], reverse=True)
        return item_gold_value > sum(gold_value_sorted_materials[:2])

    async def get_game_task(self) -> Task:
        infos = await self.get_infos()
        task = infos.task
        task_type = TaskType(infos.task_type)
        task_total = infos.task_total - infos.task_progress
        if task_type == TaskType.MONSTERS:
            task_details = self.environment.monsters[task]
        elif task_type == TaskType.ITEMS:
            task_details = self.environment.items.get(task, {})
        else:
            # FIXME define a DefaultTask?
            task = "iron"
            task_type = TaskType.ITEMS
            task_total = 66
            task_details = self.environment.items["iron"]
        return Task(
            code=task,
            type=task_type,
            total=task_total,
            details=task_details
        )

    async def get_best_objective(self):
        await self.set_objectives()
        if len(self.objectives) > 0:
            return self.objectives[0]
        return self.craftable_items[0]

    async def set_task(self, item: Item):
        infos = await self.get_infos()
        self.task = Task(
            code=item.code,
            type=item.get_task_type(),
            total=item.get_max_taskable_quantity(infos.get_inventory_max_size()),
            details=item
        )

    async def get_task(self) -> Task:

        objective = await self.get_best_objective()     # FIXME get it depending on potential XP gain
        # FIXME could be checked here amongst craftable items first, then gatherable ones

        infos = await self.get_infos()

        return Task(
            code=objective.code,
            type=objective.get_task_type(),
            total=objective.get_max_taskable_quantity(infos.get_inventory_max_size()),
            details=objective
        )

    async def prepare_for_task(self) -> dict[str, dict[str, int]]:
        gather_details, collect_details, fight_details = {}, {}, {}

        craft_recipee = self.task.details.get_craft_recipee()

        target_details = {
            k: v*self.task.total
            for k, v in craft_recipee.items()
        }

        for material_code, qty in target_details.items():

            # TODO qualify item: craftable? gatherable? fightable?
            self._logger.debug(f' Check material {material_code}')

            if await self.is_collectable_at_bank(material_code, qty):
                # FIXME Book the amount so no other character can take it beforehand?
                collect_details[material_code] = qty
                continue
            if await self.is_craftable(material_code):
                # Set material as craft target
                self._logger.info(f' Resetting task to {material_code}')
                # FIXME  Replace qty by full inventory capacity ?
                material = self.environment.items[material_code]
                await self.set_task(material)
                return await self.prepare_for_task()
            if await self.is_gatherable(material_code):
                gather_details[material_code] = qty
                self._logger.debug(f' Gatherable qty {qty}')
                continue
            if await self.is_fightable(material_code):
                fight_details[material_code] = qty
                self._logger.debug(f' Fightable for qty {qty}')
                continue
            self._logger.warning(f" Material {material_code} won't provide XP...")

        craft_details = {
            'gather': gather_details,
            'collect': collect_details,
            'fight': fight_details
        }
        self._logger.debug(f' {craft_details=}')

        return craft_details

    async def is_event_still_on(self):
        all_events = await get_all_events(self.session)
        all_events_codes = [event['map']['content']['code'] for event in all_events]
        if self.task.code in all_events_codes:
            self._logger.debug(f' Event {self.task.code} is in {all_events_codes}')
            return True
        return False

    async def execute_task(self):

        self._logger.info(f" Here is the task to be executed: "
                          f"{self.task.code} ({self.task.type.value}: {self.task.total})")

        infos = await self.get_infos()

        await self.equip_for_task()

        if self.task.type == TaskType.MONSTERS:
            await self.move_to_monster(self.task.code)
            self._logger.info(f'fighting {self.task.code} ...')
            while await self.is_up_to_fight(is_event=self.task.is_event):  # TODO add dropped material count
                # TODO decrement task total on each won combat
                cooldown_ = await self.perform_fighting()
                await asyncio.sleep(cooldown_)

        elif self.task.type == TaskType.RESOURCES:
            if self.task.is_event:
                cooldown_ = await self.move(self.task.x, self.task.y)
                await asyncio.sleep(cooldown_)

                self._logger.info(f'gathering {self.task.total} {self.task.code} ...')
                while await self.is_up_to_gather(self.task.code, self.task.total) and await self.is_event_still_on():
                    cooldown_ = await self.perform_gathering()
                    await asyncio.sleep(cooldown_)
            else:
                location_details = self.environment.get_item_dropping_location(self.task.code)
                if not location_details:
                    return
                location_coords = await self.get_nearest_coords('resource', location_details.code)

                cooldown_ = await self.move(*location_coords)
                await asyncio.sleep(cooldown_)

                self._logger.info(f'gathering {self.task.total} {self.task.code} ...')
                while await self.is_up_to_gather(self.task.code, self.task.total):
                    cooldown_ = await self.perform_gathering()
                    await asyncio.sleep(cooldown_)
            # TODO decrement task total on each target resource gathered (in inventory)

        elif self.task.type == TaskType.RECYCLE:
            await self.withdraw_items_from_bank({self.task.code: self.task.total})
            if infos.get_inventory_item_quantity(self.task.code) >= self.task.total:
                await self.move_to_workshop()
                cooldown = await self.perform_recycling(self.task.details, self.task.total)
                await asyncio.sleep(cooldown)

        elif self.task.type == TaskType.ITEMS:
            # if all available at bank -> pick it and go craft
            # nb_items_to_craft = await self.get_nb_craftable_items(self.task.code, from_inventory=True)
            nb_items_to_craft = await self.get_nb_craftable_items(self.task.details)
            craft_details = {
                material_code: material_unit_qty * min(self.task.total, nb_items_to_craft)
                # material_code: material_unit_qty * self.task.total
                for material_code, material_unit_qty in (self.task.details.get_craft_recipee()).items()
            }

            # Checking if all is available in inventory
            if all([
                infos.get_inventory_item_quantity(material_code) >= material_qty
                for material_code, material_qty in craft_details.items()
            ]):
                pass
            # Checking if all is available in bank
            elif all([
                await get_bank_item_qty(self.session, material_code) >= material_qty
                for material_code, material_qty in craft_details.items()
            ]):
                await self.deposit_items_at_bank()
                await self.withdraw_items_from_bank(craft_details)
            # Go and get it
            else:
                task_details = await self.prepare_for_task()
                await self.gather_and_collect(task_details)

            # Taking withdraw fails into account
            nb_items_to_craft = await self.get_nb_craftable_items(self.task.details, from_inventory=True)

            if nb_items_to_craft > 0:
                await self.move_to_workshop()
                cooldown_ = await self.perform_crafting(self.task.code, nb_items_to_craft)
                await asyncio.sleep(cooldown_)

    async def equip_for_task(self):

        if self.task.type == TaskType.MONSTERS:
            self._logger.debug("Equipping for fight")
            await self.equip_for_fight()
        elif self.task.type == TaskType.RESOURCES:
            self._logger.debug("Equipping for gathering")
            await self.equip_for_gathering(self.task.details.get_skill_name())

    async def get_recycling_task(self) -> Task:

        # TODO only one loop on bank equipment
        # Check if we need slots in bank first
        nb_occupied_bank_slots = len(await get_bank_items(self.session))
        if nb_occupied_bank_slots > self.environment.bank_details.slots - 5:

            for item_code in self.obsolete_equipments:
                qty_at_bank = await get_bank_item_qty(self.session, item_code)
                if qty_at_bank > 0:
                    # If yes, withdraw them and get to workshop to recycle, before getting back to bank to deposit all
                    infos = await self.get_infos()
                    nb_free_inventory_slots = infos.get_inventory_free_slots_nb()
                    recycling_qty = min(qty_at_bank, nb_free_inventory_slots // 2)  # Need room when recycling

                    # set recycling task
                    return Task(
                        code=item_code,
                        type=TaskType.RECYCLE,
                        total=recycling_qty,
                        details=self.environment.items[item_code]
                    )

            recycle_details = await self.get_unnecessary_equipments()
            for item_code, recycling_qty in recycle_details.items():
                infos = await self.get_infos()
                nb_free_inventory_slots = infos.get_inventory_free_slots_nb()
                recycling_qty = min(recycling_qty, nb_free_inventory_slots // 2)  # Need room when recycling

                # set recycling task
                return Task(
                    code=item_code,
                    type=TaskType.RECYCLE,
                    total=recycling_qty,
                    details=self.environment.items[item_code]
                )

        return Task()

    async def get_fight_for_leveling_up_task(self) -> Task:
        # If XP can be gained by fighting, go

        # Remove event monsters
        fight_objectives = [
            f
            for f in self.fight_objectives
            if not f.is_event()
        ]

        if len(fight_objectives) > 0:
            # FIXME check if monster is reachable (example: 'demon' only available during events)
            highest_fightable_monster = fight_objectives[0]
            return Task(
                code=highest_fightable_monster.code,
                type=TaskType.MONSTERS,
                total=99,   # FIXME when does it stop?
                details=highest_fightable_monster
            )
        return Task()

    async def get_craft_for_equipping_task(self) -> Task:
        total_quantities = await get_all_items_quantities(self.session)
        # Keeping only the not yet available useful equipment
        objectives_codes = [o.code for o in self.objectives if o.code not in self.obsolete_equipments]
        craftable_new_equipments = [
            i
            for i in self.craftable_items      # FIXME how is cooking handled?
            if i.code in objectives_codes and await needs_stock(self.session, i, total_quantities) and i.is_equipment()
        ]

        if len(craftable_new_equipments) > 0:
            self._logger.warning(f' New equipments to craft: {[e.code for e in craftable_new_equipments]}')
            equipment = craftable_new_equipments[0]
            equipment_qty = await get_all_map_item_qty(self.session, equipment)
            equipment_min_stock = equipment.get_min_stock_qty()
            self._logger.warning(f' Got {equipment_qty} {equipment.code} on map, need at least {equipment_min_stock}')
            return Task(
                code=equipment.code,
                type=TaskType.ITEMS,
                total=equipment_min_stock - equipment_qty,
                details=equipment
            )
        return Task()

    async def get_event_task(self) -> Task:

        infos = await self.get_infos()

        all_events = await get_all_events(self.session)
        # TODO we need prioritization of events
        eligible_tasks = []
        for event in all_events:
            if event["name"] in ["Bandit Camp", "Portal"]:
                monster_code = event["map"]["content"]["code"]
                monster_details = self.environment.monsters[monster_code]
                monster_task = Task(
                    code=monster_code,
                    type=TaskType.MONSTERS,
                    total=99,   # FIXME when does it stop?
                    details=monster_details,
                    x=event["map"]["x"],
                    y=event["map"]["y"],
                    is_event=True
                )
                eligible_tasks.append(monster_task)
            if event["name"] in ["Magic Apparition", "Strange Apparition"]:
                resource_code = event["map"]["content"]["code"]
                resource = self.environment.resource_locations[resource_code]
                if infos.get_skill_level(resource.skill) >= resource.level:
                    gathering_task = Task(
                        code=resource_code,
                        type=TaskType.RESOURCES,
                        total=99,   # FIXME when does it stop?
                        details=resource,
                        x=event["map"]["x"],
                        y=event["map"]["y"],
                        is_event=True
                    )
                    eligible_tasks.append(gathering_task)

        if len(eligible_tasks) > 0:
            return eligible_tasks[-1]

        return Task()
