import aiohttp
import asyncio
import logging
import nest_asyncio
from aiohttp.client_exceptions import ClientConnectorError
from dataclasses import dataclass, field
import re
from typing import Optional

nest_asyncio.apply()

logging.basicConfig(
    force=True,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

# Server url
SERVER = "https://api.artifactsmmo.com"
# Your account token (https://artifactsmmo.com/account)
TOKEN = ("XXX")
EXCLUDED_CONSUMABLES = ["beef_stew", "cooked_beef"]
EXCLUDED_MATERIALS = ["jasper_crystal"]     # type = "resource" + subtype = "task"
EXCLUDED_ITEMS_CODES = [
    'copper_ring', 'copper_dagger', 'sticky_sword', 'copper_armor', 'copper_helmet', 'copper_boots',
    'copper_legs_armor', 'wooden_shield', 'feather_coat'
]


def handle_incorrect_status_code(_status_code: int) -> str:
    match _status_code:
        case 404: msg = "Item not found"
        case 461: msg = "A transaction is already in progress with this item/your gold in your bank"
        case 462: msg = "Bank is full"
        case 478: msg = "Missing item or insufficient quantity in your inventory"
        case 485: msg = "The item is already equipped"
        case 486: msg = "An action is already in progress"
        case 490: msg = "Character already at destination"
        case 491: msg = "Slot is empty"
        case 493: msg = "The resource is too high-level for your character"
        case 496: msg = "Character level is not high enough"
        case 497: msg = "Your character's inventory is full"
        case 498: msg = "The character cannot be found on your account"
        case 499: msg = "Your character is in cooldown"
        case 598: msg = "Resource not found on this map"
        case _: msg = f"An error occured {_status_code}"
    return msg


async def make_request(session, method, url, params=None, payload=None, retries=3, timeout=120):
    """
    Helper function to make requests with retries, timeouts, and optional query parameters.

    Args:
        session (aiohttp.ClientSession): The aiohttp session to use for requests.
        method (str): HTTP method to use (e.g., 'GET', 'POST').
        url (str): The endpoint URL.
        params (dict, optional): Query parameters to pass with the request.
        payload (dict, optional): JSON payload for POST/PUT requests.
        retries (int, optional): Number of retry attempts in case of failure.
        timeout (int, optional): Timeout for the request in seconds.

    Returns:
        dict: The JSON response data, or None if the request fails after retries.
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {TOKEN}"  # Assumes you have a TOKEN variable
    }

    for attempt in range(retries):
        try:
            async with session.request(method, url, headers=headers, params=params, json=payload, timeout=timeout) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 404:
                    logging.error(f"Resource not found at {url}. Aborting further attempts.")
                    break
                elif response.status == 422:
                    error_data = await response.json()
                    logging.error(f"422 Error at {url}. Error details: {error_data}")
                    break
                elif response.status == 499:
                    # Handle cooldown (example use case)
                    response_data = await response.json()
                    cooldown_time = extract_cooldown_time(response_data.get("error", {}).get("message", ""))
                    logging.warning(f"Character is in cooldown for {cooldown_time} seconds.")
                    await asyncio.sleep(cooldown_time + 2)  # Cooldown sleep
                else:
                    error_msg = handle_incorrect_status_code(response.status)
                    logging.warning(f"Request to {url} failed with status {response.status}. {error_msg}. Retrying...")
        except (asyncio.TimeoutError, ClientConnectorError) as e:
            logging.error(f"Request to {url} failed due to {str(e)}. Retrying ({attempt + 1}/{retries})...")
            await asyncio.sleep(min(2 ** attempt, 60))  # Max sleep time is capped at 60 seconds
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)} while making request to {url}.")
            break  # Break out of the loop if an unexpected error occurs

    logging.error(f"Failed to make request to {url} after {retries} attempts.")
    return None


def extract_cooldown_time(message):
    """Extracts cooldown time from the message using regex."""
    match = re.search(r"Character in cooldown: (\d+)", message)
    if match:
        return int(match.group(1))
    return None


async def get_equipments(session: aiohttp.ClientSession) -> dict[str, dict]:
    equipments = {}
    skills_list = ['weaponcrafting', 'gearcrafting', 'jewelrycrafting', 'cooking']
    for skill_name in skills_list:
        equipments[skill_name] = {
            i["code"]: i
            for i in await get_all_items(session, params={'craft_skill': skill_name})
        }
    return equipments


async def get_bank_items(session: aiohttp.ClientSession, params: dict = None) -> dict:
    if params is None:
        params = {}
    url = f"{SERVER}/my/bank/items/"
    data = await make_request(session=session, method='GET', url=url, params=params)
    if data:
        return {item['code']: item['quantity'] for item in data["data"]}
    return {}


async def get_bank_item_qty(session: aiohttp.ClientSession, _item_code: str) -> int:
    res = await get_bank_items(session=session, params={"item_code": _item_code})
    return res.get(_item_code, 0)


async def get_item_infos(session: aiohttp.ClientSession, _item_code: str) -> dict:
    url = f"{SERVER}/items/{_item_code}"
    data = await make_request(session=session, method='GET', url=url)
    return data["data"]["item"] if data else {}


async def get_resource_infos(session: aiohttp.ClientSession, _resource_code: str) -> dict:
    url = f"{SERVER}/resources/{_resource_code}"
    data = await make_request(session=session, method='GET', url=url)
    return data["data"] if data else {}


async def get_craft_recipee(session: aiohttp.ClientSession, _item_code: str) -> dict[str, int]:
    item_infos = await get_item_infos(session, _item_code)
    if item_infos.get('craft', None) is not None:
        return {m['code']: m['quantity'] for m in item_infos['craft']['items']}
    logging.error(f'This material {_item_code} is not craftable')
    return {}


async def get_all_maps(session: aiohttp.ClientSession, params: dict = None) -> list:
    """
    Retrieves all maps from the API.
    Returns a list of maps with their details.
    """
    if params is None:
        params = {}
    url = f"{SERVER}/maps/"
    data = await make_request(session=session, method='GET', url=url, params=params)
    return data["data"] if data else []


async def get_place_name(session: aiohttp.ClientSession, x: int, y: int) -> str:
    url = f"{SERVER}/maps/{x}/{y}"
    data = await make_request(session=session, method='GET', url=url)
    return data["data"]["content"]["code"]


async def get_all_resources(session: aiohttp.ClientSession, params: dict = None) -> list[dict]:
    """
    Retrieves all resources from the API.
    Returns a list of resources with their details.
    """
    data = await make_request(
        session=session,
        method='GET',
        url=f"{SERVER}/resources/",
        params=params if params else {}
    )
    return data["data"] if data else []


async def get_all_monsters(session: aiohttp.ClientSession, params: dict = None) -> list[dict]:
    """
    Retrieves all resources from the API.
    Returns a list of resources with their details.
    """
    data = await make_request(
        session=session,
        method='GET',
        url=f"{SERVER}/monsters/",
        params=params if params else {}
    )
    return data["data"] if data else []


async def get_monster_infos(session: aiohttp.ClientSession, _monster_code: str) -> dict:
    all_monsters = await get_all_monsters(session)
    all_monsters_dict = {
        monster_details['code']: monster_details
        for monster_details in all_monsters
    }
    return all_monsters_dict[_monster_code]


async def get_all_items(session: aiohttp.ClientSession, params: dict) -> list[dict]:
    """
    Retrieves all items from the API.
    Returns a list of items with their details.
    """
    # elif params['craft_skill'] == 'fishing':
    #     params['craft_skill'] = 'cooking'
    data = await make_request(
        session=session,
        method='GET',
        url=f"{SERVER}/items/",
        params=params   # Beware of nb element retrieved (50 only by default) => params set as compulsory
    )
    return data["data"] if data else []


async def is_protected_material(session: aiohttp.ClientSession, _material_code: str) -> bool:
    material_infos = await get_item_infos(session, _material_code)
    is_task_material = material_infos["type"] == "resource" and material_infos["subtype"] == "task"
    is_rare_material = _material_code in ['topaz', 'emerald', 'ruby', 'sapphire', 'sap']
    return is_task_material or is_rare_material


async def get_dropping_resource_locations(session: aiohttp.ClientSession, _material_code: str) -> dict:
    resources_locations = [
        loc for loc in await get_all_resources(session)
        if _material_code in [x['code'] for x in loc['drops']]
    ]
    if resources_locations:
        best_location = min(resources_locations, key=lambda x: x['level'])
        return best_location

    # Return a default or error location if no match is found
    return {
        "name": "Unknown Location",
        "code": "unknown_location",
        "skill": 'N/A',
        "level": 0,
        "drops": []
    }


async def get_dropping_monster_locations(session: aiohttp.ClientSession, _material_code: str) -> dict:
    monsters_locations = [
        loc for loc in await get_all_monsters(session)
        if _material_code in [x['code'] for x in loc['drops']]
    ]
    if monsters_locations:
        best_location = min(monsters_locations, key=lambda x: x['level'])
        return best_location

    # Return a default or error location if no match is found
    return {
        "name": "Unknown Location",
        "code": "unknown_location",
        "skill": 'N/A',
        "level": 0,
        "drops": []
    }


async def get_monster_vulnerability(monster_infos: dict) -> str:

    resistances = sorted([
        res
        for res in ["res_fire", "res_earth", "res_water", "res_air"]
    ], key=lambda x: monster_infos[x], reverse=False
    )

    return resistances[0].replace("res_", "")


async def select_best_equipment_old(equipment1_infos: dict, sorted_valid_equipments: list[dict], vulnerability: str) -> dict:
    if len(sorted_valid_equipments) == 0:
        return equipment1_infos
    if equipment1_infos == {}:
        return sorted_valid_equipments[0]

    # Checking valid equipment one by one
    # TODO pre-select the best equipment out of valid ones
    for equipment2_infos in sorted_valid_equipments:
        if equipment2_infos['level'] > equipment1_infos['level']:
            # TODO add
            return equipment2_infos
        if equipment2_infos['level'] == equipment1_infos['level']:
            effects1_dict = {
                effect['name']: effect['value']
                for effect in equipment1_infos['effects']
            }
            effects2_dict = {
                effect['name']: effect['value']
                for effect in equipment2_infos['effects']
            }
            if equipment2_infos['type'] == "body_armor":
                if effects2_dict['hp'] > effects1_dict['hp']:
                    return equipment2_infos
            if equipment2_infos['type'] == "weapon":
                for element in ['earth', 'air', 'water', 'fire']:
                    attack_element = f'attack_{element}'
                    elt_effect1, elt_effect2 = effects1_dict.get(attack_element, 0), effects2_dict.get(attack_element, 0)
                    if element == vulnerability and elt_effect2 > elt_effect1:
                        return equipment2_infos
            if equipment2_infos['type'] in ["helmet", "ring", "amulet", "leg_armor"]:
                for element in ['earth', 'air', 'water', 'fire']:
                    dmg_element = f'dmg_{element}'
                    elt_effect1, elt_effect2 = effects1_dict.get(dmg_element, 0), effects2_dict.get(dmg_element, 0)
                    if element == vulnerability and elt_effect2 > elt_effect1:
                        return equipment2_infos
                if effects2_dict['hp'] > effects1_dict['hp']:
                    return equipment2_infos
            # TODO adapt according to equipment slot
            if min(effects2_dict.values()) > min(effects1_dict.values()):
                return equipment2_infos

    return equipment1_infos


async def select_best_equipment(equipment1_infos: dict, sorted_valid_equipments: list[dict], vulnerability: str) -> dict:
    """
    Selects the best equipment based on monster vulnerability and equipment effects.

    :param equipment1_infos: The currently equipped item information (or empty dict if none equipped).
    :param sorted_valid_equipments: A list of valid equipment items sorted by level.
    :param vulnerability: The monster's elemental vulnerability (e.g., 'fire', 'water').
    :return: The selected best equipment.
    """
    if len(sorted_valid_equipments) == 0:
        return equipment1_infos
    if not equipment1_infos:
        return sorted_valid_equipments[0]

    def calculate_effect_score(_effects: dict, _vulnerability: str) -> int:
        """
        Calculate the equipment score based on the effects, giving more weight to the monster's vulnerability.
        """
        score = 0
        for effect_name, effect_value in _effects.items():
            if effect_name.startswith("attack_"):
                element = effect_name.replace("attack_", "")
                if element == _vulnerability:
                    score += effect_value * 2  # Double weight for matching vulnerability
                else:
                    score += effect_value  # Lesser weight for other elements
            elif effect_name.startswith("dmg_"):
                element = effect_name.replace("dmg_", "")
                if element == _vulnerability:
                    score += effect_value * 1.5  # Slight weight for resistance-related effects
            elif effect_name == "hp":
                score += effect_value * 0.5  # Lower weight for HP
            elif effect_name == "defense":
                score += effect_value * 0.5  # Lower weight for defense
            else:
                score += effect_value  # Default score for other effects
        return score

    best_equipment = equipment1_infos
    best_score = calculate_effect_score(
        {effect['name']: effect['value'] for effect in equipment1_infos.get('effects', [])},
        vulnerability
    )

    for equipment2_infos in sorted_valid_equipments:
        equipment2_effects = {effect['name']: effect['value'] for effect in equipment2_infos.get('effects', [])}
        equipment2_score = calculate_effect_score(equipment2_effects, vulnerability)

        # Compare scores and select the best
        if equipment2_score > best_score:
            best_equipment = equipment2_infos
            best_score = equipment2_score

    return best_equipment


async def get_bank_consumables(session: aiohttp.ClientSession) -> list[dict]:
    return [
        consumable_infos
        for consumable_infos in await get_all_items(session, params={"type": "consumable"})
        if await get_bank_item_qty(session, consumable_infos["code"]) > 0
    ]


@dataclass
class Task:
    code: str = ""
    type: str = ""  # resources / monsters / items
    total: int = 0


@dataclass
class Character:
    session: aiohttp.ClientSession
    all_equipments: dict[str, dict]     # TODO get it to dataclass
    all_items: dict[str, list[dict]]
    name: str
    skills: list[str]
    collect_skill: str
    craft_skill: str
    craft_target: Optional[str] = None
    fight_target: str = ""
    max_fight_level: int = 0
    stock_qty_objective: int = 500
    task: Task = field(default_factory=Task)
    gatherable_resources: list[dict] = field(default_factory=list)
    craftable_items: list[dict] = field(default_factory=list)
    fightable_monsters: list[dict] = field(default_factory=list)
    fightable_materials: list[dict] = field(default_factory=list)
    objectives: list[dict] = field(default_factory=list)

    def __post_init__(self):
        # Custom logger setup
        self.logger = logging.getLogger(self.name)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(f'%(asctime)s - {self.name} - [%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.propagate = False

        self.excluded_equipments = list(self.all_equipments['excluded'].keys())  # FIXME Shared amongst all characters
        self.equipments = {k: v for c, e in self.all_equipments.items() for k, v in e.items() if c != 'excluded'}

    async def initialize(self):
        """
        This method handles the async initialization of gatherable resources, craftable items,
        and fightable monsters. It should be called explicitly after character creation.
        """
        # TODO add also check on inventory and on task status?
        await asyncio.gather(
            self.set_gatherable_resources(),
            self.set_craftable_items(),
            self.set_fightable_monsters()
        )
        await self.set_objectives()

    async def set_gatherable_resources(self):
        """
        Fetch and set gatherable resources based on the character's collect skill and level
        """
        infos = await self.get_infos()
        gatherable_resources_spots = []
        for collecting_skill in ['mining', 'woodcutting', 'fishing']:
            skill_level = infos[f'{collecting_skill}_level']
            params = {
                "skill": collecting_skill,  # Mining, woodcutting, fishing, etc.
                "max_level": skill_level,    # Limit to resources the character can gather
                # "min_level": max(1, skill_level - 10)
            }
            gatherable_resources_spots.extend(await get_all_resources(self.session, params=params))

        self.gatherable_resources = [
            await get_item_infos(self.session, dropped_material['code'])
            for gatherable_resources_spot in gatherable_resources_spots
            for dropped_material in gatherable_resources_spot['drops']
            if not await is_protected_material(self.session, dropped_material['code'])    # Exclude protected material
            # TODO exclude materials with a low rate of drop
        ]
        self.logger.info(f"Gatherable resources for {self.name}: {[r['code'] for r in self.gatherable_resources]}")

    async def set_craftable_items(self):
        """
        Fetch and set craftable items based on the character's craft skill and level
        """
        infos = await self.get_infos()
        craftable_items = []
        for crafting_skill in self.skills:
            skill_level = infos[f'{crafting_skill}_level']
            skill_craftable_items = [item for item in self.all_items[crafting_skill] if item['level'] <= skill_level]
            # max(1, skill_level - 10) <=
            self.logger.debug(f' crafting_skill: {crafting_skill} > {[i["code"] for i in skill_craftable_items]}')
            craftable_items.extend(skill_craftable_items)
        # TODO exclude protected items (such as the one using jasper_crystal)
        filtered_craftable_items = [
            item
            for item in craftable_items
        ]
        self.craftable_items = filtered_craftable_items
        self.logger.info(f"Craftable items for {self.name}: {[i['code'] for i in self.craftable_items]}")

    async def set_fightable_monsters(self):
        """
        Fetch and set fightable monsters based on the character's maximum fight level
        """

        # Fetch filtered fightable monsters from the API using query parameters
        params = {
            "max_level": self.max_fight_level,
            # "min_level": max(0, self.max_fight_level - 10)
        }
        self.fightable_monsters = await get_all_monsters(self.session, params=params)
        self.fightable_materials = [await get_item_infos(self.session, drop['code']) for monster_details in self.fightable_monsters for drop in monster_details['drops']]
        self.logger.info(f"Fightable monsters for {self.name}: {[m['code'] for m in self.fightable_monsters]}")
        self.logger.info(f"Fightable materials for {self.name}: {[m['code'] for m in self.fightable_materials]}")

    async def get_item_category(self, _item_code) -> str:
        item_details = await get_item_infos(self.session, _item_code)
        if item_details["craft"] is None and item_details["type"] == "resource" and item_details["subtype"] in ["mining", "woodcutting", "fishing"]:
            return 'gathered_resource'
        if item_details["craft"] is None and item_details["type"] == "resource" and item_details["subtype"] in ["mob", "food"]:
            return 'dropped_resource'
        if item_details["craft"] is not None:
            return 'craftable_resource'
        # 'jasper_crystal' is resource > task // 'lich_crown' is helmet > ""
        if _item_code in ['wooden_stick', 'tasks_coin', 'bandit_armor', 'death_knight_sword', 'lich_crown', 'life_crystal'] or item_details["subtype"] in ["task"]:
            return 'given_resource'
        return 'UNKNOWN'

    async def can_be_home_made(self, item_code: str) -> bool:
        # if item out of resource: is it gatherable
        # if item out of mob/drop: is it fightable
        # if craftable item: can all material be home made?
        item_category = await self.get_item_category(item_code)
        self.logger.debug(f' Checking if {item_code} can be home made from its category {item_category}')
        if item_category == 'gathered_resource':
            return item_code in [i['code'] for i in self.gatherable_resources]
        elif item_category == 'dropped_resource':
            return item_code in [i['code'] for i in self.fightable_materials]
        elif item_category == 'craftable_resource':
            craft_recipee = await get_craft_recipee(self.session, _item_code=item_code)
            return all([await self.can_be_home_made(material_code) for material_code in list(craft_recipee.keys())])
        elif item_category == 'given_resource':
            return False
        else:
            self.logger.warning(f' {item_code} to categorize')

    async def set_objectives(self):
        # Out of craftable items, which one can be handled autonomously
        objectives = [
            item
            for item in self.craftable_items
            if await self.can_be_home_made(item['code']) and await self.does_item_provide_xp(item)
        ]
        objectives.extend(self.gatherable_resources)
        objectives.extend(self.fightable_materials)
        self.objectives = objectives
        self.logger.info(f' OBJECTIVES: {[o["code"] for o in self.objectives]}')

    async def get_craft_skill(self):
        return self.craft_skill

    async def get_infos(self) -> dict:
        url = f"{SERVER}/characters/{self.name}"
        data = await make_request(session=self.session, method='GET', url=url)
        if data:
            self.logger.debug("Fetched character info successfully.")
        else:
            self.logger.error("Failed to fetch character info.")
        return data["data"] if data else {}

    async def get_potentially_craftable_items(self) -> dict:

        skill_name = await self.get_craft_skill()
        skill_level = await self.get_skill_level()

        self.logger.info(f' craft skill: {skill_name} ({skill_level})')

        params = {
            'max_level': skill_level,
            'craft_skill': skill_name
        }

        potentially_craftable_items = await get_all_items(
            session=self.session,
            params=params
        )

        potentially_craftable_items_codes = list(map(lambda x: x["code"], potentially_craftable_items))
        self.logger.debug(f'MAY craft these items: {potentially_craftable_items_codes} according to its {skill_name} craft skill (level {skill_level})')

        return {item["code"]: item for item in potentially_craftable_items}

    async def move_to_bank(self):
        bank_coords = await self.get_nearest_coords(
            content_type='bank',
            content_code='bank'
        )
        cooldown_ = await self.move(*bank_coords)
        await asyncio.sleep(cooldown_)

    async def deposit_items_at_bank(self, _items_details: dict[str, int] = None):
        if await self.get_inventory_occupied_slots_nb() > 0:
            # Go to bank and deposit all objects
            # Move to the bank
            await self.move_to_bank()

            if _items_details is None:
                _items_details = await self.get_inventory_items()
                gold_amount = await self.get_gold_amount()
                if gold_amount > 0:
                    _items_details['gold'] = gold_amount

            self.logger.debug(f'depositing at bank: {_items_details} ...')
            for item_code, item_qty in _items_details.items():
                cooldown_ = await self.bank_deposit(item_code, item_qty)
                await asyncio.sleep(cooldown_)

    async def get_gold_amount(self) -> int:
        infos = await self.get_infos()
        return infos['gold']

    async def withdraw_items_from_bank(self, _items_details: dict[str, int]):
        # Move to the bank
        await self.move_to_bank()

        self.logger.debug(f'collecting at bank: {_items_details} ...')
        for item_code, item_qty in _items_details.items():
            cooldown_ = await self.bank_withdraw(item_code, item_qty)
            await asyncio.sleep(cooldown_)

    async def gather_material(self, material_code: str, quantity: int):

        location_details = await get_dropping_resource_locations(self.session, material_code)
        location_coords = await self.get_nearest_coords('resource', location_details['code'])

        cooldown_ = await self.move(*location_coords)
        await asyncio.sleep(cooldown_)

        self.logger.info(f'gathering {quantity} {material_code} ...')
        while await self.is_up_to_gather(material_code, quantity):
            cooldown_ = await self.perform_gathering()
            await asyncio.sleep(cooldown_)

    async def is_up_to_gather(self, _material_code: str, target_qty: int):
        infos = await self.get_infos()
        inventory_items = await self.get_inventory_items()
        gathered_qty = inventory_items.get(_material_code, 0)
        inventory_not_full = sum(inventory_items.values()) < infos.get('inventory_max_items', 100)
        return inventory_not_full and gathered_qty < target_qty

    async def go_and_fight(self):
        if self.fight_target != "":
            await self.move_to_monster()
            self.logger.info(f'fighting {self.fight_target} ...')
            while await self.is_up_to_fight():
                cooldown_ = await self.perform_fighting()
                await asyncio.sleep(cooldown_)

    async def is_up_to_fight(self):
        got_enough_consumables = await self.got_enough_consumables(10)
        inventory_not_full = await self.is_inventory_not_full()
        is_task_complete = await self.is_task_completed()
        return got_enough_consumables and inventory_not_full and not is_task_complete

    async def go_and_fight_to_collect(self, material_code: str, quantity_to_get: int):
        # Identify the monster and go there IF fightable
        if not await self.is_fightable(material_code):
            return

        location_details = await get_dropping_monster_locations(self.session, material_code)
        monster_code = location_details['code']
        await self.move_to_monster(monster_code)

        while await self.is_inventory_not_full() and await self.get_inventory_quantity(material_code) < quantity_to_get:
            cooldown_ = await self.perform_fighting()
            await asyncio.sleep(cooldown_)

    async def got_enough_consumables(self, min_qty: int):
        # TODO min_qty can be an attribute linked to fight target (depending on difficulty)
        character_infos = await self.get_infos()
        consumable1_qty = character_infos.get('consumable1_slot_quantity', 0)
        consumable2_qty = character_infos.get('consumable2_slot_quantity', 0)
        return consumable1_qty + consumable2_qty > min_qty

    async def get_skill_level(self, _skill: str = None) -> int:
        # FIXME
        infos = await self.get_infos()
        if _skill is None:
            _skill = await self.get_craft_skill()
        elif _skill == 'mob':
            return await self.get_level()
        elif _skill == 'food':
            return await self.get_skill_level('cooking')
        return infos[f'{_skill}_level']

    async def get_level(self) -> int:
        infos = await self.get_infos()
        return infos['level']

    async def get_inventory_occupied_slots_nb(self) -> int:
        infos = await self.get_infos()
        return sum([
            i_infos['quantity']
            for i_infos in infos.get('inventory', [])
            if i_infos['code'] != ""
        ])

    async def get_inventory_free_slots_nb(self) -> int:
        return await self.get_inventory_max_size() - await self.get_inventory_occupied_slots_nb()

    async def get_inventory_max_size(self) -> int:
        infos = await self.get_infos()
        return infos.get('inventory_max_items', 100)

    async def get_inventory_items(self) -> dict:
        infos = await self.get_infos()
        return {
            i_infos['code']: i_infos['quantity']
            for i_infos in infos.get('inventory', {})
            if i_infos['code'] != ""
        }

    async def is_inventory_not_full(self) -> bool:
        infos = await self.get_infos()
        return sum([
            i_infos['quantity']
            for i_infos in infos.get('inventory', [])
        ]) < await self.get_inventory_max_size()

    async def get_current_location(self) -> tuple[int, int]:
        infos = await self.get_infos()
        return int(infos.get('x', 0)), int(infos.get('y', 0))   # Default to (0, 0)

    async def complete_task(self):
        url = f"{SERVER}/my/{self.name}/action/task/complete"
        data = await make_request(session=self.session, method='POST', url=url)
        if data:
            _cooldown = data["data"]["cooldown"]["total_seconds"]
            await asyncio.sleep(_cooldown)
        else:
            self.logger.error(f'failed to complete task.')

    async def accept_new_task(self):
        url = f"{SERVER}/my/{self.name}/action/task/new"
        data = await make_request(session=self.session, method='POST', url=url)
        if data:
            _cooldown = data["data"]["cooldown"]["total_seconds"]
            await asyncio.sleep(_cooldown)
        else:
            self.logger.error(f'failed to get new task.')

    async def move(self, x, y) -> int:
        current_location = await self.get_current_location()
        if current_location == (x, y):
            self.logger.debug(f'is already at the location ({x}, {y})')
            return 0

        url = f"{SERVER}/my/{self.name}/action/move"
        payload = {"x": x, "y": y}

        data = await make_request(session=self.session, method='POST', url=url, payload=payload)
        if data:
            _cooldown = data["data"]["cooldown"]["total_seconds"]
            self.logger.debug(f'moved to ({await get_place_name(self.session, x, y)}). Cooldown: {_cooldown} seconds')
            return _cooldown
        else:
            self.logger.error(f'failed to move to ({x}, {y})')
            return 0

    async def get_nearest_coords(self, content_type: str, content_code: str) -> tuple[int, int]:
        if content_type == 'workshop' and content_code == 'fishing':
            content_code = 'cooking'
        resource_locations = await get_all_maps(
            session=self.session,
            params={
                'content_type': content_type,
                'content_code': content_code
            }
        )
        nearest_resource = {'x': 4, 'y': 1}     # Default to bank
        if len(resource_locations) == 0:
            self.logger.info(f'No resource {content_code} on this map')
            return nearest_resource['x'], nearest_resource['y']

        if len(resource_locations) == 1:
            return int(resource_locations[0]['x']), int(resource_locations[0]['y'])

        min_dist = 999999
        for resource_loc in resource_locations:
            res_x, res_y = int(resource_loc['x']), int(resource_loc['y'])
            character_location_x, character_location_y = await self.get_current_location()
            dist_to_loc = (res_x - character_location_x) ** 2 + (res_y - character_location_y) ** 2
            if dist_to_loc < min_dist:
                min_dist = dist_to_loc
                nearest_resource = resource_loc
        return nearest_resource['x'], nearest_resource['y']

    async def get_inventory_quantity(self, _item_code: str) -> int:
        character_infos = await self.get_infos()
        for item_infos in character_infos.get('inventory', []):
            if item_infos['code'] == _item_code:
                return item_infos['quantity']
        return 0

    async def get_nb_craftable_items(self, _item_code: str, from_inventory: bool = False) -> int:

        craft_recipee = await get_craft_recipee(self.session, _item_code)
        self.logger.debug(f' recipee for {_item_code} is {craft_recipee}')
        total_nb_materials = sum([qty for _, qty in craft_recipee.items()])
        nb_craftable_items = await self.get_inventory_max_size() // total_nb_materials

        if from_inventory:  # Taking into account drops > update qty available in inventory
            for material_code, qty in craft_recipee.items():
                nb_craftable_items = min(await self.get_inventory_quantity(material_code)//qty, nb_craftable_items)

        self.logger.info(f' nb of craftable items {"from inventory" if from_inventory else ""} for {_item_code} is {nb_craftable_items}')

        return nb_craftable_items

    async def bank_deposit(self, item_code: str, quantity: int) -> int:
        url = f"{SERVER}/my/{self.name}/action/bank/deposit"
        if item_code == 'gold':
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
            self.logger.info(f'{quantity} {item_code} deposited. Cooldown: {_cooldown} seconds')
            return _cooldown
        else:
            self.logger.error(f'Failed to deposit {quantity} {item_code}')
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
            self.logger.info(f'{quantity} {item_code} withdrawn. Cooldown: {_cooldown} seconds')
            return _cooldown
        else:
            self.logger.error(f'Failed to withdraw {quantity} {item_code}')
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
            self.logger.info(f'{qte} {item_code} crafted. XP gained: {gained_xp}. Cooldown: {_cooldown} seconds')
            return _cooldown
        else:
            self.logger.error(f'Failed to craft {qte} {item_code}')
            return 0

    async def perform_fighting(self) -> int:
        url = f"{SERVER}/my/{self.name}/action/fight"
        data = await make_request(session=self.session, method='POST', url=url)
        if data:
            _cooldown = data["data"]["cooldown"]["total_seconds"]
            # self.logger.info(f'won. Cooldown: {_cooldown} seconds')
            return _cooldown
        else:
            self.logger.error(f'failed to perform fighting action.')
            return 0

    async def perform_gathering(self) -> int:
        url = f"{SERVER}/my/{self.name}/action/gathering"
        data = await make_request(session=self.session, method='POST', url=url)
        if data:
            _cooldown = data["data"]["cooldown"]["total_seconds"]
            # self.logger.info(f'Resource gathered. Cooldown: {_cooldown} seconds')
            return _cooldown
        else:
            self.logger.error(f'failed to gather resources.')
            return 0

    async def perform_recycling(self, item_code: str, qte: int) -> int:
        url = f"{SERVER}/my/{self.name}/action/recycling"
        payload = {
            "code": item_code,
            "quantity": qte
        }
        data = await make_request(session=self.session, method='POST', url=url, payload=payload)
        if data:
            _cooldown = data["data"]["cooldown"]["total_seconds"]
            self.logger.info(f'{qte} {item_code} recycled. Cooldown: {_cooldown} seconds')
            return _cooldown
        else:
            self.logger.error(f'failed to recycle {qte} {item_code}.')
            return 0

    async def is_valid_item(self, _item_code: str) -> bool:
        self.logger.debug(f'Checking validity of item {_item_code}')
        # If available in bank, let's consider it reachable
        if await get_bank_item_qty(self.session, _item_code) > 0:
            # Check if the item is not to be kept (precious one not to be used for now)
            return await self.is_collectable(_item_code)

        if await is_protected_material(self.session, _item_code) or _item_code in ['iron_pickaxe', 'iron_axe']:
            return False

        item_details = await get_item_infos(self.session, _item_code)
        # Check if it can be gathered
        if item_details['craft'] is None:
            # It can be a classical ressource or a bonus drop
            can_be_handled = await self.is_gatherable(_item_code) or await self.is_fightable(_item_code)
            self.logger.debug(f' Checking if {item_details["code"]} is gatherable or fightable: {can_be_handled}')
            return can_be_handled
            # if item_details['subtype'] in ['mining', 'woodcutting', 'fishing']:
            #     skill_name, skill_level = item_details['subtype'], item_details['level']
            #     return await self.get_skill_level(skill_name) >= skill_level
        else:
            self.logger.debug(f'Checking validity of item {_item_code} as a craftable item')
            if await self.is_craftable(_item_code):
                return all([await self.is_valid_item(material['code']) for material in item_details['craft']['items']])
        return False

    async def is_gatherable(self, resource_code) -> bool:
        return resource_code in [item["code"] for item in self.gatherable_resources] and resource_code in [item["code"] for item in self.objectives]

    async def is_gatherable_old(self, resource_code) -> bool:
        resource_infos = await get_item_infos(self.session, resource_code)
        if resource_infos["type"] != 'resource':
            return False
        # if resource_infos["subtype"] == 'mob':  # to be checked as fightable
        #     return False
        if resource_infos["subtype"] not in ['mining', 'woodcutting', 'fishing']:   # checking if indeed gatherable
            return False
        # TODO get an attribute to check on as "gatherable_ressources"
        return await self.get_skill_level(resource_infos['subtype']) >= resource_infos['level']

    async def is_fightable(self, material_code) -> bool:
        return material_code in [item["code"] for item in self.fightable_materials] and material_code in [item["code"] for item in self.objectives]

    async def is_fightable_old(self, material_code) -> bool:
        fightable_monsters = await get_all_monsters(self.session, {'max_level': self.max_fight_level})
        location_details = await get_dropping_monster_locations(self.session, material_code)
        monster_code = location_details['code']
        self.logger.debug(f' {material_code} can be dropped by {monster_code} and {self.name} can handle {[m["code"] for m in fightable_monsters]}')
        return monster_code in [m['code'] for m in fightable_monsters]

    async def is_craftable(self, item_code) -> bool:
        return item_code in [item["code"] for item in self.craftable_items] and item_code in [item["code"] for item in self.objectives]

    async def is_craftable_old(self, item_code) -> bool:
        item_infos = await get_item_infos(self.session, item_code)
        if item_infos['craft'] is None:
            return False
        # Do not automatically craft with rare items
        if any([await is_protected_material(self.session, item["code"]) for item in item_infos["craft"]["items"]]):
            return False
        # TODO get an attribute to check on as "craftable_items"
        return await self.get_skill_level(item_infos['craft']['skill']) >= item_infos['craft']['level']

    async def is_collectable(self, item_code: str) -> bool:
        return not (await is_protected_material(self.session, item_code))

    async def go_and_craft_item(self, item_code: str = None):

        if item_code is None:
            item_code = self.craft_target

        # Taking bonus drop into account
        nb_items_to_craft = await self.get_nb_craftable_items(item_code, from_inventory=True)

        if nb_items_to_craft > 0:
            await self.move_to_workshop(item_code)
            cooldown_ = await self.perform_crafting(item_code, nb_items_to_craft)
            await asyncio.sleep(cooldown_)

    async def move_to_workshop(self, item_code: str = None):
        if item_code is None:
            skill_name = await self.get_craft_target_skill()
        else:
            # get the skill out of item
            item_infos = await get_item_infos(self.session, item_code)
            skill_name = item_infos['craft']['skill']
        coords = await self.get_nearest_coords(content_type='workshop', content_code=skill_name)
        self.logger.info(f'{self.name} > moving to workshop at {coords}')
        cooldown_ = await self.move(*coords)
        await asyncio.sleep(cooldown_)

    async def move_to_monster(self, monster_code: str = ""):
        monster_code = await self.get_monster_target() if monster_code == "" else monster_code
        coords = await self.get_nearest_coords(content_type='monster', content_code=monster_code)
        self.logger.info(f'{self.name} > moving to monster {monster_code} at {coords}')
        cooldown_ = await self.move(*coords)
        await asyncio.sleep(cooldown_)

    async def move_to_task_master(self):
        coords = await self.get_nearest_coords(content_type='tasks_master', content_code='monsters')
        self.logger.info(f'{self.name} > moving to tasks master at {coords}')
        cooldown_ = await self.move(*coords)
        await asyncio.sleep(cooldown_)

    async def get_craft_target_skill(self):
        target_infos = await get_item_infos(self.session, self.craft_target)
        return target_infos['craft']['skill']

    async def prepare_fighting_for_item(self, _item_code: str = None):
        if _item_code is not None:
            monster_location = await get_dropping_monster_locations(self.session, _item_code)
            monster_code = monster_location['code']
            self.logger.info(f' item {_item_code} is dropped at {monster_location}')
            self.fight_target = monster_code

    async def equip_for_fight(self):
        # TODO get the best weapons and equipment for specific fight target
        await self.auto_equip()
        return

    async def get_equipment_code(self, _equipment_slot: str):
        infos = await self.get_infos()
        return infos[f'{_equipment_slot}_slot']

    async def go_and_equip(self, _equipment_slot: str, _equipment_code: str):
        current_equipment_code = await self.get_equipment_code(_equipment_slot)
        if current_equipment_code != _equipment_code:
            self.logger.info(f' will change equipment for {_equipment_slot} from {current_equipment_code} to {_equipment_code}')
            await self.move_to_bank()
            await self.withdraw_items_from_bank({_equipment_code: 1})
            if current_equipment_code != "":
                await self.unequip(_equipment_slot)
                await self.deposit_items_at_bank({current_equipment_code: 1})
            await self.equip(_equipment_code, _equipment_slot)

    async def equip_for_gathering(self, _item_code: str):
        item_infos = await get_item_infos(self.session, _item_code)

        # TODO check available equipements with "effects" > "name" == "gathering_skill"

        if item_infos["subtype"] == "mining":
            await self.go_and_equip("weapon", "iron_pickaxe")
        elif item_infos["subtype"] == "woodcutting":
            await self.go_and_equip("weapon", "iron_axe")
        elif item_infos["subtype"] == "fishing":
            await self.go_and_equip("weapon", "spruce_fishing_rod")
        return

    async def gather_and_collect(self, _craft_details: dict[str, dict[str, int]]):
        fight_details = _craft_details['fight']
        if sum([qty for _, qty in fight_details.items()]) > 0:
            self.logger.info(f'going to fight: {fight_details} ...')
            for item_code, qty in fight_details.items():

                await self.prepare_fighting_for_item(item_code)

                await self.equip_for_fight()

                await self.go_and_fight_to_collect(item_code, qty)

        for item_code, qty in _craft_details['gather'].items():

            await self.equip_for_gathering(item_code)

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
        for inventory_item_code, inventory_qty in (await self.get_inventory_items()).items():
            if inventory_item_code not in material2craft_qty:
                material2deposit_qty[inventory_item_code] = inventory_qty
            else:
                exceeding_qty = inventory_qty - material2craft_qty[inventory_item_code]
                if exceeding_qty > 0:
                    material2deposit_qty[inventory_item_code] = exceeding_qty

        return material2deposit_qty

    async def set_target(self, _craft_target_code: str):
        self.craft_target = _craft_target_code
        self.logger.info(f' Target set to {self.craft_target}')

    async def get_target_details(self) -> dict[str, int]:

        item_infos = await get_item_infos(self.session, self.craft_target)
        craft_recipee = await get_craft_recipee(self.session, item_infos['code'])

        # Can be limited by material only available in bank
        nb_items_to_craft = await self.get_nb_craftable_items(item_infos['code'])

        return {
            material: qty * nb_items_to_craft
            for material, qty in craft_recipee.items()
        }

    async def prepare_to_craft(self) -> dict[str, dict[str, int]]:
        gather_details, collect_details, fight_details = {}, {}, {}
        target_details = await self.get_target_details()
        self.logger.info(f' objectives: {target_details}')
        for material, qty in target_details.items():

            # TODO qualify item: craftable? gatherable? fightable?
            self.logger.debug(f' Check material {material}')

            qty_at_bank = await get_bank_item_qty(self.session, material)
            self.logger.debug(f' Qty of {material} available at bank: {qty_at_bank}')
            if qty_at_bank > 3*qty:
                # Réserver le montant pour qu'un autre personnage ne compte pas dessus
                qty_to_collect = min(qty, qty_at_bank)
                collect_details[material] = qty_to_collect
                qty -= qty_to_collect
                self.logger.debug(f' Collectable from bank {qty_at_bank}. remaining to get {qty}')
                if qty == 0:
                    continue
            if await self.is_craftable(material):
                # Set material as craft target
                self.logger.info(f' Resetting target to {material}')
                await self.set_target(material)
                # "Dé-réserver" les articles de banque
                return await self.prepare_to_craft()
            if await self.is_gatherable(material):
                gather_details[material] = qty
                self.logger.debug(f' Gatherable qty {qty}')
                continue
            if await self.is_fightable(material):
                fight_details[material] = qty
                self.logger.debug(f' Fightable for qty {qty}')
                continue
            self.logger.warning(f' Material {material} has not been categorized...')
        return {
            'gather': gather_details,
            'collect': collect_details,
            'fight': fight_details
        }

    async def does_item_provide_xp(self, item_details: dict) -> bool:
        """
        Determine if a given item will provide XP when crafted, gathered, or fished.
        This depends on the item's level compared to the character's skill level.
        """
        # Check if the item is craftable
        if item_details.get('craft', None):
            skill_name = item_details['craft']['skill']
        else:
            # Non-craftable items are checked by subtype (gathering, fishing, etc.)
            skill_name = item_details.get('subtype', None)

        # If there's no valid skill (item can't be crafted or gathered), it provides no XP
        if not skill_name:
            return False

        # Get the character's skill level for the relevant skill (crafting, gathering, etc.)
        skill_level = await self.get_skill_level(skill_name)

        # Item level must be within range of skill level to provide XP (e.g., within 10 levels)
        item_level = item_details['level']

        # Example threshold: if item is within 10 levels of the character's skill level, it gives XP
        return item_level >= (skill_level - 10)

    async def select_eligible_targets(self) -> list[str]:
        """
        Select eligible targets (tasks) for the character, ensuring that they will gain XP from them.
        Includes crafting, gathering, and fishing tasks. Returns a list of eligible items that provide XP.
        """
        self.logger.debug(f"Selecting eligible targets for {self.name}")

        # Fetch all potentially craftable items based on character's skill level
        # craftable_items_dict = await self.get_potentially_craftable_items()

        # Filter items that are valid and provide XP (this includes craftable, gatherable, and fishable items)
        # valid_craftable_items = [
        #     item_details for item_code, item_details in craftable_items_dict.items()
        #     if await self.is_valid_item(item_code) and await self.does_item_provide_xp(item_details) and not await is_protected_material(self.session, item_code)
        # ]

        valid_craftable_items = self.gatherable_resources

        # Log eligible items
        self.logger.debug(f'Eligible items for XP: {[item["code"] for item in valid_craftable_items]}')

        if not valid_craftable_items:
            self.logger.warning(f'No valid craftable/gatherable items found for {self.name} that provide XP.')
            return []

        # Sort items by their rarity in the bank (to prioritize items that are rarer)
        items2bank_qty = {
            craftable_item['code']: await get_bank_item_qty(self.session, craftable_item['code'])
            for craftable_item in valid_craftable_items
        }

        self.craftable_items = sorted(valid_craftable_items, key=lambda x: items2bank_qty.get(x['code'], 0), reverse=False)

        return [item['code'] for item in self.craftable_items]

    async def is_valid_equipment(self, equipment_infos: dict) -> bool:
        character_level = await self.get_level()
        return character_level >= equipment_infos['level']

    async def auto_equip(self):

        # Identify vulnerability
        monster_infos = await get_monster_infos(self.session, await self.get_monster_target())
        vulnerability = await get_monster_vulnerability(monster_infos=monster_infos)
        self.logger.info(f' monster {self.fight_target} vulnerability is {vulnerability}')

        # Manage equipment
        for equipment_slot in ['weapon', 'shield', 'helmet', 'body_armor', 'leg_armor', 'boots', 'ring1', 'ring2',
                               'amulet', 'artifact1', 'artifact2', 'artifact3']:
            await self.equip_best_equipment(equipment_slot, vulnerability)

        # Manage consumables
        await self.equip_best_consumables()

    async def get_2_best_consumables_including_equipped(self) -> list[dict]:
        """
        Fetches the two best consumables, including currently equipped ones, and ranks them.
        """
        # Fetch all consumables from the bank (TODO and inventory)
        bank_consumables = await get_bank_consumables(self.session)
        valid_consumables = [
            consumable for consumable in bank_consumables
            if not await self.is_protected_consumable(consumable['code']) and await self.is_valid_consumable(consumable['level'])
        ]

        # Add the currently equipped consumables to the list of valid ones (if they are equipped)
        valid_consumables_codes = [c["code"] for c in valid_consumables]
        ordered_current_consumables = await self.get_ordered_current_consumables()
        for current_consumable in ordered_current_consumables:
            if current_consumable and current_consumable["code"] not in valid_consumables_codes:
                valid_consumables.append(current_consumable)

        # Sort all consumables by level (higher is better)
        sorted_two_best_consumables = sorted(valid_consumables, key=lambda x: x['level'], reverse=True)[:2]

        # Initialize result as None placeholders for two consumables
        two_best_consumables = [None, None]

        # Single loop to determine the best consumables for each slot
        for consumable in sorted_two_best_consumables:
            if ordered_current_consumables[0] and consumable["code"] == ordered_current_consumables[0]["code"]:
                # Keep in the same slot if it's already consumable1
                two_best_consumables[0] = consumable
            elif ordered_current_consumables[1] and consumable["code"] == ordered_current_consumables[1]["code"]:
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

    async def is_protected_consumable(self, consumable_code: str) -> bool:
        return consumable_code in EXCLUDED_CONSUMABLES

    async def is_valid_consumable(self, consumable_level: int) -> bool:
        return consumable_level <= await self.get_level()

    async def equip_best_consumables(self):
        """
        Equip the best available consumables in the consumable1 and consumable2 slots.
        This function avoids unequipping if the consumable is already equipped and fully stocked.
        """
        # Fetch the two best consumables, ensuring they match the current slot order
        new_consumables = await self.get_2_best_consumables_including_equipped()

        for i, slot in enumerate(["consumable1", "consumable2"]):
            new_consumable = new_consumables[i]

            if new_consumable is None:
                self.logger.info(f"No valid consumable available for {slot}. Skipping.")
                continue

            # Get current consumable details for the slot
            character_infos = await self.get_infos()
            current_code = character_infos.get(f"{slot}_slot", "")
            current_qty = character_infos.get(f"{slot}_slot_quantity", 0)
            new_code = new_consumable["code"]

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
                    self.logger.info(f"{slot} already equipped with {new_code} and fully stocked.")
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

    async def get_ordered_current_consumables(self) -> list[dict]:
        character_infos = await self.get_infos()
        currently_equipped_consumables = [character_infos['consumable1_slot'], character_infos['consumable2_slot']]
        self.logger.debug(f' Currently equipped consumables: {currently_equipped_consumables}')
        return [self.equipments[c] if c else None for c in currently_equipped_consumables]

    async def equip_best_equipment(self, _equipment_slot: str, vulnerability: str):
        available_equipments = await self.get_bank_equipments_for_slot(_equipment_slot)
        sorted_valid_equipments = sorted([
            equipment
            for equipment in available_equipments
            if await self.is_valid_equipment(equipment)
        ], key=lambda x: x['level'], reverse=True)

        self.logger.debug(f'may be equipped with {[e["code"] for e in sorted_valid_equipments]}')

        current_equipment_code = await self.get_equipment_code(_equipment_slot)
        if len(sorted_valid_equipments) == 0:
            return
        current_equipment_infos = self.equipments.get(current_equipment_code, {})
        new_equipment_details = await select_best_equipment(current_equipment_infos, sorted_valid_equipments, vulnerability)
        self.logger.debug(f' has been assigned {new_equipment_details.get("code", "")} for slot {_equipment_slot} instead of {current_equipment_infos.get("code", "")}')
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
            self.logger.info(f'{qte} {slot_code} unequipped. Cooldown: {_cooldown} seconds')
            return _cooldown
        else:
            self.logger.error(f'Failed to unequip {qte} {slot_code}')
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
            self.logger.info(f'{qte} {item_code} equipped at slot {slot_code}. Cooldown: {_cooldown} seconds')
            return _cooldown
        else:
            self.logger.error(f'Failed to equip {qte} {item_code} at slot {slot_code}')
            return 0

    async def unequip(self, slot_code: str, qte: int = 1):
        cooldown_ = await self.perform_unequip(slot_code, qte)
        await asyncio.sleep(cooldown_)

    async def equip(self, item_code: str, slot_code: str, qte: int = 1):
        cooldown_ = await self.perform_equip(item_code, slot_code, qte)
        await asyncio.sleep(cooldown_)

    async def get_bank_equipments_for_slot(self, equipment_slot: str) -> list[dict]:
        return [
            self.equipments[item_code]
            for item_code in (await get_bank_items(self.session)).keys()
            # 'ring' in 'ring1'
            if self.equipments.get(item_code, None) is not None and self.equipments[item_code]['type'] in equipment_slot
        ]

    async def manage_task(self):
        character_infos = await self.get_infos()
        # if task completed (or none assigned yet), go to get rewards and renew task
        if await self.is_task_completed():
            # go to task master
            await self.move_to_task_master()
            # if task, get reward
            if character_infos["task"] != "":
                await self.complete_task()
            # ask for new task
            await self.accept_new_task()

    async def is_task_completed(self) -> bool:

        # TODO can also be a personal task (amount of collectibles) - allow for some materials to be collected by others

        character_infos = await self.get_infos()
        return character_infos.get("task_progress", "A") == character_infos.get("task_total", "B")

    async def update_fight_target(self):
        # list all fightable monsters according to max level (+ add the dropped items to the collectibles)
        fightable_monsters = await get_all_monsters(self.session, {'max_level': self.max_fight_level})

        self.logger.info(f' task target is {self.task.code} and fightable monsters: {[m["code"] for m in fightable_monsters]}')

        if len(fightable_monsters) > 0:
            highest_monster = fightable_monsters[-1]
            if self.task.code in [monster['code'] for monster in fightable_monsters]:
                await self.set_fight_target(self.task.code)
            elif await self.get_level() - 10 <= highest_monster['level']:
                await self.set_fight_target(highest_monster['code'])
            else:
                # Deactivation of fight mode
                await self.set_fight_target("")
            # TODO set its dropped items as collectibles
        return

    async def get_monster_target(self):
        return self.fight_target

    async def set_fight_target(self, monster_code: str):
        self.fight_target = monster_code

    async def is_on_fight_mode(self):
        if self.fight_target != "":
            await self.equip_best_consumables()
            is_filled_up = await self.got_enough_consumables(10)
            self.logger.debug(f' Ready to fight? {is_filled_up}')
            return is_filled_up
        return False

    async def get_stock_items(self) -> dict:
        # List items corresponding to collect skill
        collect_params = {
            'craft_skill': 'cooking' if self.collect_skill == 'fishing' else self.collect_skill,
            'max_level': await self.get_skill_level(self.collect_skill)
        }
        self.logger.info(collect_params)
        return {
            item["code"]: item
            for item in await get_all_items(self.session, collect_params)
        }

    async def update_stocks(self):
        craftable_stock_items = await self.get_stock_items()
        craftable_stock_items = [v for v in craftable_stock_items.values() if await self.is_valid_item(v['code'])]
        sorted_craftable_stock_items = sorted(craftable_stock_items, key=lambda x: x['level'], reverse=True)
        for item_to_craft in sorted_craftable_stock_items:
            # stock status
            item_to_craft_code = item_to_craft['code']
            if await get_bank_item_qty(self.session, item_to_craft_code) < self.stock_qty_objective:
                self.logger.info(f' stock assessment for {item_to_craft_code}: {await get_bank_item_qty(self.session, item_to_craft_code)} VS {self.stock_qty_objective}')
                await self.set_target(item_to_craft_code)
                return

    async def recycle_excluded_equipments(self):
        for item_code in self.excluded_equipments:
            qty_at_bank = await get_bank_item_qty(self.session, item_code)
            if qty_at_bank > 0:
                # If yes, withdraw them and get to workshop to recycle them, before getting back to bank to deposit all
                nb_free_inventory_slots = await self.get_inventory_free_slots_nb()
                recycling_qty = min(qty_at_bank, nb_free_inventory_slots // 2)  # Need room in inventory when recycling
                await self.withdraw_items_from_bank({item_code: recycling_qty})
                await self.move_to_workshop(item_code)
                cooldown = await self.perform_recycling(item_code, recycling_qty)
                await asyncio.sleep(cooldown)
                await self.deposit_items_at_bank()

    async def is_worth_selling(self, _item_code) -> bool:
        item_infos = await get_item_infos(self.session, _item_code)
        item_gold_value = item_infos["ge"]["sell_price"]
        craft_recipee = await get_craft_recipee(self.session, _item_code)
        materials = [await get_item_infos(self.session, material) for material in craft_recipee.keys()]
        if any([await is_protected_material(self.session, material["code"]) for material in materials]):
            return False
        gold_value_sorted_materials = sorted(materials, key=lambda x: x["ge"]["buy_price"], reverse=True)
        return item_gold_value > sum(gold_value_sorted_materials[:2])

    async def get_game_task(self) -> Task:
        infos = await self.get_infos()
        return Task(
            code=infos["task"],
            type=infos["task_type"],
            total=infos["task_total"]-infos["task_progress"]
        )


async def run_bot(character_object: Character):
    while True:

        ### DEPOSIT ALL ###
        await character_object.deposit_items_at_bank()

        ### RECYCLE ###
        await character_object.recycle_excluded_equipments()
        # TODO recycle equipments when more than 5

        character_object.logger.debug(" ### BEGIN STOCKS UPDATE ### ")
        await character_object.update_stocks()
        character_object.logger.debug(" ### END STOCKS UPDATE ### ")

        ## MANAGE TASK ###
        await character_object.manage_task()


        ### SET ATTRIBUTES about craftable/gatherable/fightable stuff ###



        if character_object.task.total == 0:
            # either get to task master if game task or proceed to next step/task (bank or crafting) if custom task
            if character_object.task.type == "monsters":
                # go to task master, get reward and get new task
                await character_object.move_to_task_master()
                await character_object.complete_task()
                await character_object.accept_new_task()

        # Check if game task is feasible, assign if it is
        game_task = await character_object.get_game_task()
        if game_task.type == "monsters" and game_task.code in character_object.fightable_monsters:
            character_object.task = game_task

        # Proceed with assigned task
        if character_object.task.type == "":
            # find and assign a valid task
            character_object.logger.info(f' can gather {[x["code"] for x in character_object.gatherable_resources]}')
            character_object.logger.info(f' can craft {[x["code"] for x in character_object.craftable_items]}')
            character_object.logger.info(f' can fight {[x["code"] for x in character_object.fightable_monsters]}')
        #     character_object.set_task()     # From a list?
        #
        # character_object.prepare()   # Depending on character_object.task.type / including auto_equip
        # character_object.go()   # Depending on character_object.task.type
        # character_object.act()   # Depending on character_object.task.type


        # TODO update fight_target accordingly to task / consider cancelling task
        await character_object.update_fight_target()

        ### TARGET SET AND CHECK ###

        if await character_object.is_on_fight_mode():
            character_object.logger.debug(" ### BEGIN FIGHT ### ")
            # ASSESS IF CAN WIN

            # CHECK FOR BETTER EQUIPMENT SPECIFIC TO TARGET
            await character_object.equip_for_fight()

            # GO AND FIGHT
            await character_object.go_and_fight()
            character_object.logger.debug(" ### END FIGHT ### ")
            continue

        character_object.logger.debug(" ### BEGIN CRAFT PREPARATION ### ")

        if not character_object.craft_target:
            eligible_targets = await character_object.select_eligible_targets()
            if len(eligible_targets) > 0:
                # TODO consider setting a qty objective here? (to take into account availability in bank)
                await character_object.set_target(eligible_targets[0])
            else:
                # TODO Switch skill? / default skill
                # FIXME case where there is no eligible item to craft > revert to basic
                # Need to level up on the skill linked to the crafting one to be able to craft the high level items

                if character_object.craft_skill == 'cooking':
                    character_object.logger.warning("Need to be better at fishing")

                character_object.logger.warning("NO ELIGIBLE TARGET FOUND")
                continue

        ### PREPARATION ###

        craft_details = await character_object.prepare_to_craft()
        character_object.logger.info(f' crafting planned: {craft_details}')

        character_object.logger.debug(" ### END CRAFT PREPARATION ### ")

        ### GATHER, COLLECT AND CRAFT ###

        character_object.logger.debug(" ### BEGIN GATHERING AND CRAFTING ### ")

        # TODO if gather mission, equip accordingly
        await character_object.gather_and_collect(craft_details)
        await character_object.go_and_craft_item()

        # Reinitialize the target
        character_object.craft_target = None

        character_object.logger.debug(" ### END GATHERING AND CRAFTING ### ")

# personae
# ** WOOD CUTTER / PLANK CRAFTER
# ** FISHER / COOKING MASTER
# ** ORE MINER / BLACKSMITH
# ** WARRIOR
# + WEAPON CRAFTER
# + GEAR CRAFTER
# + JEWEL CRAFTER


async def main():
    async with aiohttp.ClientSession() as session:

        all_equipments = await get_equipments(session)
        all_equipments['excluded'] = {i: 'excluded' for i in EXCLUDED_ITEMS_CODES}

        all_items = {
            crafting_skill: await get_all_items(session, params={"craft_skill": crafting_skill})
            for crafting_skill in ['weaponcrafting', 'gearcrafting', 'jewelrycrafting', 'cooking', 'woodcutting', 'mining']
        }

        # LOCAL_BANK = await get_bank_items(session)

        characters_ = [
            Character(session=session, all_items=all_items, all_equipments=all_equipments, name='Kersh', craft_skill='weaponcrafting', max_fight_level=8, collect_skill='mining', skills=['weaponcrafting', 'mining', 'woodcutting']),  # 'weaponcrafting', 'mining'
            Character(session=session, all_items=all_items, all_equipments=all_equipments, name='Capu', craft_skill='gearcrafting', max_fight_level=8, collect_skill='woodcutting', skills=['gearcrafting', 'mining', 'woodcutting']),  # 'gearcrafting', 'woodcutting'
            Character(session=session, all_items=all_items, all_equipments=all_equipments, name='Brubu', craft_skill='mining', max_fight_level=12, collect_skill='mining', skills=['mining', 'woodcutting']),  # 'mining', 'mining'
            Character(session=session, all_items=all_items, all_equipments=all_equipments, name='Crabex', craft_skill='jewelrycrafting', max_fight_level=8, collect_skill='mining', skills=['jewelrycrafting', 'mining', 'woodcutting']),  # 'jewelrycrafting', 'woodcutting'
            Character(session=session, all_items=all_items, all_equipments=all_equipments, name='JeaGa', craft_skill='cooking', max_fight_level=8, collect_skill='fishing', skills=['cooking', 'woodcutting']),  # 'cooking', 'fishing'
        ]

        # Initialize all characters asynchronously
        await asyncio.gather(*[character.initialize() for character in characters_])

        # Start the bot for all characters
        await asyncio.gather(*[run_bot(character) for character in characters_])

if __name__ == '__main__':
    logging.info("Bot started.")
    asyncio.run(main())
    logging.info("Bot finished.")
