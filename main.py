import asyncio
from enum import Enum
import logging
from aiohttp import ClientSession
from aiohttp.client_exceptions import ClientConnectorError
import re
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
from typing import List, Optional
from datetime import datetime

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()
TOKEN = os.getenv('ARTIFACTSMMO_TOKEN')
if not TOKEN:
    raise ValueError("Le TOKEN n'est pas défini. Veuillez le définir dans les variables d'environnement.")


# Server url
SERVER = "https://api.artifactsmmo.com"
# Your account token (https://artifactsmmo.com/account)
EQUIPMENTS_SLOTS = ['weapon', 'shield', 'helmet', 'body_armor', 'leg_armor', 'boots', 'ring1', 'ring2',
                    'amulet', 'artifact1', 'artifact2', 'artifact3']
EQUIPMENTS_TYPES = ['weapon', 'shield', 'helmet', 'body_armor', 'leg_armor', 'boots', 'ring',
                    'amulet', 'artifact']
EXCLUDED_MONSTERS = ["cultist_acolyte", "cultist_emperor", "lich", "bat"]


class Craft(BaseModel):
    skill: str
    level: int
    items: list


class Effect(BaseModel):
    name: str
    value: int


class Ge(BaseModel):
    code: str
    stock: int
    sell_price: int
    buy_price: int
    max_quantity: int


class Drop(BaseModel):
    code: str
    rate: int
    min_quantity: int
    max_quantity: int


class Monster(BaseModel):
    name: str
    code: str
    level: int
    hp: int
    attack_fire: int
    attack_earth: int
    attack_water: int
    attack_air: int
    res_fire: int
    res_earth: int
    res_water: int
    res_air: int
    min_gold: int
    max_gold: int
    drops: list[Drop]

    def get_vulnerabilities(self) -> dict[str, int]:
        vulnerabilities = {
            'fire': self.res_fire,
            'earth': self.res_earth,
            'water': self.res_water,
            'air': self.res_air
        }

        # FIXME
        if self.code == 'bandit_lizard':
            vulnerabilities = {"water": 5}

        return vulnerabilities

    def is_event(self) -> bool:
        return self.code in ["demon", "bandit_lizard", "cultist_emperor", "rosenblood"]


class Resource(BaseModel):
    name: str
    code: str
    skill: str
    level: int
    drops: list[Drop]


class Item(BaseModel):
    name: str
    code: str
    level: int
    type: str
    subtype: str
    description: str
    effects: list[Effect] = Field(default_factory=list)
    craft: Optional[Craft] = None
    ge: Optional[Ge] = None

    def is_gatherable(self) -> bool:
        return self.craft is None and self.type == "resource" and self.subtype in ["mining", "woodcutting", "fishing"]

    def is_from_task(self) -> bool:
        return self.type == "resource" and self.subtype == "task"

    def is_rare_drop(self) -> bool:
        # TODO get it dynamic through the drop rate ?
        return self.code in ['topaz', 'emerald', 'ruby', 'sapphire', 'sap', 'magic_sap', 'diamond']

    def is_dropped(self) -> bool:
        return self.type == "resource" and self.subtype in ["mob", "food"]

    def is_crafted(self) -> bool:
        return self.craft is not None

    def is_equipment(self) -> bool:
        return self.type in EQUIPMENTS_TYPES

    def is_consumable(self) -> bool:
        craft = self.craft
        if craft is None:
            return False
        return self.craft.skill in ['cooking']

    def is_given(self) -> bool:
        return self.is_dropped() or self.is_from_task() or self.type in ["currency"] or self.code == "wooden_stick"

    def get_min_stock_qty(self) -> int:     # TODO get it as a post init attribute?
        if self.type == "ring":
            return 10
        return 5

    def is_protected_consumable(self) -> bool:
        # if any(["boost" in effect["name"] for effect in cooked_consumable['effects']])
        return self.is_consumable()

    def is_not_recyclable(self) -> bool:
        return self.craft and self.craft.skill in ['woodcutting', 'mining', 'cooking']

    def is_valid_equipment(self, character_level: int) -> bool:
        return character_level >= self.level

    def get_sell_price(self) -> int:
        # TODO adapt code to get updated price
        return self.ge.sell_price

    def is_protected(self) -> bool:
        return self.is_from_task() or self.is_rare_drop()

    def is_skill_compliant(self, skill_names: list[str]) -> bool:
        is_craft_skill_compliant = self.craft and self.craft.skill in skill_names
        is_gathering_skill_compliant = self.type == 'resource' and self.subtype in skill_names
        return is_craft_skill_compliant or is_gathering_skill_compliant


class TaskType(Enum):
    RESOURCES = "resources"
    MONSTERS = "monsters"
    ITEMS = "items"
    RECYCLE = "recycle"
    IDLE = "idle"


class Task(BaseModel):
    code: str = ""
    type: TaskType = TaskType.IDLE
    total: int = 0
    details: Item | Monster | Resource = None
    x: int = 0,
    y: int = 0,
    is_event: bool = False

    def is_craft_type(self):
        return self.type == TaskType.ITEMS

    def is_fight_type(self):
        return self.type == TaskType.MONSTERS

    def is_gather_type(self):
        return self.type == TaskType.RESOURCES

    # TODO get max_fight_level from character_infos?
    def is_feasible(self, character_infos: dict, max_fight_level: int) -> bool:
        if self.is_fight_type() and self.details.level <= max_fight_level:
            return True
        if self.is_craft_type():
            skill_level_key = f'{self.details.craft.skill}_level'
            skill_level = character_infos.get(skill_level_key)
            if skill_level is not None and self.details.level <= skill_level:
                return True
        if self.is_gather_type():
            skill_level_key = f'{self.details.skill}_level'
            skill_level = character_infos.get(skill_level_key)
            if skill_level is not None and self.details.level <= skill_level:
                return True
        return False


class Announcement(BaseModel):
    message: str
    created_at: datetime


class Status(BaseModel):
    status: str
    version: str
    max_level: int
    characters_online: int
    server_time: datetime
    announcements: List[Announcement]
    last_wipe: str
    next_wipe: str


class InventoryItem(BaseModel):
    slot: int
    code: str
    quantity: int


class CharacterInfos(BaseModel):
    name: str
    skin: str
    level: int
    xp: int
    max_xp: int
    achievements_points: int
    gold: int
    speed: int
    mining_level: int
    mining_xp: int
    mining_max_xp: int
    woodcutting_level: int
    woodcutting_xp: int
    woodcutting_max_xp: int
    fishing_level: int
    fishing_xp: int
    fishing_max_xp: int
    weaponcrafting_level: int
    weaponcrafting_xp: int
    weaponcrafting_max_xp: int
    gearcrafting_level: int
    gearcrafting_xp: int
    gearcrafting_max_xp: int
    jewelrycrafting_level: int
    jewelrycrafting_xp: int
    jewelrycrafting_max_xp: int
    cooking_level: int
    cooking_xp: int
    cooking_max_xp: int
    hp: int
    haste: int
    critical_strike: int
    stamina: int
    attack_fire: int
    attack_earth: int
    attack_water: int
    attack_air: int
    dmg_fire: int
    dmg_earth: int
    dmg_water: int
    dmg_air: int
    res_fire: int
    res_earth: int
    res_water: int
    res_air: int
    x: int
    y: int
    cooldown: int
    cooldown_expiration: datetime
    weapon_slot: str
    shield_slot: str
    helmet_slot: str
    body_armor_slot: str
    leg_armor_slot: str
    boots_slot: str
    ring1_slot: str
    ring2_slot: str
    amulet_slot: str
    artifact1_slot: str
    artifact2_slot: str
    artifact3_slot: str
    consumable1_slot: str
    consumable1_slot_quantity: int
    consumable2_slot: str
    consumable2_slot_quantity: int
    task: str
    task_type: str
    task_progress: int
    task_total: int
    inventory_max_items: int
    inventory: List[InventoryItem]


class Environment(BaseModel):
    items: dict[str, Item]
    monsters: dict[str, Monster]
    resource_locations: dict[str, Resource]
    maps: list[dict]
    status: Status
    crafted_items: list[Item] = None
    equipments: dict[str, Item] = None
    consumables: dict[str, Item] = None
    dropped_items: list[Item] = None

    # Allow arbitrary types if necessary (e.g., for custom types like Status)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(**data)
        self.crafted_items = self.get_crafted_items()
        self.equipments = self.get_equipments()
        self.consumables = self.get_consumables()
        self.dropped_items = self.get_dropped_items()

    def get_craftable_items(self, character_infos: dict) -> list[Item]:
        return [
            item
            for item in self.environment.crafted_items
            if item.level < character_infos[f'{item.craft.skill}_level']
        ]

    def get_item_dropping_monsters(self, item_code: str) -> list[tuple[Monster, int]]:
        item_dropping_monsters = []
        for monster in self.monsters.values():
            for item_details in monster.drops:
                if item_details.code == item_code:
                    item_dropping_monsters.append((monster, item_details.rate))
        item_dropping_monsters = sorted(item_dropping_monsters, key=lambda x: x[1], reverse=False)
        return item_dropping_monsters

    def get_item_dropping_monster(self, item_code: str) -> Optional[Monster]:
        item_dropping_monsters = self.get_item_dropping_monsters(item_code)
        if len(item_dropping_monsters) > 0:
            return item_dropping_monsters[0][0]
        return None

    def get_item_dropping_locations(self, item_code: str) -> list[tuple[Resource, int]]:
        item_dropping_locations = []
        for resource_location in self.resource_locations.values():
            for drop in resource_location.drops:
                if drop.code == item_code:
                    item_dropping_locations.append((resource_location, drop.rate))
        item_dropping_locations = sorted(item_dropping_locations, key=lambda x: x[1], reverse=False)
        return item_dropping_locations

    def get_item_dropping_location(self, item_code: str) -> Optional[Resource]:
        item_dropping_locations = self.get_item_dropping_locations(item_code)
        if len(item_dropping_locations) > 0:
            return item_dropping_locations[0][0]
        return None

    def get_item_dropping_max_rate(self, item_code: str) -> int:
        item_dropping_locations = self.get_item_dropping_locations(item_code)
        if len(item_dropping_locations) > 0:
            return item_dropping_locations[0][1]
        return 9999

    def get_dropped_items(self) -> list[Item]:
        return [
            item
            for item in self.items.values()
            if item.is_dropped()
        ]

    def get_crafted_items(self) -> list[Item]:
        return [
            item
            for _, item in self.items.items()
            if item.is_crafted()
        ]

    def get_equipments(self) -> dict[str, Item]:
        return {
            item_code: item
            for item_code, item in self.items.items()
            if item.is_equipment()
        }

    def get_consumables(self) -> dict[str, Item]:
        return {
            item.code: item
            for item in self.crafted_items
            if item.is_consumable()
        }


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
        session (ClientSession): The aiohttp session to use for requests.
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
            async with session.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=payload,
                    timeout=timeout
            ) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    # Handle rate limiting with exponential backoff
                    # backoff_time = min(2 ** attempt, 60)  # Cap the backoff time at 60 seconds
                    # logging.warning(f"Rate limited. Retrying after {backoff_time} seconds.")
                    # await asyncio.sleep(backoff_time)
                    # Handle rate limiting
                    retry_after = int(response.headers.get('Retry-After', '1'))
                    logging.warning(f"Rate limited. Retrying after {retry_after} seconds.")
                    await asyncio.sleep(retry_after)
                    continue  # Retry the request
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


async def get_status(session: ClientSession) -> dict:
    url = f"{SERVER}/"
    data = await make_request(session=session, method='GET', url=url)
    return data["data"] if data else {}


async def get_all_status(session: ClientSession) -> dict:
    status = await get_status(session)
    return {
        "max_level": status.get("max_level", 40),
        "server_time": status.get("server_time", "")
    }


def extract_cooldown_time(message):
    """Extracts cooldown time from the message using regex."""
    match = re.search(r"Character in cooldown: (\d+)", message)
    if match:
        return int(match.group(1))
    return None


def get_items_list_by_type(_items: dict[str, Item], _item_type: str) -> list[Item]:
    return [
        item
        for _, item in _items.items()
        if item.type == _item_type
    ]


def get_equipments_by_type(_items: dict[str, Item]) -> dict[str, list[Item]]:
    return {
        equipment_type: get_items_list_by_type(_items, equipment_type)
        for equipment_type in EQUIPMENTS_TYPES
    }


def is_equipment_better(equipment_a: Item, equipment_b: Item) -> bool:
    """
    Returns True if equipment_a is strictly better than equipment_b.
    """
    # Parse effects into dictionaries
    effects_a = {effect.name: effect.value for effect in equipment_a.effects}
    effects_b = {effect.name: effect.value for effect in equipment_b.effects}

    # Get the union of all effect names
    all_effects = set(effects_a.keys()) | set(effects_b.keys())

    better_or_equal_in_all = True
    strictly_better_in_at_least_one = False

    for effect in all_effects:
        value_a = effects_a.get(effect, 0)
        value_b = effects_b.get(effect, 0)
        if effect in ["fishing", "woodcutting", "mining"]:
            # Best effect is negative (cooldown reduction)
            value_a = -value_a
            value_b = -value_b
        if value_a < value_b:
            better_or_equal_in_all = False
            break
        elif value_a > value_b:
            strictly_better_in_at_least_one = True

    return better_or_equal_in_all and strictly_better_in_at_least_one


def identify_obsolete_equipments(equipment_groups: dict[str, list[Item]]) -> list[Item]:
    obsolete_equipments = []
    for equipment_type, equipments in equipment_groups.items():
        # Sort equipments by level and effects
        equipments_sorted = sorted(equipments, key=lambda e: (
            -sum(effect.value for effect in e.effects)  # Higher total effects first
        ))
        # Keep track of the best equipments seen so far
        best_equipments = []
        for equipment in equipments_sorted:
            is_obsolete = False
            for best in best_equipments:
                if is_equipment_better(best, equipment):
                    is_obsolete = True
                    break
            if not is_obsolete:
                best_equipments.append(equipment)
            else:
                obsolete_equipments.append(equipment)
    return obsolete_equipments


async def get_obsolete_equipments(session: ClientSession, _environment: Environment) -> dict[str, Item]:
    _items = _environment.items
    equipment_groups = get_equipments_by_type(_items)

    # Récupérer les quantités des items une seule fois
    total_quantities = await get_all_items_quantities(session)

    # Filtrer les équipements existants sur la carte
    map_equipments = {}
    for equipment_type, equipments in equipment_groups.items():
        filtered_equipments = []
        for equipment in equipments:
            # Seulement si la quantité minimale est disponible
            if total_quantities.get(equipment.code, 0) >= equipment.get_min_stock_qty():
                filtered_equipments.append(equipment)
        map_equipments[equipment_type] = filtered_equipments

    obsolete_equipments = identify_obsolete_equipments(map_equipments)
    return {equipment.code: equipment for equipment in obsolete_equipments}


async def get_all_items_quantities(session: ClientSession) -> dict[str, int]:
    # Start fetching bank items and character infos concurrently
    bank_items_task = asyncio.create_task(get_bank_items(session))
    characters_infos_task = asyncio.create_task(get_my_characters(session))

    # Wait for both tasks to complete
    bank_items = await bank_items_task
    characters_infos = await characters_infos_task

    total_quantities = {}

    # Add bank items to total quantities
    for code, qty in bank_items.items():
        total_quantities[code] = total_quantities.get(code, 0) + qty

    # Process characters' inventories and equipments
    for character in characters_infos:
        # Inventory items
        for slot in character.get('inventory', []):
            code = slot['code']
            qty = slot['quantity']
            if code:
                total_quantities[code] = total_quantities.get(code, 0) + qty

        # Equipped items
        for equipment_slot in EQUIPMENTS_SLOTS:
            code = character.get(f'{equipment_slot}_slot', '')
            if code:
                total_quantities[code] = total_quantities.get(code, 0) + 1

    return total_quantities


async def needs_stock(session: ClientSession, _item: Item, total_quantities: dict[str, int] = None) -> bool:
    if total_quantities is None:
        total_quantities = await get_all_items_quantities(session)
    return await get_all_map_item_qty(session, _item, total_quantities) < _item.get_min_stock_qty()


async def get_bank_items(session: ClientSession, params: dict = None) -> dict:
    if params is None:
        params = {"size": 100}
    else:
        params.setdefault('size', 100)

    url = f"{SERVER}/my/bank/items/"
    all_items = {}
    page = 1

    while True:
        params['page'] = page
        data = await make_request(session=session, method='GET', url=url, params=params)
        if data and data["data"]:
            for item in data["data"]:
                code = item['code']
                qty = item['quantity']
                all_items[code] = all_items.get(code, 0) + qty
            page += 1
            # Check if we've reached the last page
            if len(data["data"]) < params['size']:
                break
        else:
            break

    return all_items


async def get_bank_item_qty(session: ClientSession, _item_code: str) -> int:
    res = await get_bank_items(session=session, params={"item_code": _item_code})
    return res.get(_item_code, 0)


def get_craft_recipee(_item: Item) -> dict[str, int]:
    if _item.craft is not None:
        return {m['code']: m['quantity'] for m in _item.craft.items}
    logging.error(f'This material {_item.code} is not craftable')
    return {_item.code: 1}


async def get_all_events(session: ClientSession, params: dict = None) -> list[dict]:
    """
    Retrieves all maps from the API.
    Returns a list of maps with their details.
    """
    if params is None:
        params = {}
    url = f"{SERVER}/events"
    data = await make_request(session=session, method='GET', url=url, params=params)
    return data["data"] if data else []


async def get_all_maps(session: ClientSession, params: dict = None) -> list[dict]:
    """
    Retrieves all maps from the API.
    Returns a list of maps with their details.
    """
    if params is None:
        params = {}
    url = f"{SERVER}/maps/"
    data = await make_request(session=session, method='GET', url=url, params=params)
    return data["data"] if data else []


async def get_place_name(session: ClientSession, x: int, y: int) -> str:
    url = f"{SERVER}/maps/{x}/{y}"
    data = await make_request(session=session, method='GET', url=url)
    return data["data"]["content"]["code"]


async def get_my_characters(session: ClientSession) -> list:
    url = f"{SERVER}/my/characters"
    data = await make_request(session=session, method='GET', url=url)
    return data["data"] if data else []


async def get_all_map_item_qty(session: ClientSession, _item: Item, total_quantities: dict[str, int] = None) -> int:
    if total_quantities is None:
        total_quantities = await get_all_items_quantities(session)
    return total_quantities.get(_item.code, 0)


def get_min_stock_qty(item: Item) -> int:
    if item.type == "ring":
        return 10
    return 5


async def get_all_resources(session: ClientSession, params: dict = None) -> dict[str, Resource]:
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
    return {elt["code"]: Resource(**elt) for elt in data["data"]} if data else {}


async def get_all_monsters(session: ClientSession, params: dict = None) -> dict[str, Monster]:
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
    return {elt["code"]: Monster(**elt) for elt in data["data"]} if data else {}


async def get_all_items(session: ClientSession, params: dict = None) -> dict[str, Item]:
    if params is None:
        params = {}
    items = []
    page = 1
    while True:
        params['page'] = page
        data = await make_request(
            session=session,
            method='GET',
            url=f"{SERVER}/items/",
            params=params
        )
        if data and data["data"]:
            items.extend([Item(**item_data) for item_data in data["data"]])
            page += 1
        else:
            break
    return {item.code: item for item in items}


async def get_dropping_resource_locations(session: ClientSession, _material_code: str) -> Resource:
    resources_locations = [
        loc for loc in (await get_all_resources(session)).values()
        if _material_code in [x.code for x in loc.drops]
    ]
    if resources_locations:
        best_location = min(resources_locations, key=lambda x: x.level)
        return best_location

    # FIXME Return a default or error location if no match is found
    return Resource(**{
        "name": "Unknown Location",
        "code": "unknown_location",
        "skill": 'N/A',
        "level": 0,
        "drops": []
    })


def select_best_support_equipment(
        current_item: Optional[Item],
        equipment_list: list[Item],
        vulnerabilities: dict[str, int],
        best_weapon: Item
) -> Optional[Item]:
    if not equipment_list:
        return current_item
    if not current_item:
        current_item = equipment_list[0]

    # Extract weapon elements
    weapon_effects = {effect.name: effect.value for effect in best_weapon.effects}
    weapon_elements = set()
    for effect_name in weapon_effects:
        if effect_name.startswith('attack_'):
            element = effect_name.replace('attack_', '')
            weapon_elements.add(element)

    # TODO _effects to change to list[Effect]
    def calculate_support_score(_effects: dict, _vulnerabilities: dict[str, int], _weapon_elements: set) -> float:
        score = 0.0
        for _effect_name, effect_value in _effects.items():
            if _effect_name.startswith("dmg_"):
                _element = _effect_name.replace("dmg_", "")
                if _element in _weapon_elements:
                    resistance = _vulnerabilities.get(_element, 0)
                    # Boost score if the support equipment's damage effect matches the weapon's attack element
                    if resistance < 0:
                        score += effect_value * 4 * (1 + abs(resistance) / 100 * 5)
                    elif resistance > 0:
                        score += effect_value * 2 * (1 - resistance / 100 * 2)
                    else:
                        score += effect_value * 1.0
                else:
                    # Lesser weight if the element doesn't match the weapon's attack element
                    score += effect_value * 0.5
            elif _effect_name.startswith("res_"):
                _element = _effect_name.replace("res_", "")
                resistance = _vulnerabilities.get(_element, 0)
                if resistance < 0:
                    score += effect_value * 3 * (1 + abs(resistance) / 100 * 5)
                elif resistance > 0:
                    score += effect_value * 1.5 * (1 - resistance / 100 * 2)
                else:
                    score += effect_value * 1.0
            elif _effect_name == "hp":
                score += effect_value * 0.2
            elif _effect_name == "defense":
                score += effect_value * 0.5
            else:
                score += effect_value
        return score

    best_item = current_item
    best_score = calculate_support_score(
        {effect.name: effect.value for effect in current_item.effects},
        vulnerabilities,
        weapon_elements
    )
    logging.debug(f"Current equipment: {current_item.code}, Score: {best_score}")

    for item in equipment_list:
        item_effects = {effect.name: effect.value for effect in item.effects}
        item_score = calculate_support_score(item_effects, vulnerabilities, weapon_elements)
        logging.debug(f"Evaluating equipment: {item.code}, Effects: {item_effects}, Score: {item_score}")
        if item_score > best_score:
            best_item = item
            best_score = item_score
            logging.info(f"Best equipment updated to {best_item.code} with score {best_score}")

    return best_item


async def select_best_equipment_set(
        current_equipments: dict[str, Item],
        sorted_valid_equipments: dict[str, list[Item]],
        vulnerabilities: dict
) -> dict[str, Item]:
    selected_equipments = {}

    # Sélectionner la meilleure arme (code existant)
    best_weapon = select_best_weapon(
        current_equipments.get('weapon', None),       # FIXME None is not a valid Item
        sorted_valid_equipments.get('weapon', []),
        vulnerabilities
    )
    selected_equipments['weapon'] = best_weapon

    # Sélectionner les meilleurs équipements de support en fonction des vulnérabilités
    for slot in EQUIPMENTS_SLOTS:
        if slot == 'weapon':
            continue
        current_item = current_equipments.get(slot, {})
        equipment_list = sorted_valid_equipments.get(slot, [])
        best_item = select_best_support_equipment(current_item, equipment_list, vulnerabilities, best_weapon)
        selected_equipments[slot] = best_item

    return selected_equipments


def calculate_weapon_score(_effects: dict, _vulnerabilities: dict[str, int]) -> float:
    score = 0.0
    for effect_name, effect_value in _effects.items():
        if effect_name.startswith("attack_"):
            element = effect_name.replace("attack_", "")
            resistance = _vulnerabilities.get(element, 0)
            if resistance < 0:
                # Monster is vulnerable, increase the score significantly
                score += effect_value * 4 * (1 + abs(resistance) / 100 * 5)
            elif resistance > 0:
                # Monster has resistance, decrease the score significantly
                score += effect_value * 4 * (1 - resistance / 100 * 2)
            else:
                # Neutral element, give lower weight
                score += effect_value * 2
        elif effect_name.startswith("dmg_"):
            # Peut être pris en compte si pertinent
            pass
        elif effect_name == "hp":
            score += effect_value * 0.25
        elif effect_name == "defense":
            score += effect_value * 0.5
        else:
            score += effect_value
    return score


def select_best_weapon(current_weapon: Item, weapon_list: list[Item], vulnerabilities: dict[str, int]) -> Item:
    """
    Selects the best weapon based on vulnerabilities.
    """
    if not weapon_list:
        return current_weapon
    if not current_weapon:
        current_weapon = weapon_list[0]

    best_weapon = current_weapon
    best_score = calculate_weapon_score(
        {effect.name: effect.value for effect in current_weapon.effects},
        vulnerabilities
    )
    logging.debug(f"Current weapon: {current_weapon.code}, Score: {best_score}")

    for weapon in weapon_list:
        weapon_effects = {effect.name: effect.value for effect in weapon.effects}
        weapon_score = calculate_weapon_score(weapon_effects, vulnerabilities)
        logging.debug(f"Evaluating weapon: {weapon.code}, Effects: {weapon_effects}, Score: {weapon_score}")
        if weapon_score > best_score:
            best_weapon = weapon
            best_score = weapon_score
            logging.info(f"Best weapon updated to {best_weapon.code} with score {best_score}")

    return best_weapon


async def select_best_equipment(
        equipment1: Item,
        sorted_valid_equipments: list[Item],
        vulnerabilities: dict[str, int]
) -> Item:
    """
    Selects the best equipment based on monster vulnerability and equipment effects.

    :param equipment1: The currently equipped item information (or empty dict if none equipped).
    :param sorted_valid_equipments: A list of valid equipment items sorted by level.
    :param vulnerabilities: The monster's elemental vulnerabilities (e.g., 'fire', 'water').
    :return: The selected best equipment.
    """
    if len(sorted_valid_equipments) == 0:
        return equipment1
    if not equipment1:
        return sorted_valid_equipments[0]

    # TODO use list[Effect] for _effects
    def calculate_effect_score(_effects: dict, _vulnerabilities: dict[str, int]) -> int:
        """
        Calculate the equipment score based on the effects, giving more weight to the monster's vulnerability.
        """
        score = 0
        total_attack = 0  # Used when vulnerabilities are equal to prioritize damage
        all_vulnerabilities_equal = len(_vulnerabilities) == 4
        for effect_name, effect_value in _effects.items():

            if effect_name.startswith("attack_"):
                element = effect_name.replace("attack_", "")
                total_attack += effect_value
                if all_vulnerabilities_equal:
                    score += effect_value * 4  # High weight for attack effects
                else:
                    if element in _vulnerabilities:
                        percentage = _vulnerabilities[element]
                        score += effect_value * 4 * (percentage / 100)  # Weight by vulnerability percentage
                    else:
                        score += effect_value  # Lesser weight for non-vulnerable elements
            elif effect_name.startswith("dmg_"):
                element = effect_name.replace("dmg_", "")
                if all_vulnerabilities_equal:
                    score += effect_value * 3
                else:
                    if element in _vulnerabilities:
                        percentage = _vulnerabilities[element]
                        score += effect_value * 3 * (percentage / 100)
            elif effect_name == "hp":
                score += effect_value * 0.25
            elif effect_name == "defense":
                score += effect_value * 0.5
            else:
                score += effect_value  # Default score for other effects

        # If vulnerabilities are equal, prioritize equipment with the highest total attack
        if all_vulnerabilities_equal:
            score += total_attack * 2  # Additional weight for total attack
        return score

    best_equipment = equipment1
    best_score = calculate_effect_score(
        {effect.name: effect.value for effect in equipment1.effects},
        vulnerabilities
    )

    for equipment2 in sorted_valid_equipments:
        equipment2_effects = {effect.name: effect.value for effect in equipment2.effects}
        equipment2_score = calculate_effect_score(equipment2_effects, vulnerabilities)

        # Compare scores and select the best
        if equipment2_score > best_score:
            best_equipment = equipment2
            best_score = equipment2_score
            logging.debug(f"Best equipment updated to {best_equipment.code} with score {best_score}'")

    return best_equipment


class Character(BaseModel):
    session: ClientSession
    environment: Environment
    obsolete_equipments: dict[str, dict]
    name: str
    skills: list[str]
    max_fight_level: int = 0
    stock_qty_objective: int = 500
    task: Task = Field(default_factory=Task)
    gatherable_resources: list[Item] = Field(default_factory=list)
    craftable_items: list[Item] = Field(default_factory=list)
    fightable_monsters: list[Monster] = Field(default_factory=list)
    fightable_materials: list[Item] = Field(default_factory=list)
    objectives: list[Item] = Field(default_factory=list)
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
        infos = await self.get_infos()
        await asyncio.gather(
            self.set_gatherable_resources(infos),
            self.set_craftable_items(infos),
            self.set_fightable_monsters()
        )
        await self.set_objectives()

    async def set_gatherable_resources(self, infos: dict):
        """
        Fetch and set gatherable resources based on the character's collect skill and level
        """
        gatherable_resources = []

        for item_code, item in self.environment.items.items():
            if item.is_gatherable() and item.level <= infos[f'{item.subtype}_level']:
                if self.environment.get_item_dropping_max_rate(item_code) <= 100:
                    gatherable_resources.append(item)

        self.gatherable_resources = gatherable_resources
        self._logger.info(f"Gatherable resources for {self.name}: {[r.code for r in self.gatherable_resources]}")

    async def set_craftable_items(self, character_infos: dict):
        """
        Fetch and set craftable items based on the character's craft skill and level
        """
        skill_craftable_items = self.environment.get_craftable_items(character_infos)
        craftable_items = skill_craftable_items[::-1]
        # TODO exclude protected items (such as the one using jasper_crystal)

        excluded_item_codes = []
        for item_code in ['strangold', 'obsidian', 'magical_plank']:
            for material_code in get_craft_recipee(self.environment.items[item_code]):
                if material_code in ['strange_ore', 'piece_of_obsidian', 'magic_wood']:
                    if await get_bank_item_qty(self.session, material_code) < self.stock_qty_objective:
                        excluded_item_codes.append(item_code)
                        continue

        filtered_craftable_items = [
            item
            for item in craftable_items
            if item.code not in excluded_item_codes
        ]
        self.craftable_items = filtered_craftable_items
        self._logger.warning(f"Craftable items for {self.name}: {[i.code for i in self.craftable_items]}")

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
            return item.code in [i.code for i in self.gatherable_resources]
        elif item.is_dropped():
            return item.code in [i.code for i in self.fightable_materials]
        elif item.is_crafted():
            return all([
                await self.can_be_home_made(self.environment.items[material_code])
                for material_code in list(get_craft_recipee(item).keys())
            ])
        elif item.is_given():
            return False
        else:
            self._logger.warning(f' {item.code} to categorize')

    async def set_objectives(self):
        # Out of craftable items, which one can be handled autonomously
        objectives = [
            item
            for item in self.craftable_items
            if await self.can_be_home_made(item) and await self.does_item_provide_xp(item)
        ]

        need_level_up_craftable_items = [
            item
            for item in self.craftable_items
            if not await self.can_be_home_made(item) and await self.does_item_provide_xp(item)
        ]
        self._logger.info(f' NEED LEVELING UP OR SPECIAL MATERIALS TO CRAFT: {[o.code for o in need_level_up_craftable_items]}')

        # Sort items by their rarity in the bank (to prioritize items that are rarer)
        items2bank_qty = {
            craftable_item.code: await get_bank_item_qty(self.session, craftable_item.code)
            for craftable_item in objectives
        }

        item_objectives = sorted(objectives, key=lambda x: items2bank_qty.get(x.code, 0), reverse=False)

        resource_objectives = [
            resource
            for resource in self.gatherable_resources
            if await self.can_be_home_made(resource) and await self.does_item_provide_xp(resource) and resource.code not in ["magic_tree", "demon"]
        ]

        self._logger.info(f' RESOURCES OBJECTIVES: {[o.code for o in resource_objectives]}')

        item_objectives.extend(resource_objectives[::-1])

        # Filter according to defined craft skills
        item_objectives = [
            item
            for item in item_objectives
            if item.is_skill_compliant(self.skills)
        ]

        fight_objectives = [
            monster
            for monster in self.fightable_monsters
            if await self.can_be_vanquished(monster) and await self.does_fight_provide_xp(monster)
        ]

        self.objectives = item_objectives
        self.fight_objectives = fight_objectives[::-1]
        self._logger.info(f' CAN GET XP WITH: {[o.code for o in self.objectives]}')

    async def can_be_vanquished(self, monster: Monster) -> bool:
        return monster.code in [m.code for m in self.fightable_monsters]

    async def get_infos(self) -> dict:
        url = f"{SERVER}/characters/{self.name}"
        data = await make_request(session=self.session, method='GET', url=url)
        if data:
            self._logger.debug("Fetched character info successfully.")
        else:
            self._logger.error("Failed to fetch character info.")
        return data["data"] if data else {}

    async def get_inventory_qty(self) -> dict[str, int]:
        url = f"{SERVER}/characters/{self.name}"
        data = await make_request(session=self.session, method='GET', url=url)
        if data:
            self._logger.debug("Fetched character inventory successfully.")
        else:
            self._logger.error("Failed to fetch character inventory.")
        character_infos = data["data"] if data else {}
        inventory_slots = [slot for slot in character_infos["inventory"]]
        return {
            slot["code"]: slot["quantity"]
            for slot in inventory_slots
            if slot["code"] != ""
        }

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
                    _items_details['money'] = gold_amount

            self._logger.debug(f'depositing at bank: {_items_details} ...')
            for item_code, item_qty in _items_details.items():
                cooldown_ = await self.bank_deposit(item_code, item_qty)
                await asyncio.sleep(cooldown_)

    async def get_gold_amount(self) -> int:
        infos = await self.get_infos()
        return infos['gold']

    async def withdraw_items_from_bank(self, _items_details: dict[str, int]):
        # Move to the bank
        await self.move_to_bank()

        self._logger.debug(f'collecting at bank: {_items_details} ...')
        for item_code, item_qty in _items_details.items():
            # FIXME check to be done beforehand?
            nb_free_slots = await self.get_inventory_free_slots_nb()
            cooldown_ = await self.bank_withdraw(item_code, min(item_qty, nb_free_slots))
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
        inventory_items = await self.get_inventory_items()
        gathered_qty = inventory_items.get(_material_code, 0)
        inventory_not_full = sum(inventory_items.values()) < infos.get('inventory_max_items', 100)
        return inventory_not_full and gathered_qty < target_qty

    async def is_up_to_fight(self):
        got_enough_consumables = await self.got_enough_consumables(-1)
        inventory_not_full = await self.is_inventory_not_full()
        is_goal_completed = await self.is_goal_completed()
        is_not_at_spawn_place = not await self.is_at_spawn_place()
        return got_enough_consumables and inventory_not_full and not is_goal_completed and is_not_at_spawn_place

    async def is_at_spawn_place(self) -> bool:
        current_location = await self.get_current_location()
        if current_location == (0, 0):
            self._logger.debug(f'is already at spawn place - likely killed by a monster')
            return True
        return False

    async def go_and_fight_to_collect(self, material_code: str, quantity_to_get: int):
        # Identify the monster and go there IF fightable
        if not await self.is_fightable(material_code):
            return

        monster = self.environment.get_item_dropping_monster(material_code)
        if not monster:
            return
        await self.move_to_monster(monster.code)

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
        if _skill == 'mob':
            return await self.get_level()
        elif _skill == 'food':
            _skill = "cooking"
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
            self._logger.error(f'failed to complete task.')

    async def cancel_task(self):
        url = f"{SERVER}/my/{self.name}/action/task/cancel"
        data = await make_request(session=self.session, method='POST', url=url)
        if data:
            _cooldown = data["data"]["cooldown"]["total_seconds"]
            await asyncio.sleep(_cooldown)
        else:
            self._logger.error(f'failed to cancel task.')

    async def accept_new_task(self):
        url = f"{SERVER}/my/{self.name}/action/task/new"
        data = await make_request(session=self.session, method='POST', url=url)
        if data:
            _cooldown = data["data"]["cooldown"]["total_seconds"]
            await asyncio.sleep(_cooldown)
        else:
            self._logger.error(f'failed to get new task.')

    async def exchange_tasks_coins(self):
        url = f"{SERVER}/my/{self.name}/action/task/exchange"
        data = await make_request(session=self.session, method='POST', url=url)
        if data:
            _cooldown = data["data"]["cooldown"]["total_seconds"]
            await asyncio.sleep(_cooldown)
        else:
            self._logger.error(f'failed to get new task.')

    async def move(self, x, y) -> int:
        current_location = await self.get_current_location()
        if current_location == (x, y):
            self._logger.debug(f'is already at the location ({x}, {y})')
            return 0

        url = f"{SERVER}/my/{self.name}/action/move"
        payload = {"x": x, "y": y}

        data = await make_request(session=self.session, method='POST', url=url, payload=payload)
        if data:
            _cooldown = data["data"]["cooldown"]["total_seconds"]
            self._logger.debug(f'moved to ({await get_place_name(self.session, x, y)}). Cooldown: {_cooldown} seconds')
            return _cooldown
        else:
            self._logger.error(f'failed to move to ({x}, {y})')
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
            self._logger.warning(f'No resource {content_code} on this map')
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

    async def get_nb_craftable_items(self, _item: Item, from_inventory: bool = False) -> int:

        craft_recipee = get_craft_recipee(_item)
        self._logger.debug(f' recipee for {_item.code} is {craft_recipee}')
        total_nb_materials = sum([qty for _, qty in craft_recipee.items()])
        nb_craftable_items = await self.get_inventory_max_size() // total_nb_materials

        if from_inventory:  # Taking into account drops > update qty available in inventory
            for material_code, qty in craft_recipee.items():
                material_inventory_qty = await self.get_inventory_quantity(material_code)
                nb_craftable_items = min(material_inventory_qty//qty, nb_craftable_items)

        self._logger.debug(f' nb of craftable items {"from inventory" if from_inventory else ""} for {_item.code} is {nb_craftable_items}')

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
        if 'jasper_crystal' in get_craft_recipee(_item).keys() or 'magical_cure' in get_craft_recipee(_item).keys():
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
        return resource_code in [item.code for item in self.gatherable_resources]

    async def is_fightable(self, material_code) -> bool:
        return material_code in [item.code for item in self.fightable_materials]

    async def is_craftable(self, item_code) -> bool:
        return item_code in [item.code for item in self.craftable_items]

    async def is_collectable_at_bank(self, item_code, quantity) -> bool:
        qty_at_bank = await get_bank_item_qty(self.session, item_code)
        return qty_at_bank > 3 * quantity

    async def move_to_workshop(self):
        # get the skill out of item
        skill_name = self.task.details.craft.skill
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

    async def get_equipment_code(self, _equipment_slot: str) -> str:
        infos = await self.get_infos()
        key = f'{_equipment_slot}_slot'
        if key in infos:
            return infos[key]
        else:
            self._logger.warning(f"Equipment slot '{key}' not found in character info.")
            return ""

    async def get_current_equipments(self) -> dict[str, Item]:
        infos = await self.get_infos()
        return {
            equipment_slot: self.environment.items[infos[f'{equipment_slot}_slot']]
            if infos[f'{equipment_slot}_slot'] != "" else None
            for equipment_slot in EQUIPMENTS_SLOTS
        }

    async def go_and_equip(self, _equipment_slot: str, _equipment_code: str):
        current_equipment_code = await self.get_equipment_code(_equipment_slot)
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
        # Get the list of equipments from the bank for the 'weapon' slot
        bank_equipments = await self.get_bank_equipments_for_slot('weapon')

        # Filter equipments that have effects matching the gathering skill
        valid_equipments = []
        for equipment in bank_equipments:
            for effect in equipment.get('effects', []):
                if effect['name'] == gathering_skill:
                    valid_equipments.append(equipment)
                    break  # No need to check other effects

        if not valid_equipments:
            self._logger.info(f"No valid equipment found for gathering skill {gathering_skill}")
            return

        # Ensure equipments are valid for the character's level
        valid_equipments = [
            equipment for equipment in valid_equipments
            if equipment.is_valid_equipment(await self.get_level())
        ]

        if not valid_equipments:
            self._logger.info(f"No valid equipment found for gathering skill {gathering_skill} at character's level")
            return

        # Sort equipments based on the effect value for the gathering skill
        def get_effect_value(_equipment, effect_name):
            for _effect in _equipment.get('effects', []):
                if _effect['name'] == effect_name:
                    return _effect['value']
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
            await self.equip_for_gathering(self.environment.items[item_code]["subtype"])
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

    async def does_item_provide_xp(self, _item: Item) -> bool:
        """
        Determine if a given item will provide XP when crafted, gathered, or fished.
        This depends on the item's level compared to the character's skill level.
        """
        # Check if the item is craftable
        if _item.craft:
            skill_name = _item.craft.skill
        else:
            # Non-craftable items are checked by subtype (gathering, fishing, etc.)
            skill_name = _item.subtype

        # If there's no valid skill (item can't be crafted or gathered), it provides no XP
        if not skill_name:
            return False

        # Get the character's skill level for the relevant skill (crafting, gathering, etc.)
        skill_level = await self.get_skill_level(skill_name)

        # Item level must be within range of skill level to provide XP (e.g., within 10 levels)
        item_level = _item.level

        # Example threshold: if item is within 10 levels of the character's skill level, it gives XP
        return item_level >= (skill_level - 10) and skill_level < self.environment.status.max_level

    async def does_fight_provide_xp(self, monster: Monster) -> bool:
        character_level = await self.get_level()
        return monster.level >= (character_level - 10) and character_level < self.environment.status.max_level

    async def select_eligible_targets(self) -> list[str]:
        """
        Select eligible targets (tasks) for the character, ensuring that they will gain XP from them.
        Includes crafting, gathering, and fishing tasks. Returns a list of eligible items that provide XP.
        """
        self._logger.debug(f"Selecting eligible targets for {self.name}")

        # Filter items that are valid and provide XP (this includes craftable, gatherable, and fishable items)
        valid_craftable_items = self.gatherable_resources

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

    async def is_valid_equipment(self, _equipment: Item) -> bool:
        character_level = await self.get_level()
        return character_level >= _equipment.level

    async def equip_for_fight(self, _monster: Monster = None):

        if _monster is None:
            _monster = self.task.details

        # Identify vulnerability
        vulnerabilities = _monster.get_vulnerabilities()
        self._logger.debug(f' monster {_monster.code} vulnerabilities are {vulnerabilities}')

        current_equipments = await self.get_current_equipments()
        sorted_valid_bank_equipments = await self.get_sorted_valid_bank_equipments()

        selected_equipments = await select_best_equipment_set(current_equipments, sorted_valid_bank_equipments, vulnerabilities)
        for equipment_slot, equipment in selected_equipments.items():
            if equipment is not None:
                await self.go_and_equip(equipment_slot, equipment.code)
            else:
                self._logger.debug(f"No equipment selected for slot {equipment_slot}")

        # Manage consumables
        await self.equip_best_consumables()

    async def get_eligible_bank_consumables(self) -> list[Item]:
        return [
            consumable
            for consumable in self.environment.consumables.values()
            if await get_bank_item_qty(self.session, consumable.code) > 0 and consumable.level <= await self.get_level()
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
        This function avoids unequipping if the consumable is already equipped and fully stocked.
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
            current_code = character_infos.get(f"{slot}_slot", "")
            current_qty = character_infos.get(f"{slot}_slot_quantity", 0)
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
        currently_equipped_consumables = [character_infos['consumable1_slot'], character_infos['consumable2_slot']]
        self._logger.debug(f' Currently equipped consumables: {currently_equipped_consumables}')
        return [self.environment.consumables[c] if c else None for c in currently_equipped_consumables]

    async def equip_best_equipment(self, _equipment_slot: str, vulnerabilities: dict[str, int]):
        available_equipments = await self.get_bank_equipments_for_slot(_equipment_slot)
        self._logger.debug(f'available equipment at bank {[e.code for e in available_equipments]}')
        sorted_valid_equipments = sorted([
            equipment
            for equipment in available_equipments
            if equipment.is_valid_equipment(await self.get_level())
        ], key=lambda x: x.level, reverse=True)

        self._logger.debug(f'may be equipped with {[e.code for e in sorted_valid_equipments]}')

        current_equipment_code = await self.get_equipment_code(_equipment_slot)
        if len(sorted_valid_equipments) == 0:
            return
        current_equipment_infos = self.environment.equipments.get(current_equipment_code, {})
        new_equipment_details = await select_best_equipment(current_equipment_infos, sorted_valid_equipments, vulnerabilities)
        self._logger.debug(f' has been assigned {new_equipment_details.get("code", "")} for slot {_equipment_slot} instead of {current_equipment_infos.get("code", "")}')
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
        bank_items = await get_bank_items(self.session)
        return [
            self.environment.equipments[item_code]
            for item_code in bank_items.keys()
            # 'ring' in 'ring1'
            if self.environment.equipments.get(item_code, None) is not None and self.environment.equipments[item_code].type in equipment_slot
        ]

    async def get_sorted_valid_bank_equipments(self) -> dict[str, list[Item]]:
        bank_items = await get_bank_items(self.session)     # TODO use Item ?
        bank_equipments = {}
        for equipment_slot in EQUIPMENTS_SLOTS:
            slot_equipments = [
                self.environment.equipments[item_code]
                for item_code in bank_items.keys()
                # 'ring' in 'ring1'
                if self.environment.equipments.get(item_code, None) is not None and self.environment.equipments[item_code].type in equipment_slot
            ]
            sorted_slot_equipments = sorted([
                equipment
                for equipment in slot_equipments
                if equipment.is_valid_equipment(await self.get_level())
            ], key=lambda x: x.level, reverse=True)
            bank_equipments[equipment_slot] = sorted_slot_equipments
        return bank_equipments

    async def manage_task(self):
        game_task = await self.get_game_task()
        # if task completed (or none assigned yet), go to get rewards and renew task

        nb_tasks_coins = await get_bank_item_qty(self.session, "tasks_coin")
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
                await self.complete_task()
            # ask for new task
            await self.accept_new_task()
            game_task = await self.get_game_task()

        # If task is too difficult, change
        while game_task.code in EXCLUDED_MONSTERS:
            if await self.get_inventory_quantity("tasks_coin") == 0:
                await self.withdraw_items_from_bank({"tasks_coin": 1})
            await self.move_to_task_master()
            await self.cancel_task()
            await self.accept_new_task()
            game_task = await self.get_game_task()

    async def is_task_completed(self) -> bool:

        # TODO can also be a personal task (amount of collectibles) - allow for some materials to be collected by others

        character_infos = await self.get_infos()
        return character_infos.get("task_progress", "A") == character_infos.get("task_total", "B")

    async def is_goal_completed(self) -> bool:
        return await self.is_event_still_on() or await self.is_task_completed()

    async def get_unnecessary_equipments(self) -> dict[str, int]:
        recycle_details = {}
        # Apply only on those that can be crafted again
        for item_code, item_qty in (await get_bank_items(self.session)).items():
            # No recycling for planks and ores and cooking
            item = self.environment.items[item_code]
            if item.is_not_recyclable():
                continue
            min_qty = item.get_min_stock_qty()
            if item.is_crafted() and item_qty > min_qty:
                recycle_details[item_code] = item_qty - min_qty
        return recycle_details

    async def is_worth_selling(self, _item: Item) -> bool:
        item_gold_value = _item.get_sell_price()   # FIXME this could be moving // need to be up to date
        materials = [self.environment.items[material] for material in get_craft_recipee(_item).keys()]
        if any([material.is_protected() for material in materials]):
            return False
        gold_value_sorted_materials = sorted(materials, key=lambda x: x["ge"]["buy_price"], reverse=True)
        return item_gold_value > sum(gold_value_sorted_materials[:2])

    async def get_game_task(self) -> Task:
        infos = await self.get_infos()
        task = infos.get("task", "")
        task_type = TaskType(infos.get("task_type", "idle"))
        task_total = infos.get("task_total", 0) - infos.get("task_progress", 0)
        if task_type == TaskType.MONSTERS:
            task_details = self.environment.monsters[task]
        elif task_type == TaskType.ITEMS:
            task_details = self.environment.items.get(task, {})
        else:
            # FIXME define a DefaultTask?
            return Task(
                code="iron",
                type=TaskType.ITEMS,
                total=44,
                details=self.environment.items["iron"]
            )
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
        return self.environment.items["iron"]

    async def set_task(self, item: Item, task_type: TaskType, quantity: int):
        total_nb_materials = sum([qty for _, qty in get_craft_recipee(item).items()])
        self.task = Task(
            code=item.code,
            type=task_type,
            total=min(await self.get_inventory_max_size(), quantity) // total_nb_materials,
            details=item
        )

    async def get_task_type(self, item: Item) -> TaskType:
        if item.code in [i.code for i in self.gatherable_resources]:
            return TaskType.RESOURCES
        return TaskType.ITEMS

    async def get_task(self) -> Task:

        objective = await self.get_best_objective()     # FIXME get it depending on potential XP gain

        # FIXME could be checked here amongst craftable items first, then gatherable ones

        # objective is an item > task will be crafting or gathering
        total_nb_materials = sum([qty for _, qty in get_craft_recipee(objective).items()])

        return Task(
            code=objective.code,
            type=await self.get_task_type(objective),
            total=(await self.get_inventory_max_size()) // total_nb_materials,
            details=objective
        )

    async def prepare_for_task(self) -> dict[str, dict[str, int]]:
        gather_details, collect_details, fight_details = {}, {}, {}

        craft_recipee = get_craft_recipee(self.task.details)

        target_details = {
            k: v*self.task.total
            for k, v in craft_recipee.items()
        }

        for material_code, qty in target_details.items():

            # TODO qualify item: craftable? gatherable? fightable?
            self._logger.debug(f' Check material {material_code}')

            if await self.is_collectable_at_bank(material_code, qty):
                # FIXME Réserver le montant pour qu'un autre personnage ne compte pas dessus
                collect_details[material_code] = qty
                continue
            if await self.is_craftable(material_code):
                # Set material as craft target
                self._logger.info(f' Resetting task to {material_code}')
                await self.set_task(self.environment.items[material_code], TaskType.ITEMS, qty)
                # "Dé-réserver" les articles de banque
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
        return {
            'gather': gather_details,
            'collect': collect_details,
            'fight': fight_details
        }

    async def is_event_still_on(self):
        all_events = await get_all_events(self.session)
        all_events_codes = [event['map']['content']['code'] for event in all_events]
        if self.task.code in all_events_codes:
            return True
        return False

    async def execute_task(self):

        self._logger.info(f" Here is the task to be executed: {self.task.code} ({self.task.type.value})")

        # TODO if inventory filled up, deposit?
        self._logger.debug(f' Current inventory occupied slots: {await self.get_inventory_occupied_slots_nb()}')

        await self.equip_for_task()

        if self.task.type == TaskType.MONSTERS:
            await self.move_to_monster(self.task.code)
            self._logger.info(f'fighting {self.task.code} ...')
            while await self.is_up_to_fight():  # Includes "task completed" check > TODO add dropped material count
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
            if await self.get_inventory_quantity(self.task.code) >= self.task.total:
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
                for material_code, material_unit_qty in get_craft_recipee(self.task.details).items()
            }

            # Checking if all is available in inventory
            if all([(await self.get_inventory_qty()).get(material_code, 0) >= material_qty for material_code, material_qty in craft_details.items()]):
                pass
            # Checking if all is available in bank
            elif all([await get_bank_item_qty(self.session, material_code) >= material_qty for material_code, material_qty in craft_details.items()]):
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
            await self.equip_for_gathering(self.task.details["subtype"])

    async def get_recycling_task(self) -> Task:

        # TODO only one loop on bank equipment

        for item_code in self.obsolete_equipments:
            qty_at_bank = await get_bank_item_qty(self.session, item_code)
            if qty_at_bank > 0:
                # If yes, withdraw them and get to workshop to recycle them, before getting back to bank to deposit all
                nb_free_inventory_slots = await self.get_inventory_free_slots_nb()
                recycling_qty = min(qty_at_bank, nb_free_inventory_slots // 2)  # Need room in inventory when recycling

                # set recycling task
                return Task(
                    code=item_code,
                    type=TaskType.RECYCLE,
                    total=recycling_qty,
                    details=self.environment.items[item_code]
                )

        recycle_details = await self.get_unnecessary_equipments()
        for item_code, recycling_qty in recycle_details.items():
            nb_free_inventory_slots = await self.get_inventory_free_slots_nb()
            recycling_qty = min(recycling_qty, nb_free_inventory_slots // 2)  # Need room in inventory when recycling

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

    async def get_craft_for_equiping_task(self) -> Task:
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
            equipment_min_stock = get_min_stock_qty(equipment)
            self._logger.warning(f' Got {equipment_qty} {equipment.code} on map, need at least {equipment_min_stock}')
            return Task(
                code=equipment.code,
                type=TaskType.ITEMS,
                total=equipment_min_stock - equipment_qty,
                details=equipment
            )
        return Task()

    async def get_event_task(self) -> Task:
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
                if await self.get_skill_level(resource.skill) >= resource.level:
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


async def run_bot(character_object: Character):
    while True:

        await character_object.deposit_items_at_bank()

        # FIXME Define a stock Task, where the gathering / crafting is made until bank stock is reached

        await character_object.manage_task()

        # Check if game task is feasible, assign if it is / necessarily existing
        event_task = await character_object.get_event_task()
        # event_task = None
        game_task = await character_object.get_game_task()
        recycling_task = await character_object.get_recycling_task()
        craft_for_equiping_task = await character_object.get_craft_for_equiping_task()
        fight_for_leveling_up_task = await character_object.get_fight_for_leveling_up_task()
        if event_task.type != TaskType.IDLE:
            character_object.task = event_task
        # No need to do game tasks if already a lot of task coins
        elif (game_task.is_feasible(await character_object.get_infos(), character_object.max_fight_level) and (await get_bank_item_qty(character_object.session, "tasks_coin") < 200)) or len(character_object.objectives) == 0:
            character_object.task = game_task
        elif recycling_task.type != TaskType.IDLE:
            character_object.task = recycling_task
        # TODO get a task of leveling up on gathering if craftable items without autonomy
        elif craft_for_equiping_task.type != TaskType.IDLE:
            character_object.task = craft_for_equiping_task
        elif fight_for_leveling_up_task.type != TaskType.IDLE and await character_object.got_enough_consumables(1):
            character_object.task = fight_for_leveling_up_task
        elif character_object.task.type == TaskType.IDLE:
            # find and assign a valid task
            character_object.task = await character_object.get_task()     # From a list?
        ### SET TASK END ###

        await character_object.execute_task()

        # Reinitialize task
        character_object.task = Task()


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
    async with ClientSession() as session:
        # Parallelization of initial API calls
        # tasks = [
        #     asyncio.create_task(get_all_items(session)),
        #     asyncio.create_task(get_all_monsters(session)),
        #     asyncio.create_task(get_all_resources(session)),
        #     asyncio.create_task(get_all_maps(session)),
        #     asyncio.create_task(get_all_status(session))
        # ]

        items, monsters, resources_data, maps_data, status = await asyncio.gather(*[
            asyncio.create_task(get_all_items(session)),
            asyncio.create_task(get_all_monsters(session)),
            asyncio.create_task(get_all_resources(session)),
            asyncio.create_task(get_all_maps(session)),
            asyncio.create_task(get_all_status(session))
        ])

        environment = Environment(
            items=items,
            monsters=monsters,
            resource_locations=resources_data,
            maps=maps_data,
            status=status
        )

        obsolete_equipments = await get_obsolete_equipments(session, environment)

        # LOCAL_BANK = await get_bank_items(session)

        # Lich 96% > cursed_specter, gold_shield, cursed_hat, malefic_armor, piggy_pants, gold_boots, ruby_ring, ruby_ring, ruby_amulet
        # Lich 100% > cursed_specter, gold_shield, cursed_hat, malefic_armor, piggy_pants, gold_boots, ruby_ring, ruby_ring, magic_stone_amulet
        # Lich > gold_sword, gold_shield, lich_crown, obsidian_armor, gold_platelegs, lizard_boots, dreadful_ring, topaz_ring, topaz_amulet

        # Test filter
        given_items = [
            item
            for item in items.values()
            if item.is_given()
        ]
        logging.warning(f"Equipments that can only be given or dropped: {list(map(lambda x: x.code, given_items))}")

        characters_ = [
            Character(session=session, environment=environment, obsolete_equipments=obsolete_equipments, name='Kersh', max_fight_level=30, skills=['weaponcrafting', 'cooking', 'mining', 'woodcutting']),  # 'weaponcrafting', 'mining', 'woodcutting'
            Character(session=session, environment=environment, obsolete_equipments=obsolete_equipments, name='Capu', max_fight_level=30, skills=['gearcrafting', 'woodcutting', 'mining']),  # 'gearcrafting',
            Character(session=session, environment=environment, obsolete_equipments=obsolete_equipments, name='Brubu', max_fight_level=30, skills=['cooking', 'woodcutting', 'mining']),  # , 'fishing', 'mining', 'woodcutting'
            Character(session=session, environment=environment, obsolete_equipments=obsolete_equipments, name='Crabex', max_fight_level=30, skills=['jewelrycrafting', 'mining', 'woodcutting']),  # 'jewelrycrafting', 'woodcutting', 'mining'
            Character(session=session, environment=environment, obsolete_equipments=obsolete_equipments, name='JeaGa', max_fight_level=30, skills=['fishing', 'jewelrycrafting', 'mining', 'woodcutting']),  # 'cooking', 'fishing'
        ]

        # Initialize all characters asynchronously
        await asyncio.gather(*[character.initialize() for character in characters_])

        # Start the bot for all characters
        await asyncio.gather(*[run_bot(character) for character in characters_])

if __name__ == '__main__':

    logging.basicConfig(
        force=True,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/output.log"),
            logging.StreamHandler()
        ]
    )

    logging.info("Bot started.")
    asyncio.run(main())
    logging.info("Bot finished.")
