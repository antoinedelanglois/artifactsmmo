import aiohttp
import asyncio
import logging
from aiohttp.client_exceptions import ClientConnectorError
from dataclasses import dataclass, field
import re
import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()
TOKEN = os.getenv('ARTIFACTSMMO_TOKEN')
if not TOKEN:
    raise ValueError("Le TOKEN n'est pas défini. Veuillez le définir dans les variables d'environnement.")

logging.basicConfig(
    force=True,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/output.log"),
        logging.StreamHandler()
    ]
)

# Server url
SERVER = "https://api.artifactsmmo.com"
# Your account token (https://artifactsmmo.com/account)
EQUIPMENTS_SLOTS = ['weapon', 'shield', 'helmet', 'body_armor', 'leg_armor', 'boots', 'ring1', 'ring2',
                    'amulet', 'artifact1', 'artifact2', 'artifact3']
EQUIPMENTS_TYPES = ['weapon', 'shield', 'helmet', 'body_armor', 'leg_armor', 'boots', 'ring',
                    'amulet', 'artifact']


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


async def get_crafted_items(session: aiohttp.ClientSession) -> dict[str, dict]:
    equipments = {}
    skills_list = ['weaponcrafting', 'gearcrafting', 'jewelrycrafting', 'cooking']
    for skill_name in skills_list:
        equipments[skill_name] = {
            i["code"]: i
            for i in await get_all_items(session, params={'craft_skill': skill_name})
        }
    return equipments


async def get_equipments_by_type(session: aiohttp.ClientSession) -> dict[str, list[dict]]:
    equipments = {}
    for equipment_type in EQUIPMENTS_TYPES:
        equipments[equipment_type] = await get_all_items(session, params={'type': equipment_type})
    return equipments


def is_equipment_better(equipment_a: dict, equipment_b: dict) -> bool:
    """
    Returns True if equipment_a is strictly better than equipment_b.
    """
    # Parse effects into dictionaries
    effects_a = {effect['name']: effect['value'] for effect in equipment_a.get('effects', [])}
    effects_b = {effect['name']: effect['value'] for effect in equipment_b.get('effects', [])}

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


def identify_obsolete_equipments(equipment_groups: dict[str, list[dict]]) -> list[dict]:
    obsolete_equipments = []
    for equipment_type, equipments in equipment_groups.items():
        for equipment in equipments:
            is_obsolete = False
            for other_equipment in equipments:
                if other_equipment['code'] == equipment['code']:
                    continue  # Skip comparison with itself
                if is_equipment_better(other_equipment, equipment):
                    is_obsolete = True
                    break
            if is_obsolete:
                obsolete_equipments.append(equipment)
    return obsolete_equipments


def identify_obsolete_equipments_optimized(equipment_groups: dict[str, list[dict]]) -> list[dict]:
    obsolete_equipments = []
    for equipment_type, equipments in equipment_groups.items():
        # Sort equipments by level and effects
        equipments_sorted = sorted(equipments, key=lambda e: (
            -sum(effect['value'] for effect in e.get('effects', []))  # Higher total effects first
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


async def get_obsolete_equipments(session: aiohttp.ClientSession) -> list[dict]:
    equipment_groups = await get_equipments_by_type(session)

    # Filter the existing equipments on map
    map_equipments = {}
    for equipment_type, equipments in equipment_groups.items():
        filtered_equipments = []
        for equipment in equipments:
            # Only if minimum quantity available
            if await get_all_map_item_qty(session, equipment["code"]) >= await get_min_stock_qty(session, equipment["code"]):
                filtered_equipments.append(equipment)
        map_equipments[equipment_type] = filtered_equipments

    obsolete_equipments = identify_obsolete_equipments_optimized(map_equipments)
    return obsolete_equipments


async def get_all_items_quantities(session: aiohttp.ClientSession) -> dict[str, int]:
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


async def needs_stock(session: aiohttp.ClientSession, _item_code: str, total_quantities: dict[str, int] = None) -> bool:
    if total_quantities is None:
        total_quantities = await get_all_items_quantities(session)
    return await get_all_map_item_qty(session, _item_code, total_quantities) < await get_min_stock_qty(session, _item_code)


async def get_bank_items(session: aiohttp.ClientSession, params: dict = None) -> dict:
    if params is None:
        params = {"size": 100}  # FIXME when bank inventory will be more than 100 // need to get pages
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


async def get_all_events(session: aiohttp.ClientSession, params: dict = None) -> list:
    """
    Retrieves all maps from the API.
    Returns a list of maps with their details.
    """
    if params is None:
        params = {}
    url = f"{SERVER}/events"
    data = await make_request(session=session, method='GET', url=url, params=params)
    return data["data"] if data else []


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


async def get_my_characters(session: aiohttp.ClientSession) -> list:
    url = f"{SERVER}/my/characters"
    data = await make_request(session=session, method='GET', url=url)
    return data["data"] if data else []


async def get_inventories_qty(session: aiohttp.ClientSession) -> dict[str, int]:
    characters_infos = await get_my_characters(session)
    slot_inventories = [slot for c in characters_infos for slot in c['inventory']]
    inventories_dict = {}
    for slot in slot_inventories:
        if slot["code"] == "":
            continue
        inventories_dict[slot["code"]] = inventories_dict.get(slot["code"], 0) + slot["quantity"]
    return inventories_dict


async def get_equipments_qty(session: aiohttp.ClientSession) -> dict[str, int]:
    characters_infos = await get_my_characters(session)
    equipments = [c[f'{equipment_slot}_slot'] for c in characters_infos for equipment_slot in EQUIPMENTS_SLOTS]
    equipments_dict = {}
    for equipment in equipments:
        if equipment == "":
            continue
        equipments_dict[equipment] = equipments_dict.get(equipment, 0) + 1
    return equipments_dict


async def get_all_map_item_qty(session: aiohttp.ClientSession, _item_code: str, total_quantities: dict[str, int] = None) -> int:
    if total_quantities is None:
        total_quantities = await get_all_items_quantities(session)
    return total_quantities.get(_item_code, 0)


async def get_min_stock_qty(session: aiohttp.ClientSession, item_code: str) -> int:
    item_details = await get_item_infos(session, item_code)
    if item_details["type"] == "ring":
        return 10
    return 5


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
    data = await make_request(
        session=session,
        method='GET',
        url=f"{SERVER}/monsters/{_monster_code}"
    )
    return data["data"] if data else {}


async def get_all_items(session: aiohttp.ClientSession, params: dict) -> list[dict]:
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
            items.extend(data["data"])
            page += 1
        else:
            break
    return items


async def is_protected_material(session: aiohttp.ClientSession, _material_code: str) -> bool:
    material_infos = await get_item_infos(session, _material_code)
    is_task_material = material_infos["type"] == "resource" and material_infos["subtype"] == "task"
    is_rare_material = _material_code in ['topaz', 'emerald', 'ruby', 'sapphire', 'sap']
    return is_task_material or is_rare_material


async def is_valid_equipment(_equipment: dict) -> bool:
    return _equipment['type'] in EQUIPMENTS_SLOTS + ['ring', 'artifact']


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


async def get_monster_vulnerabilities(monster_infos: dict) -> dict[str, int]:

    resistances = sorted([
        res
        for res in ["res_fire", "res_earth", "res_water", "res_air"]
    ], key=lambda x: monster_infos[x], reverse=False
    )

    if any([monster_infos.get(res, 0) < 0 for res in resistances]):
        resistances = [res for res in resistances if monster_infos.get(res, 0) < 0]
    else:
        resistances = [resistances[0]]      # FIXME get the lesser values

    return {resistance.replace("res_", ""): -1 * monster_infos[resistance] for resistance in resistances}


async def select_best_equipment_set(current_equipment: dict, sorted_valid_equipments: dict[str, list[dict]], vulnerabilities: dict) -> dict:
    """
    Selects the best equipment set based on monster vulnerabilities and equipment effects.

    :param current_equipment: The currently equipped items per slot (or empty dict if none equipped).
    :param sorted_valid_equipments: A dictionary of lists of valid equipment items per slot, sorted by level.
    :param vulnerabilities: A dictionary of the monster's elemental vulnerabilities with their percentages.
    :return: The selected best equipment set.
    """
    selected_equipment = {}

    # First, select the best weapon based on vulnerabilities
    weapon_current = current_equipment.get('weapon', {})
    weapon_list = sorted_valid_equipments.get('weapon', [])
    best_weapon = select_best_weapon(weapon_current, weapon_list, vulnerabilities)
    selected_equipment['weapon'] = best_weapon

    # Determine the primary attack elements of the selected weapon
    weapon_effects = {effect['name']: effect['value'] for effect in best_weapon.get('effects', [])}
    primary_attack_elements = [effect_name.replace('attack_', '') for effect_name in weapon_effects.keys() if effect_name.startswith('attack_')]

    # Now, select the best equipment for other slots that enhance the selected weapon
    for slot in ['ring', 'amulet', 'armor']:
        current_item = current_equipment.get(slot, {})
        equipment_list = sorted_valid_equipments.get(slot, [])
        best_item = select_best_support_equipment(current_item, equipment_list, primary_attack_elements)
        selected_equipment[slot] = best_item

    return selected_equipment


def select_best_weapon(current_weapon: dict, weapon_list: list[dict], vulnerabilities: dict) -> dict:
    """
    Selects the best weapon based on vulnerabilities.
    """
    if not weapon_list:
        return current_weapon
    if not current_weapon:
        current_weapon = weapon_list[0]

    # Determine if all vulnerabilities are equal
    vulnerability_percentages = list(vulnerabilities.values())
    vulnerabilities_equal = all(pct == vulnerability_percentages[0] for pct in vulnerability_percentages)

    def calculate_weapon_score(_effects: dict, _vulnerabilities: dict, vulnerabilities_equal: bool) -> float:
        """
        Calculate the weapon score based on the effects and vulnerabilities.
        """
        score = 0.0
        total_attack = 0.0  # Used when vulnerabilities are equal to prioritize damage
        for effect_name, effect_value in _effects.items():
            if effect_name.startswith("attack_"):
                element = effect_name.replace("attack_", "")
                total_attack += effect_value
                if vulnerabilities_equal:
                    score += effect_value * 4  # High weight for attack effects
                else:
                    if element in _vulnerabilities:
                        percentage = _vulnerabilities[element]
                        score += effect_value * 4 * (percentage / 100)  # Weight by vulnerability percentage
                    else:
                        score += effect_value  # Lesser weight for non-vulnerable elements
            elif effect_name.startswith("dmg_"):
                element = effect_name.replace("dmg_", "")
                if vulnerabilities_equal:
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
        if vulnerabilities_equal:
            score += total_attack * 2  # Additional weight for total attack
        return score

    best_weapon = current_weapon
    best_score = calculate_weapon_score(
        {effect['name']: effect['value'] for effect in current_weapon.get('effects', [])},
        vulnerabilities,
        vulnerabilities_equal
    )

    for weapon in weapon_list:
        weapon_effects = {effect['name']: effect['value'] for effect in weapon.get('effects', [])}
        weapon_score = calculate_weapon_score(weapon_effects, vulnerabilities, vulnerabilities_equal)
        if weapon_score > best_score:
            best_weapon = weapon
            best_score = weapon_score
            logging.debug(f"Best weapon updated to {best_weapon['code']} with score {best_score}'")

    return best_weapon


def select_best_support_equipment(current_item: dict, equipment_list: list[dict], primary_attack_elements: list[str]) -> dict:
    """
    Selects the best support equipment that enhances the selected weapon.

    :param current_item: The currently equipped item (or empty dict if none equipped).
    :param equipment_list: A list of valid equipment items for the slot.
    :param primary_attack_elements: List of primary attack elements from the selected weapon.
    :return: The selected best support equipment.
    """
    if not equipment_list:
        return current_item
    if not current_item:
        current_item = equipment_list[0]

    def calculate_support_score(_effects: dict, primary_elements: list[str]) -> float:
        """
        Calculate the support equipment score based on how well it enhances the weapon's primary elements.
        """
        score = 0.0
        for effect_name, effect_value in _effects.items():
            if effect_name.startswith("attack_"):
                element = effect_name.replace("attack_", "")
                if element in primary_elements:
                    score += effect_value * 4  # High weight for matching attack elements
                else:
                    score += effect_value  # Lesser weight for other elements
            elif effect_name.startswith("dmg_"):
                element = effect_name.replace("dmg_", "")
                if element in primary_elements:
                    score += effect_value * 3  # Weight for matching damage effects
                else:
                    score += effect_value
            elif effect_name == "hp":
                score += effect_value * 0.25
            elif effect_name == "defense":
                score += effect_value * 0.5
            else:
                score += effect_value  # Default score for other effects
        return score

    best_item = current_item
    best_score = calculate_support_score(
        {effect['name']: effect['value'] for effect in current_item.get('effects', [])},
        primary_attack_elements
    )

    for item in equipment_list:
        item_effects = {effect['name']: effect['value'] for effect in item.get('effects', [])}
        item_score = calculate_support_score(item_effects, primary_attack_elements)
        if item_score > best_score:
            best_item = item
            best_score = item_score
            logging.debug(f"Best {item.get('slot', 'equipment')} updated to {best_item['code']} with score {best_score}'")

    return best_item


async def select_best_equipment(equipment1_infos: dict, sorted_valid_equipments: list[dict], vulnerabilities: dict[str, int]) -> dict:
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

    best_equipment = equipment1_infos
    best_score = calculate_effect_score(
        {effect['name']: effect['value'] for effect in equipment1_infos.get('effects', [])},
        vulnerabilities
    )

    for equipment2_infos in sorted_valid_equipments:
        equipment2_effects = {effect['name']: effect['value'] for effect in equipment2_infos.get('effects', [])}
        equipment2_score = calculate_effect_score(equipment2_effects, vulnerabilities)

        # Compare scores and select the best
        if equipment2_score > best_score:
            best_equipment = equipment2_infos
            best_score = equipment2_score
            logging.debug(f"Best equipment updated to {best_equipment['code']} with score {best_score}'")

    return best_equipment


@dataclass
class Task:
    code: str = ""
    type: str = ""  # resources / monsters / items
    total: int = 0
    details: dict = None

    async def is_feasible(self, character_max_level: int) -> bool:
        if self.type == "monsters" and self.details['level'] <= character_max_level:
            if self.code == "imp":
                return False
            return True
        return False


@dataclass
class Character:
    session: aiohttp.ClientSession
    all_equipments: dict[str, dict]     # TODO get it to dataclass
    all_items: dict[str, list[dict]]
    excluded_items: dict[str, list[dict]]
    name: str
    skills: list[str]
    max_fight_level: int = 0
    stock_qty_objective: int = 500
    task: Task = field(default_factory=Task)
    gatherable_resources: list[dict] = field(default_factory=list)
    craftable_items: list[dict] = field(default_factory=list)
    fightable_monsters: list[dict] = field(default_factory=list)
    fightable_materials: list[dict] = field(default_factory=list)
    objectives: list[dict] = field(default_factory=list)
    fight_objectives: list[dict] = field(default_factory=list)

    def __post_init__(self):
        # Custom logger setup
        self.logger = logging.getLogger(self.name)
        # handler = logging.StreamHandler()
        # formatter = logging.Formatter(f'%(asctime)s - {self.name} - [%(levelname)s] %(message)s')
        # handler.setFormatter(formatter)
        # self.logger.addHandler(handler)
        # self.logger.propagate = False

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
            # await get_item_infos(self.session, dropped_material['code'])
            {item["code"]: item for item in self.all_items['resources']}[dropped_material['code']]
            for gatherable_resources_spot in gatherable_resources_spots
            for dropped_material in gatherable_resources_spot['drops']
            if (not await is_protected_material(self.session, dropped_material['code'])) and dropped_material['rate'] <= 100   # Exclude protected material
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
            if crafting_skill == 'fishing':
                crafting_skill = 'cooking'
            skill_level = infos[f'{crafting_skill}_level']
            skill_craftable_items = [item for item in self.all_items[crafting_skill] if item['level'] <= skill_level]
            self.logger.debug(f' crafting_skill: {crafting_skill} > {[i["code"] for i in skill_craftable_items]}')
            craftable_items.extend(skill_craftable_items[::-1])
        # TODO exclude protected items (such as the one using jasper_crystal)
        filtered_craftable_items = [
            item
            for item in craftable_items
        ]
        self.craftable_items = filtered_craftable_items
        self.logger.warning(f"Craftable items for {self.name}: {[i['code'] for i in self.craftable_items]}")

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
        self.fightable_materials = [
            # await get_item_infos(self.session, drop['code'])
            {item["code"]: item for item in self.all_items['resources']}[drop['code']]
            for monster_details in self.fightable_monsters
            for drop in monster_details['drops']
            if drop['rate'] <= 25
        ]
        self.logger.debug(f"Fightable monsters for {self.name}: {[m['code'] for m in self.fightable_monsters]}")
        self.logger.debug(f"Fightable materials for {self.name}: {[m['code'] for m in self.fightable_materials]}")

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

        need_level_up_craftable_items = [
            item
            for item in self.craftable_items
            if not await self.can_be_home_made(item['code']) and await self.does_item_provide_xp(item)
        ]
        self.logger.info(f' NEED LEVELING UP OR SPECIAL MATERIALS TO CRAFT: {[o["code"] for o in need_level_up_craftable_items]}')

        # Sort items by their rarity in the bank (to prioritize items that are rarer)
        items2bank_qty = {
            craftable_item['code']: await get_bank_item_qty(self.session, craftable_item['code'])
            for craftable_item in objectives
        }

        item_objectives = sorted(objectives, key=lambda x: items2bank_qty.get(x['code'], 0), reverse=False)

        resource_objectives = [
            resource
            for resource in self.gatherable_resources
            if await self.can_be_home_made(resource['code']) and await self.does_item_provide_xp(resource)
        ]

        item_objectives.extend(resource_objectives[::-1])
        # objectives.extend(self.fightable_materials[::-1])

        fight_objectives = [
            monster
            for monster in self.fightable_monsters
            if await self.can_be_vanquished(monster['code']) and await self.does_fight_provide_xp(monster)
        ]

        self.objectives = item_objectives
        self.fight_objectives = fight_objectives[::-1]
        self.logger.info(f' CAN GET XP WITH: {[o["code"] for o in self.objectives]}')

    async def can_be_vanquished(self, monster_code: str) -> bool:
        return monster_code in [m["code"] for m in self.fightable_monsters]

    async def get_infos(self) -> dict:
        url = f"{SERVER}/characters/{self.name}"
        data = await make_request(session=self.session, method='GET', url=url)
        if data:
            self.logger.debug("Fetched character info successfully.")
        else:
            self.logger.error("Failed to fetch character info.")
        return data["data"] if data else {}

    async def get_inventory_qty(self) -> dict[str, int]:
        url = f"{SERVER}/characters/{self.name}"
        data = await make_request(session=self.session, method='GET', url=url)
        if data:
            self.logger.debug("Fetched character inventory successfully.")
        else:
            self.logger.error("Failed to fetch character inventory.")
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
            # FIXME check to be done beforehand?
            nb_free_slots = await self.get_inventory_free_slots_nb()
            cooldown_ = await self.bank_withdraw(item_code, min(item_qty, nb_free_slots))
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

    async def is_up_to_fight(self):
        got_enough_consumables = await self.got_enough_consumables(-1)
        inventory_not_full = await self.is_inventory_not_full()
        is_task_complete = await self.is_task_completed()
        is_not_at_spawn_place = not await self.is_at_spawn_place()
        return got_enough_consumables and inventory_not_full and not is_task_complete and is_not_at_spawn_place

    async def is_at_spawn_place(self) -> bool:
        current_location = await self.get_current_location()
        if current_location == (0, 0):
            self.logger.debug(f'is already at spawn place - likely killed by a monster')
            return True
        return False

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
            self.logger.warning(f'No resource {content_code} on this map')
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
                material_inventory_qty = await self.get_inventory_quantity(material_code)
                nb_craftable_items = min(material_inventory_qty//qty, nb_craftable_items)

        self.logger.debug(f' nb of craftable items {"from inventory" if from_inventory else ""} for {_item_code} is {nb_craftable_items}')

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
            self.logger.debug(f'{quantity} {item_code} deposited. Cooldown: {_cooldown} seconds')
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
            self.logger.debug(f'{quantity} {item_code} withdrawn. Cooldown: {_cooldown} seconds')
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
            return _cooldown
        else:
            self.logger.error(f'failed to perform fighting action.')
            return 0

    async def perform_gathering(self) -> int:
        url = f"{SERVER}/my/{self.name}/action/gathering"
        data = await make_request(session=self.session, method='POST', url=url)
        if data:
            _cooldown = data["data"]["cooldown"]["total_seconds"]
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

        # SECURITY CHECK ON RARE ITEMS
        craft_recipee = await get_craft_recipee(self.session, item_code)
        if 'jasper_crystal' in craft_recipee.keys():
            self.logger.warning(f' Item {item_code} is rare so better not to recycle it.')
            return 0

        data = await make_request(session=self.session, method='POST', url=url, payload=payload)
        if data:
            _cooldown = data["data"]["cooldown"]["total_seconds"]
            self.logger.info(f'{qte} {item_code} recycled. Cooldown: {_cooldown} seconds')
            return _cooldown
        else:
            self.logger.error(f'failed to recycle {qte} {item_code}.')
            return 0

    async def is_gatherable(self, resource_code) -> bool:
        # return resource_code in [item["code"] for item in self.gatherable_resources] and resource_code in [item["code"] for item in self.objectives]
        return resource_code in [item["code"] for item in self.gatherable_resources]

    async def is_fightable(self, material_code) -> bool:
        # return material_code in [item["code"] for item in self.fightable_materials] and material_code in [item["code"] for item in self.objectives]
        return material_code in [item["code"] for item in self.fightable_materials]

    async def is_craftable(self, item_code) -> bool:
        # return item_code in [item["code"] for item in self.craftable_items] and item_code in [item["code"] for item in self.objectives]
        return item_code in [item["code"] for item in self.craftable_items]

    async def is_collectable(self, item_code: str) -> bool:
        return not (await is_protected_material(self.session, item_code))

    async def move_to_workshop(self, item_code: str = None):
        if item_code is None:
            item_code = self.task.code
        # get the skill out of item
        item_infos = await get_item_infos(self.session, item_code)
        skill_name = item_infos['craft']['skill']
        coords = await self.get_nearest_coords(content_type='workshop', content_code=skill_name)
        self.logger.debug(f'{self.name} > moving to workshop at {coords}')
        cooldown_ = await self.move(*coords)
        await asyncio.sleep(cooldown_)

    async def move_to_monster(self, monster_code: str = ""):
        monster_code = self.task.code if monster_code == "" else monster_code
        coords = await self.get_nearest_coords(content_type='monster', content_code=monster_code)
        self.logger.debug(f'{self.name} > moving to monster {monster_code} at {coords}')
        cooldown_ = await self.move(*coords)
        await asyncio.sleep(cooldown_)

    async def move_to_task_master(self):
        coords = await self.get_nearest_coords(content_type='tasks_master', content_code='monsters')
        self.logger.debug(f'{self.name} > moving to tasks master at {coords}')
        cooldown_ = await self.move(*coords)
        await asyncio.sleep(cooldown_)

    async def prepare_fighting_for_item(self, _item_code: str = None):
        if _item_code is not None:
            monster_location = await get_dropping_monster_locations(self.session, _item_code)
            monster_code = monster_location['code']
            self.logger.debug(f' item {_item_code} is dropped by {monster_code}')

    async def get_equipment_code(self, _equipment_slot: str):
        infos = await self.get_infos()
        return infos[f'{_equipment_slot}_slot']

    async def go_and_equip(self, _equipment_slot: str, _equipment_code: str):
        current_equipment_code = await self.get_equipment_code(_equipment_slot)
        if current_equipment_code != _equipment_code:
            self.logger.debug(f' will change equipment for {_equipment_slot} from {current_equipment_code} to {_equipment_code}')
            await self.move_to_bank()
            await self.withdraw_items_from_bank({_equipment_code: 1})
            if current_equipment_code != "":
                await self.unequip(_equipment_slot)
                await self.deposit_items_at_bank({current_equipment_code: 1})
            await self.equip(_equipment_code, _equipment_slot)

    async def equip_for_gathering(self, _item_code: str):
        item_infos = await get_item_infos(self.session, _item_code)
        if item_infos.get("subtype", None) is None:
            return

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
            self.logger.debug(f'going to fight: {fight_details} ...')
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

    async def does_fight_provide_xp(self, monster: dict) -> bool:
        character_level = await self.get_level()
        monster_level = monster['level']
        return monster_level >= (character_level - 10)

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

    async def equip_for_fight(self):

        # Identify vulnerability
        monster_infos = await get_monster_infos(self.session, self.task.code)
        vulnerabilities = await get_monster_vulnerabilities(monster_infos=monster_infos)
        self.logger.info(f' monster {self.task.code} vulnerabilities are {vulnerabilities}')

        # Manage equipment
        for equipment_slot in EQUIPMENTS_SLOTS:
            await self.equip_best_equipment(equipment_slot, vulnerabilities)

        # Manage consumables
        await self.equip_best_consumables()

    async def get_eligible_bank_consumables(self) -> list[dict]:
        return [
            consumable_infos
            for consumable_infos in self.all_items["cooking"]
            if await get_bank_item_qty(self.session, consumable_infos["code"]) > 0 and consumable_infos['level'] <= await self.get_level()
        ]

    async def get_2_best_consumables_including_equipped(self) -> list[dict]:
        """
        Fetches the two best consumables, including currently equipped ones, and ranks them.
        """
        # Fetch all consumables from the bank (TODO and inventory)
        valid_consumables = await self.get_eligible_bank_consumables()

        # Add the currently equipped consumables to the list of valid ones (if they are equipped)
        valid_consumables_codes = [c["code"] for c in valid_consumables]
        ordered_current_consumables = await self.get_ordered_current_consumables()
        for current_consumable in ordered_current_consumables:
            if current_consumable and current_consumable["code"] not in valid_consumables_codes:
                valid_consumables.append(current_consumable)

        valid_consumables = [
            consumable for consumable in valid_consumables
            if not await self.is_protected_consumable(consumable['code'])
        ]

        self.logger.debug(f' eligible consumables are {valid_consumables}')
        self.logger.debug(f' ordered current consumables are {ordered_current_consumables}')

        # Sort all consumables by level (higher is better)
        sorted_two_best_consumables = sorted(valid_consumables, key=lambda x: x['level'], reverse=True)[:2]
        self.logger.debug(f' two best consumables are {sorted_two_best_consumables}')

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
        return consumable_code in [i["code"] for i in self.excluded_items['consumables']]

    # async def is_valid_consumable(self, consumable_level: int) -> bool:
    #     return consumable_level <= await self.get_level()

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
                self.logger.debug(f"No valid consumable available for {slot}. Skipping.")
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
                    self.logger.debug(f"{slot} already equipped with {new_code} and fully stocked.")
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

    async def equip_best_equipment(self, _equipment_slot: str, vulnerabilities: dict[str, int]):
        available_equipments = await self.get_bank_equipments_for_slot(_equipment_slot)
        self.logger.debug(f'available equipment at bank {[e["code"] for e in available_equipments]}')
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
        new_equipment_details = await select_best_equipment(current_equipment_infos, sorted_valid_equipments, vulnerabilities)
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
            self.logger.debug(f'{qte} {slot_code} unequipped. Cooldown: {_cooldown} seconds')
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
            self.logger.debug(f'{qte} {item_code} equipped at slot {slot_code}. Cooldown: {_cooldown} seconds')
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
        bank_items = await get_bank_items(self.session)
        return [
            self.equipments[item_code]
            for item_code in bank_items.keys()
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

    async def get_unnecessary_equipments(self) -> dict[str, int]:
        recycle_details = {}
        # Apply only on those that can be crafted again
        for item_code, item_qty in (await get_bank_items(self.session)).items():
            # No recycling for planks and ores and cooking
            if item_code in [i['code'] for i in self.all_items['woodcutting'] + self.all_items['mining'] + self.all_items['cooking']]:
                continue
            min_qty = await get_min_stock_qty(self.session, item_code)
            if item_code in [i['code'] for i in self.craftable_items] and item_qty > min_qty:
                recycle_details[item_code] = item_qty - min_qty
        return recycle_details

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
        task = infos["task"]
        task_type = infos["task_type"]
        task_total = infos["task_total"]-infos["task_progress"]
        if task_type == 'monsters':
            task_details = await get_monster_infos(self.session, task)
        elif task_type == 'items':
            task_details = await get_item_infos(self.session, task)
        else:
            raise NotImplementedError()
        return Task(
            code=task,
            type=task_type,
            total=task_total,
            details=task_details
        )

    async def get_task_type(self, item_details: dict) -> str:
        # if item_details.get('craft') is not None:
        if item_details['code'] in [item['code'] for item in self.craftable_items]:
            return 'items'
        # if item_details['subtype'] in ['mining', 'woodcutting', 'fishing']:
        if item_details['code'] in [item['code'] for item in self.gatherable_resources]:
            return 'resources'
        if item_details['code'] in [item['code'] for item in self.fightable_materials]:
            return 'monsters'
        self.logger.error(f'Unknown item {item_details}')
        return 'unknown'

    async def get_task(self, item_code: str = None) -> Task:

        if item_code is None:
            objective = self.objectives[0]
        else:
            objective = await get_item_infos(self.session, item_code)
        task_type = await self.get_task_type(objective)       # resources / monsters / items
        if task_type == 'monsters':
            target_code = await get_dropping_monster_locations(self.session, objective['code'])
        else:
            target_code = objective['code']
        max_nb = await self.get_inventory_max_size()
        if task_type == 'items':
            craft_recipee = await get_craft_recipee(self.session, target_code)
            total_nb_materials = sum([qty for _, qty in craft_recipee.items()])
            nb = max_nb // total_nb_materials
        else:
            nb = max_nb
        return Task(
            code=target_code,
            type=task_type,
            total=nb,
            details=objective
        )

    async def prepare_for_task(self) -> dict[str, dict[str, int]]:
        gather_details, collect_details, fight_details = {}, {}, {}

        if self.task.type == 'items':
            # craft_recipee = await get_craft_recipee(self.session, self.task.code)
            craft_recipee = {m['code']: m['quantity'] for m in self.task.details['craft']['items']}
        else:
            craft_recipee = {self.task.code: 1}

        target_details = {
            k: v*self.task.total
            for k, v in craft_recipee.items()
        }

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
                self.logger.info(f' Resetting task to {material}')
                resetted_task = await self.get_task(material)
                self.task = resetted_task
                # "Dé-réserver" les articles de banque
                return await self.prepare_for_task()
            if await self.is_gatherable(material):
                gather_details[material] = qty
                self.logger.debug(f' Gatherable qty {qty}')
                continue
            if await self.is_fightable(material):
                fight_details[material] = qty
                self.logger.debug(f' Fightable for qty {qty}')
                continue
            self.logger.warning(f" Material {material} won't provide XP...")
        return {
            'gather': gather_details,
            'collect': collect_details,
            'fight': fight_details
        }

    async def execute_task(self):

        # TODO if inventory filled up, deposit?
        self.logger.debug(f' Current inventory occupied slots: {await self.get_inventory_occupied_slots_nb()}')

        if self.task.type == 'monsters':
            await self.equip_for_task()
            await self.move_to_monster(self.task.code)
            self.logger.info(f'fighting {self.task.code} ...')
            while await self.is_up_to_fight():  # Includes "task completed" check > TODO add dropped material count
                # TODO decrement task total on each won combat
                cooldown_ = await self.perform_fighting()
                await asyncio.sleep(cooldown_)

        elif self.task.type == 'event':
            await self.equip_for_task()
            await self.move(*self.task.details['location'])
            self.logger.info(f'fighting {self.task.code} ...')
            while await self.is_up_to_fight():  # Includes "task completed" check > TODO add dropped material count
                # TODO decrement task total on each won combat
                cooldown_ = await self.perform_fighting()
                await asyncio.sleep(cooldown_)

        elif self.task.type == 'resources':
            await self.equip_for_gathering(self.task.code)
            # TODO decrement task total on each target resource gathered (in inventory)
            await self.gather_material(self.task.code, self.task.total)

        elif self.task.type == 'recycle':
            await self.move_to_bank()
            await self.withdraw_items_from_bank({self.task.code: self.task.total})
            await self.move_to_workshop()
            cooldown = await self.perform_recycling(self.task.code, self.task.total)
            await asyncio.sleep(cooldown)

        elif self.task.type == 'items':
            # if all available at bank -> pick it and go craft
            craft_recipee = await get_craft_recipee(self.session, self.task.code)
            # nb_items_to_craft = await self.get_nb_craftable_items(self.task.code, from_inventory=True)
            nb_items_to_craft = await self.get_nb_craftable_items(self.task.code)
            craft_details = {
                material_code: material_unit_qty * min(self.task.total, nb_items_to_craft)
                # material_code: material_unit_qty * self.task.total
                for material_code, material_unit_qty in craft_recipee.items()
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
            nb_items_to_craft = await self.get_nb_craftable_items(self.task.code, from_inventory=True)

            if nb_items_to_craft > 0:
                await self.move_to_workshop(self.task.code)
                cooldown_ = await self.perform_crafting(self.task.code, nb_items_to_craft)
                await asyncio.sleep(cooldown_)

    async def equip_for_task(self):

        if self.task.type in ['monsters', 'event']:
            # Identify vulnerability
            monster_infos = await get_monster_infos(self.session, self.task.code)
            vulnerabilities = await get_monster_vulnerabilities(monster_infos=monster_infos)
            self.logger.info(f' monster {self.task.code} vulnerabilities are {vulnerabilities}')



            # Manage equipment
            for equipment_slot in ['weapon', 'shield', 'helmet', 'body_armor', 'leg_armor', 'boots', 'ring1', 'ring2',
                                   'amulet', 'artifact1', 'artifact2', 'artifact3']:
                await self.equip_best_equipment(equipment_slot, vulnerabilities)

            # Manage consumables
            await self.equip_best_consumables()

    async def get_recycling_task(self) -> Task:

        # TODO only one loop on bank equipment

        for item_code in self.excluded_equipments:
            qty_at_bank = await get_bank_item_qty(self.session, item_code)
            if qty_at_bank > 0:
                # If yes, withdraw them and get to workshop to recycle them, before getting back to bank to deposit all
                nb_free_inventory_slots = await self.get_inventory_free_slots_nb()
                recycling_qty = min(qty_at_bank, nb_free_inventory_slots // 2)  # Need room in inventory when recycling

                # set recycling task
                return Task(
                    code=item_code,
                    type="recycle",
                    total=recycling_qty,
                    details=await get_item_infos(self.session, item_code)
                )

        recycle_details = await self.get_unnecessary_equipments()
        for item_code, recycling_qty in recycle_details.items():
            nb_free_inventory_slots = await self.get_inventory_free_slots_nb()
            recycling_qty = min(recycling_qty, nb_free_inventory_slots // 2)  # Need room in inventory when recycling

            # set recycling task
            return Task(
                code=item_code,
                type="recycle",
                total=recycling_qty,
                details=await get_item_infos(self.session, item_code)
            )
        return None

    async def get_fight_for_leveling_up_task(self) -> Task:
        # If XP can be gained by fighting, go
        # highest_fightable_monster = sorted(self.fightable_monsters, key=lambda x: x['level'], reverse=True)[0]
        if len(self.fight_objectives) > 0:
            highest_fightable_monster = self.fight_objectives[0]
            return Task(
                code=highest_fightable_monster['code'],
                type='monsters',
                total=99,   # FIXME when does it stop?
                details=highest_fightable_monster
            )
        return None

    async def get_craft_for_equiping_task(self) -> Task:
        total_quantities = await get_all_items_quantities(self.session)
        craftable_new_equipments = [
            i
            for i in self.craftable_items      # FIXME how is cooking handled?
            if i['code'] in [
                o['code']
                for o in self.objectives
            ] and await needs_stock(self.session, i['code'], total_quantities) and i['type'] in EQUIPMENTS_SLOTS + ['ring', 'artifact']
        ]

        if len(craftable_new_equipments) > 0:
            self.logger.warning(f' New equipments to craft: {craftable_new_equipments}')
            equipment_code = craftable_new_equipments[0]['code']
            equipment_qty = await get_all_map_item_qty(self.session, equipment_code)
            equipment_min_stock = await get_min_stock_qty(self.session, equipment_code)
            self.logger.warning(f' Got {equipment_qty} {equipment_code} on map, need at least {equipment_min_stock}')
            return Task(
                code=equipment_code,
                type='items',
                total=equipment_min_stock - equipment_qty,
                details=craftable_new_equipments[0]
            )
        return None

    async def get_event_task(self) -> Task:
        all_events = await get_all_events(self.session)
        for event in all_events:
            if event["name"] == "Bandit Camp":
                monster_code = event["map"]["content"]["code"]
                monster_details = await get_monster_infos(self.session, monster_code)
                monster_details['location'] = (event["map"]["x"], event["map"]["y"])
                return Task(
                    code=monster_code,
                    type="event",
                    total=99,   # FIXME when does it stop?
                    details=monster_details
                )
        return None


async def run_bot(character_object: Character):
    while True:

        ### DEPOSIT ALL ###
        await character_object.deposit_items_at_bank()

        # FIXME Define a stock Task, where the gathering / crafting is made until bank stock is reached

        ## MANAGE TASK ###
        await character_object.manage_task()

        ### SET TASK BEGIN ###
        # Check if game task is feasible, assign if it is / necessarily existing
        event_task = await character_object.get_event_task()
        game_task = await character_object.get_game_task()
        recycling_task = await character_object.get_recycling_task()
        craft_for_equiping_task = await character_object.get_craft_for_equiping_task()
        fight_for_leveling_up_task = await character_object.get_fight_for_leveling_up_task()
        if event_task is not None:
            character_object.task = event_task
        elif await game_task.is_feasible(character_object.max_fight_level):
            character_object.task = game_task
        elif recycling_task is not None:
            character_object.task = recycling_task
        # TODO get a task of leveling up on gathering if craftable items without autonomy
        elif craft_for_equiping_task is not None:
            character_object.task = craft_for_equiping_task
        elif fight_for_leveling_up_task is not None and await character_object.got_enough_consumables(-1):
            character_object.task = fight_for_leveling_up_task
        elif character_object.task.type == "":
            # find and assign a valid task
            character_object.task = await character_object.get_task()     # From a list?
        ### SET TASK END ###

        # task_details = await character_object.prepare_for_task()   # Depending on character_object.task.type / including auto_equip

        character_object.logger.warning(f" Here is the task to be executed: {character_object.task}")
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

# personae
# ** WOOD CUTTER / PLANK CRAFTER
# ** FISHER / COOKING MASTER
# ** ORE MINER / BLACKSMITH
# ** WARRIOR
# + WEAPON CRAFTER
# + GEAR CRAFTER
# + JEWEL CRAFTER

# FIXME consumable should be of the correct level to be equipped
# FIXME if there is a task implying fight, make it priority

async def main():
    async with aiohttp.ClientSession() as session:

        EXCLUDED_MATERIALS = ["jasper_crystal"]     # type = "resource" + subtype = "task"

        all_items = {
            crafting_skill: await get_all_items(session, params={"craft_skill": crafting_skill})
            for crafting_skill in ['weaponcrafting', 'gearcrafting', 'jewelrycrafting', 'cooking', 'woodcutting', 'mining']
        }
        all_items["resources"] = await get_all_items(session, params={"type": "resource"})

        excluded_items = {}

        excluded_items['consumables'] = [
            cooked_consumable
            for cooked_consumable in all_items['cooking']
            # if any(["boost" in effect["name"] for effect in cooked_consumable['effects']])
        ]
        logging.info(f'Excluded consumables: {[c["code"] for c in excluded_items["consumables"]]}')

        obsolete_equipments = await get_obsolete_equipments(session)
        all_equipments = await get_crafted_items(session)
        all_equipments['excluded'] = {equipment['code']: 'excluded' for equipment in obsolete_equipments}

        # LOCAL_BANK = await get_bank_items(session)

        characters_ = [
            Character(session=session, excluded_items=excluded_items, all_items=all_items, all_equipments=all_equipments, name='Kersh', max_fight_level=28, skills=['weaponcrafting', 'mining', 'woodcutting']),  # 'weaponcrafting', 'mining', 'woodcutting'
            Character(session=session, excluded_items=excluded_items, all_items=all_items, all_equipments=all_equipments, name='Capu', max_fight_level=28, skills=['gearcrafting','woodcutting', 'mining']),  # 'gearcrafting',
            Character(session=session, excluded_items=excluded_items, all_items=all_items, all_equipments=all_equipments, name='Brubu', max_fight_level=28, skills=['woodcutting', 'mining']),  # , 'fishing', 'mining', 'woodcutting'
            Character(session=session, excluded_items=excluded_items, all_items=all_items, all_equipments=all_equipments, name='Crabex', max_fight_level=28, skills=['jewelrycrafting', 'woodcutting', 'mining']),  # 'jewelrycrafting', 'woodcutting'
            Character(session=session, excluded_items=excluded_items, all_items=all_items, all_equipments=all_equipments, name='JeaGa', max_fight_level=28, skills=['mining', 'woodcutting']),  # 'cooking', 'fishing'
        ]

        # Initialize all characters asynchronously
        await asyncio.gather(*[character.initialize() for character in characters_])

        # Start the bot for all characters
        await asyncio.gather(*[run_bot(character) for character in characters_])

if __name__ == '__main__':
    logging.info("Bot started.")
    asyncio.run(main())
    logging.info("Bot finished.")
