import logging
import asyncio
from aiohttp import ClientSession
from models import Status, Item, Monster, Resource, Announcement, BankDetails, CharacterInfos
from aiohttp.client_exceptions import ClientConnectorError
import os
import re
from dotenv import load_dotenv
from datetime import datetime
import pytz
from constants import SERVER, EQUIPMENTS_SLOTS
import random

UTC = pytz.UTC

load_dotenv()
TOKEN = os.getenv('ARTIFACTSMMO_TOKEN')
if not TOKEN:
    raise ValueError("TOKEN is not defined - Add it to .env")


API_RATE_LIMIT = 5
API_SEMAPHORE = asyncio.Semaphore(API_RATE_LIMIT)


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
        case _: msg = f"An error occurred {_status_code}"
    return msg


def extract_cooldown_time(message):
    """Extracts cooldown time from the message using regex."""
    match = re.search(r"Character in cooldown: (\d+)", message)
    if match:
        return int(match.group(1))
    return None


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

    async with API_SEMAPHORE:
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
                        # Implementing exponential backoff with jitter
                        retry_after = int(response.headers.get('Retry-After', '1'))
                        backoff_time = min(2 ** attempt + random.uniform(0, 1), 60)  # Add jitter
                        logging.warning(f"Rate limited. "
                                        f"Retrying after {backoff_time} seconds (Retry-After: {retry_after}).")
                        await asyncio.sleep(max(backoff_time, retry_after))
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
                        logging.warning(f"Request to {url} failed. Status {response.status}. {error_msg}. Retrying...")
            except asyncio.TimeoutError as e:
                logging.exception(f"Request to {url} timed out. Retrying ({attempt + 1}/{retries})... ({e})")
                await asyncio.sleep(min(2 ** attempt, 60))
            except ClientConnectorError as e:
                logging.exception(f"Connection error while accessing {url}: "
                                  f"{str(e)}. Retrying ({attempt + 1}/{retries})...")
                await asyncio.sleep(min(2 ** attempt, 60))
            except asyncio.CancelledError:
                logging.error("Request was cancelled. Cleaning up...")
                raise  # Re-raise to ensure proper task cancellation handling
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)} while making request to {url}.")
                break  # Break out of the loop if an unexpected error occurs

        logging.error(f"Failed to make request to {url} after {retries} attempts.")
        return None


async def get_character_move(session: ClientSession, name: str, x: int, y: int) -> dict:
    url = f"{SERVER}/my/{name}/action/move"
    payload = {"x": x, "y": y}
    data = await make_request(session=session, method='POST', url=url, payload=payload)
    return data["data"] if data else {}


async def get_status(session: ClientSession) -> dict:
    url = f"{SERVER}/"
    data = await make_request(session=session, method='GET', url=url)
    return data["data"] if data else {}


async def get_all_status(session: ClientSession) -> Status:
    status_data = await get_status(session)
    if not status_data:
        # Handle the case where status_data is None or empty
        return Status(**{
            "status": "unknown",
            "version": "unknown",
            "max_level": 40,
            "characters_online": 0,
            "server_time": datetime.now(UTC),
            "announcements": [],
            "last_wipe": "",
            "next_wipe": ""
        })

    # Convert server_time to datetime if necessary
    server_time = status_data.get("server_time")
    if isinstance(server_time, str):
        server_time = datetime.fromisoformat(server_time.replace("Z", "+00:00"))

    # Process announcements if they are provided
    announcements_data = status_data.get("announcements", [])
    announcements = [Announcement(**a) for a in announcements_data]

    return Status(**{
        "status": status_data.get("status", ""),
        "version": status_data.get("version", ""),
        "max_level": status_data.get("max_level", 40),
        "characters_online": status_data.get("characters_online", 0),
        "server_time": server_time,
        "announcements": announcements,
        "last_wipe": status_data.get("last_wipe", ""),
        "next_wipe": status_data.get("next_wipe", "")
    })


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


async def get_character_infos(session: ClientSession, name: str) -> dict:
    url = f"{SERVER}/characters/{name}"
    data = await make_request(session=session, method='GET', url=url)
    return data["data"] if data else {}


async def get_all_infos(session: ClientSession, name: str) -> CharacterInfos:
    infos = await get_character_infos(session, name)
    return CharacterInfos(**infos)


async def get_my_characters(session: ClientSession) -> list:
    url = f"{SERVER}/my/characters"
    data = await make_request(session=session, method='GET', url=url)
    return data["data"] if data else []


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


async def get_bank_details(session: ClientSession) -> dict[str, int]:
    url = f"{SERVER}/my/bank/"
    data = await make_request(session=session, method='GET', url=url)
    return data["data"] if data else {}


async def get_all_bank_details(session: ClientSession) -> BankDetails:
    bank_details_data = await get_bank_details(session)
    return BankDetails(**bank_details_data)


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


async def needs_stock(session: ClientSession, _item: Item, total_quantities: dict[str, int] = None) -> bool:
    if total_quantities is None:
        total_quantities = await get_all_items_quantities(session)
    return await get_all_map_item_qty(session, _item, total_quantities) < _item.get_min_stock_qty()


async def get_bank_item_qty(session: ClientSession, _item_code: str) -> int:
    res = await get_bank_items(session=session, params={"item_code": _item_code})
    return res.get(_item_code, 0)


async def get_all_map_item_qty(session: ClientSession, _item: Item, total_quantities: dict[str, int] = None) -> int:
    if total_quantities is None:
        total_quantities = await get_all_items_quantities(session)
    return total_quantities.get(_item.code, 0)
