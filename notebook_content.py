import aiohttp
import asyncio
import json

# Server url
server = "https://api.artifactsmmo.com"
# Your account token (https://artifactsmmo.com/account)
token = "XXX"
# Name of your character
cooldown = 3

def handle_incorrect_status_code(_status_code: int, _character: str = 'N/A') -> int:
    if _status_code == 498:
        print(f"{_character} cannot be found on your account.")
        return -2
    elif _status_code == 497:
        print(f"{_character}'s inventory is full.")
        return -2
    elif _status_code == 499:
        print(f"{_character} is in cooldown.")
        return -1
    elif _status_code == 493:
        print(f"The resource is too high-level for {_character}.")
        return -2
    print(f"{_character} > An error occured while gathering the ressource.")
    return -2

async def get_headers(_token: str) -> dict:
  return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {_token}"
    }

async def get_items_infos(session: aiohttp.ClientSession, item_code: str) -> list:
  url = f"{server}/items/{item_code}"
  headers = await get_headers(token)

  async with session.get(url, headers=headers) as response:
    if response.status != 200:
      r = handle_incorrect_status_code(response.status)
      return r
    else:
      data = await response.json()
      return data["data"]

async def get_craft_details(session: aiohttp.ClientSession, _item_code: str) -> list:
  items_infos = await get_items_infos(session, _item_code)
  materials = items_infos['item']['craft']['items']
  return [(m['code'], m['quantity']) for m in materials]

async def get_all_maps(session: aiohttp.ClientSession, params: dict = None) -> list:
  """
  Retrieves all maps from the API.
  Returns a list of maps with their details.
  """
  if params is None:
    params = {} 
  url = f"{server}/maps/?{'&'.join([f'{k}={v}' for k,v in params.items()])}"
  headers = await get_headers(token)

  async with session.get(url, headers=headers) as response:
      if response.status != 200:
          r = handle_incorrect_status_code(response.status)
          return r
      else:
          data = await response.json()
          return data["data"]

async def get_all_resources(session: aiohttp.ClientSession) -> list:
  """
  Retrieves all maps from the API.
  Returns a list of maps with their details.
  """
  url = f"{server}/resources"
  headers = await get_headers(token)

  async with session.get(url, headers=headers) as response:
      if response.status != 200:
          r = handle_incorrect_status_code(response.status)
          return r
      else:
          data = await response.json()
          return data["data"]

async def get_all_items(session: aiohttp.ClientSession, params: dict = None) -> list:
  """
  Retrieves all items from the API.
  Returns a list of items with their details.
  """
  if params is None:
    params = {} 
  url = f"{server}/items/?{'&'.join([f'{k}={v}' for k,v in params.items()])}"
  headers = await get_headers(token)

  async with session.get(url, headers=headers) as response:
      if response.status != 200:
          r = handle_incorrect_status_code(response.status)
          return r
      else:
          data = await response.json()
          return data["data"]

async def get_crafted_item_code(session: aiohttp.ClientSession, craft_material: str, craft_skill: str) -> str:

  craftable_items = await get_all_items(
    session=session,
    params={
      'craft_material': craft_material,
      'craft_skill': craft_skill
      }
    )

  for craftable_item in craftable_items:
    if len(craftable_item['craft']['items']) == 1:
      return craftable_item['code']
  
  return 'Need more materials'

scenarii = {
    '01_build_copper_daggers': [
        ('move', 'copper_rocks'),
        ('gather', 100),
        ('move', 'forge'),
        ('craft', 'copper'),
        ('move', 'weapon_smith'),
        ('craft', 'copper_dagger'),
        ('recycle', 'copper_dagger')
    ],
    '02_build_iron_daggers': [
        ('move', 'iron_rocks'),
        ('gather', 100),
        ('move', 'forge'),
        ('craft', 'iron'),
        ('move', 'weapon_smith'),
        ('craft', 'iron_dagger'),
        ('recycle', 'iron_dagger')
    ]
}


class Character:

  def __init__(self, name: str, session: aiohttp.ClientSession):
      self.name = name
      self.session = session  
  
  async def get_infos(self) -> list:
    url = f"{server}/characters/{self.name}"
    headers = await get_headers(token)

    async with self.session.get(url, headers=headers) as response:
        if response.status != 200:
            r = handle_incorrect_status_code(response.status, self.name)
            return r
        else:
            data = await response.json()
            return data["data"]

  async def get_skill_level(self, skill_name: str):
    infos = await self.get_infos()
    return infos[f'{skill_name}_level']
  
  async def is_inventory_not_full(self) -> bool:
    infos = await self.get_infos()
    return  sum([
        i_infos['quantity'] 
        for i_infos in infos['inventory']
        ]) < infos['inventory_max_items']

  async def get_current_location(self) -> tuple[int]:
    infos = await self.get_infos()
    return (int(infos['x']), int(infos['y']))

  async def move(self, x, y):

    current_location = await self.get_current_location()

    if current_location == (x, y):
      return 0

    url = f"{server}/my/{self.name}/action/move"
    headers = await get_headers(token)
    payload = {
        "x": x,
        "y": y
    }

    async with self.session.post(url, data=json.dumps(payload), headers=headers) as response:
        if response.status != 200:
            r = handle_incorrect_status_code(response.status, self.name)
            return r
        else:
            data = await response.json()
            print(f'{self.name} > Move to ({x}, {y}). Cooldown: {data["data"]["cooldown"]["total_seconds"]}')

            # Return the cooldown in seconds
            return data["data"]["cooldown"]["total_seconds"]

  async def get_nearest_coords(self, content_type: str, content_code: str) -> list[int]:
    resource_locations = await get_all_maps(
        session=self.session, 
        params={
            'content_type': content_type,
            'content_code': content_code
            }
        )
    if len(resource_locations) == 0:
      print(f'{self.name} > No resource {content_code} on this map')
      return (0, 0)
    
    if len(resource_locations) == 1:
      return int(resource_locations[0]['x']), int(resource_locations[0]['y'])
    
    min_dist = 999999
    for resource_loc in resource_locations:
      res_x, res_y = int(resource_loc['x']), int(resource_loc['y'])
      character_location_x, character_location_y = await self.get_current_location()
      dist_to_loc = (res_x-character_location_x)**2 + (res_y-character_location_y)**2
      if dist_to_loc < min_dist:
        min_dist = dist_to_loc
        nearest_resource = resource_loc
    return (nearest_resource['x'], nearest_resource['y'])

  async def get_inventory_quantity(self, _item_code: str) -> dict[str, int]:
    character_infos = await self.get_infos()
    for item_infos in character_infos['inventory']:
      if item_infos['code'] == _item_code:
        return item_infos['quantity']
    return 0

  async def get_nb_craftable_items(self, _item_code: str) -> int:
    nb_craftable_items = 99999
    craft_details = await get_craft_details(self.session, _item_code)
    for craft_material, required_quantity in craft_details:
      print(f'{self.name} > ingredient: {craft_material} and qty {required_quantity}')
      available_material_qty = await self.get_inventory_quantity(craft_material)
      print(f'{self.name} > available {craft_material}: {available_material_qty}')
      n = available_material_qty//required_quantity
      nb_craftable_items = min(nb_craftable_items, n)
    return nb_craftable_items

  async def bank_deposit(self, item_code: str, quantity: int) -> int:
    # TODO if no qty in params -> deposit all available by default
    url = f"{server}/my/{self.name}/action/bank/deposit"
    headers = await get_headers(token)
    payload = {
        "code": item_code,
        "quantity": quantity
    }

    async with self.session.post(url, data=json.dumps(payload), headers=headers) as response:
        if response.status != 200:
            handle_incorrect_status_code(response.status, self.name)
            return -1
        else:
            data = await response.json()
            print(f'{self.name} > {quantity} {item_code} deposited. Cooldown: {data["data"]["cooldown"]["total_seconds"]}')

            # Return the cooldown in seconds
            return data["data"]["cooldown"]["total_seconds"]

  async def perform_crafting(self, item_code: str, qte: int) -> int:
    url = f"{server}/my/{self.name}/action/crafting"
    headers = await get_headers(token)
    payload = {
        "code": item_code,
        "quantity": qte
    }

    async with self.session.post(url, data=json.dumps(payload), headers=headers) as response:
        if response.status != 200:
            handle_incorrect_status_code(response.status, self.name)
            return -1
        else:
            data = await response.json()
            print(f'{self.name} > {qte} {item_code} crafted. Cooldown: {data["data"]["cooldown"]["total_seconds"]}')

            # Return the cooldown in seconds
            return data["data"]["cooldown"]["total_seconds"]

  async def perform_fighting(self) -> int:
    url = f"{server}/my/{self.name}/action/fight"
    headers = await get_headers(token)

    async with self.session.post(url, headers=headers) as response:
        if response.status != 200:
            handle_incorrect_status_code(response.status, self.name)
            return -1
        else:
            data = await response.json()
            print(f'{self.name} won. Cooldown: {data["data"]["cooldown"]["total_seconds"]}')

            # Return the cooldown in seconds
            return data["data"]["cooldown"]["total_seconds"]

  async def perform_gathering(self) -> int:
    url = f"{server}/my/{self.name}/action/gathering"
    headers = await get_headers(token)

    async with self.session.post(url, headers=headers) as response:
        if response.status != 200:
            handle_incorrect_status_code(response.status, self.name)
            return -1
        else:
            data = await response.json()
            print(f'{self.name} > Resource gathered. Cooldown: {data["data"]["cooldown"]["total_seconds"]}')

            # Return the cooldown in seconds
            return data["data"]["cooldown"]["total_seconds"]

  async def perform_recycling(self, item_code: str, qte: int) -> int:
    url = f"{server}/my/{self.name}/action/recycling"
    headers = await get_headers(token)
    payload = {
        "code": item_code,
        "quantity": qte
    }

    async with self.session.post(url, data=json.dumps(payload), headers=headers) as response:
        if response.status != 200:
            handle_incorrect_status_code(response.status, self.name)
            return -1
        else:
            data = await response.json()
            print(f'{self.name} > {qte} {item_code} recycled. Cooldown: {data["data"]["cooldown"]["total_seconds"]}')

            # Return the cooldown in seconds
            return data["data"]["cooldown"]["total_seconds"]


class Map:

  def __init__(self, session: aiohttp.ClientSession):
    self.resources: list = asyncio.run(get_all_resources(session))
    self.session = session
  
  async def get_skill_location_details(self, level: int, skill: str) -> dict:
      """
      Determines the best skill location based on the character's level.
      Uses the API to fetch map details.
      """
      # Get all maps and filter for woodcutting locations
      resources_locations = [
          loc for loc in self.resources
          if loc['skill'] == skill and loc['level'] <= level
      ]

      # If there are valid locations, return the one with the highest level requirement
      if resources_locations:
        # TODO Possible optimization depending on distance to move there
          best_location = max(resources_locations, key=lambda x: x['level'])
          return best_location

      # Return a default or error location if no match is found
      return {
          "name": "Unknown Location", 
          "code": "unknown_location", 
          "skill": skill, 
          "level": 0, 
          "drops": []
          }

# ** WOOD CUTTER / PLANK CRAFTER

async def run_bot(session: aiohttp.ClientSession, character_name: str, skill_name: str, map_object: Map):

  character_object = Character(character_name, session)

  while True:

    # GATHER
    # Get skill level
    skill_level = await character_object.get_skill_level(skill_name)
    print(f'{character_object.name} > level of {skill_name}: {skill_level}')
    # Look where the character can go to best express his skill
    location_details = await map_object.get_skill_location_details(level=skill_level, skill=skill_name)
    resource_location_code = location_details['code']

    print(f'{character_object.name} > location of {skill_name}: {resource_location_code}')
    first_resource = location_details['drops'][0]['code']
    print(f'{character_object.name} > resource: {first_resource}')
    # Move to the (TODO: nearest) wood location
    if await character_object.is_inventory_not_full():
      nearest_resource_coords = await character_object.get_nearest_coords(
          content_type='resource', 
          content_code=resource_location_code
          )
      cooldown = await character_object.move(*nearest_resource_coords)
      await asyncio.sleep(cooldown)
      # Gather wood
      while await character_object.is_inventory_not_full():
        cooldown = await character_object.perform_gathering()
        await asyncio.sleep(cooldown)
    
    # CRAFT

    # if all the necessary materials are available - else go pick them
    # what can be build solely with the first resource?

    # Move to the crafting location
    nearest_workshop_coords = await character_object.get_nearest_coords(
      content_type='workshop', 
      content_code=skill_name
      )
    print(f'{character_object.name} > location of {skill_name} workshop: {nearest_workshop_coords}')
    cooldown = await character_object.move(*nearest_workshop_coords)
    await asyncio.sleep(cooldown)

    ## Once at the workshop, what can be crafted?
    # Craft the goods
    crafted_item_code = await get_crafted_item_code(session, craft_material=first_resource, craft_skill=skill_name)
    crafted_item_qty = await character_object.get_nb_craftable_items(crafted_item_code)
    print(f'{character_object.name} > craftable items: {crafted_item_qty}')

    if crafted_item_qty > 0:
      cooldown = await character_object.perform_crafting(crafted_item_code, crafted_item_qty)
      await asyncio.sleep(cooldown)
    # Move to the bank
    bank_coords = await character_object.get_nearest_coords(
      content_type='bank', 
      content_code='bank'
      )
    cooldown = await character_object.move(*bank_coords)
    await asyncio.sleep(cooldown)
    # Deposit the goods
    available_material_qty = await character_object.get_inventory_quantity(crafted_item_code)
    print(f'{character_object.name} > available items {crafted_item_code}: {available_material_qty}')
    cooldown = await character_object.bank_deposit(crafted_item_code, available_material_qty)
    await asyncio.sleep(cooldown)

# personae
# ** WOOD CUTTER / PLANK CRAFTER
# ** FISHER / COOKING MASTER
# ** ORE MINER / BLACKSMITH
# ** WARRIOR
# + WEAPON CRAFTER
# + GEAR CRAFTER
# + JEWEL CRAFTER

###
###  Yes asyncio with aiohttp
###. Then just run all your characters in a taskgroup

  # TODO > recycle

async def main():
  async with aiohttp.ClientSession() as session:

    map_object = Map(session)

    await asyncio.gather(
      run_bot(session, 'Kersh', 'mining', map_object),
      run_bot(session, 'Capu', 'woodcutting', map_object),
      run_bot(session, 'Crabex', 'mining', map_object),
      run_bot(session, 'Brubu', 'woodcutting', map_object),
      run_bot(session, 'JeaGa', 'fishing', map_object)
    )

asyncio.run(main())
