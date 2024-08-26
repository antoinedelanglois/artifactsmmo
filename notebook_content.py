import aiohttp
import asyncio
import json
import nest_asyncio
nest_asyncio.apply()

# Server url
server = "https://api.artifactsmmo.com"
# Your account token (https://artifactsmmo.com/account)
token = "XXX"
# Name of your character
characters = ["Kersh", "Capu", "Brubu", "Crabex", "JeaGa"]
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
    print(f"{_character} > An error occured {_status_code}")
    return -2

async def get_headers(_token: str) -> dict:
  return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {_token}"
    }

async def get_bank_items(session: aiohttp.ClientSession) -> dict:
  url = f"{server}/my/bank/items/"
  headers = await get_headers(token)

  async with session.get(url, headers=headers) as response:
    if response.status != 200:
      r = handle_incorrect_status_code(response.status, f'GET BANK ITEMS')
      return r
    else:
      data = await response.json()
      return {item['code']: item['quantity'] for item in data["data"]}

async def get_item_infos(session: aiohttp.ClientSession, _item_code: str) -> dict:
  """
  Retrieves one item details from the API.
  Returns a list of items with their details.
  """
  url = f"{server}/items/{_item_code}"
  headers = await get_headers(token)

  async with session.get(url, headers=headers) as response:
    if response.status != 200:
      r = handle_incorrect_status_code(response.status, f'GET ITEMS INFOS {_item_code}')
      return r
    else:
      data = await response.json()
      # print(f'DEBUG get_item_infos for {_item_code} > {data}')
      return data["data"]["item"]

async def get_craft_details(session: aiohttp.ClientSession, _item_code: str) -> list:
  item_infos = await get_item_infos(session, _item_code)
  # print(f'DEBUG > {item_infos}')
  if item_infos['craft'] is not None:
    materials = item_infos['craft']['items']
    # print(f'DEBUG > materials {materials}')
    return [(m['code'], m['quantity']) for m in materials]
  return []

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

async def get_all_resources(session: aiohttp.ClientSession, params: dict = None) -> list:
  """
  Retrieves all maps from the API.
  Returns a list of maps with their details.
  """
  if params is None:
    params = {}
  url = f"{server}/resources/?{'&'.join([f'{k}={v}' for k,v in params.items()])}"
  headers = await get_headers(token)

  async with session.get(url, headers=headers) as response:
      if response.status != 200:
          r = handle_incorrect_status_code(response.status, 'GET ALL RESOURCES')
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
  elif params['craft_skill'] == 'fishing':
    params['craft_skill'] = 'cooking'
  url = f"{server}/items/?{'&'.join([f'{k}={v}' for k,v in params.items()])}"
  headers = await get_headers(token)

  async with session.get(url, headers=headers) as response:
      if response.status != 200:
          r = handle_incorrect_status_code(response.status, f'GET ALL ITEMS {params}')
          return r
      else:
          data = await response.json()
          return data["data"]

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

  def __init__(self, session: aiohttp.ClientSession, name: str, gathering_skill_name: str, crafting_skill_name: str):
      self.name = name
      self.session = session
      self.gathering_skill_name = gathering_skill_name
      self.crafting_skill_name = crafting_skill_name

  async def set_gathering_skill(self, gathering_skill_name: str):
    self.gathering_skill_name = gathering_skill_name

  async def set_crafting_skill(self, crafting_skill_name: str):
    self.crafting_skill_name = crafting_skill_name

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

  async def get_craftable_items(self, craft_skill: str) -> dict:
    skill_level = await self.get_skill_level(craft_skill)
    infos = await self.get_infos()
    craftable_items = await get_all_items(
      session=self.session,
      params={
        'max_level': skill_level,
        'craft_skill': craft_skill
        }
      )
    craftable_items = {
        item['code']: await self.get_nb_craftable_items(item['code'])
        for item in craftable_items
    }
    return {
        item_code: qty
        for item_code, qty in craftable_items.items()
        if qty > 0
    }

  async def get_potentially_craftable_items(self, craft_skill: str) -> list[dict]:
    skill_level = await self.get_skill_level(craft_skill)
    infos = await self.get_infos()
    potentially_craftable_items = await get_all_items(
      session=self.session,
      params={
        'max_level': skill_level,
        'craft_skill': craft_skill
        }
      )
    return potentially_craftable_items

   #   "type": "resource",
   #   "subtype": "woodcutting",

  async def get_depositable_items(self) -> dict:
    infos = await self.get_infos()
    depositable_items = {}
    for item_infos in infos['inventory']:
      if item_infos['code']:  # There are '' items in inventory...
        # print(f'DEBUG get_depositable_items > Check craft details for {item_infos}')
        craft_details = await get_craft_details(self.session, item_infos['code'])
        if len(craft_details) > 0 or item_infos['code'] in ['birch_wood', 'coal']:
          depositable_items[item_infos['code']] = item_infos['quantity']
    return depositable_items

  async def get_skill_level(self, skill_name: str):
    infos = await self.get_infos()
    return infos[f'{skill_name}_level']

  async def get_inventory_size(self) -> int:
    infos = await self.get_infos()
    return sum([
        i_infos['quantity']
        for i_infos in infos['inventory']
        ])

  async def get_inventory_items(self) -> dict:
    infos = await self.get_infos()
    return {
        i_infos['code']: i_infos['quantity']
        for i_infos in infos['inventory']
    }

  async def is_inventory_not_full(self) -> bool:
    infos = await self.get_infos()
    return  self.get_inventory_size() < infos['inventory_max_items']

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
    # print(f'{self.name} > Locations for resource {content_code} and {content_type}: {resource_locations}')
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
      # print(f'{self.name} > ingredient: {craft_material} and qty {required_quantity}')
      available_material_qty = await self.get_inventory_quantity(craft_material)
      # print(f'{self.name} > available {craft_material}: {available_material_qty}')
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

  async def bank_withdraw(self, item_code: str, quantity: int) -> int:
    url = f"{server}/my/{self.name}/action/bank/withdraw"
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
            print(f'{self.name} > {quantity} {item_code} withdrawed. Cooldown: {data["data"]["cooldown"]["total_seconds"]}')

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

async def run_bot(character_object: Character, map_object: Map):

  while True:

    # GATHER

    # Get gathering skill level
    skill_level = await character_object.get_skill_level(character_object.gathering_skill_name)
    print(f'{character_object.name} > level of {character_object.gathering_skill_name}: {skill_level}')

    # Get crafting skill level
    crafting_level = await character_object.get_skill_level(character_object.crafting_skill_name)
    print(f'{character_object.name} > level of {character_object.crafting_skill_name}: {crafting_level}')
    # What can they craft (thus what should they gather?)

    # TODO If material is available in bank and secondary crafting skill set, let's craft some

    # Get the craftable items depending on level
    craftable_items = await character_object.get_potentially_craftable_items(character_object.gathering_skill_name)

    if len(craftable_items) > 0:
      sorted_craftable_items = sorted(craftable_items, key=lambda x: x['level'], reverse=True)
      print(f'{character_object.name} > sorted items: {sorted_craftable_items}')
      # Select the highest level item
      highest_level_item = sorted_craftable_items[0]
      print(f'{character_object.name} > highest level item: {highest_level_item}')

      # Define gathering objectives accordingly (depending of available room in inventory)
      # Can either collect it by itself or get it out of bank
      # Loop on successive craftable items to select the most eligible
      for craftable_item in sorted_craftable_items:
        print(f'{character_object.name} > Check reachability of materials for craftable item: {craftable_item}')
        # What are the materials?
        craft_materials = craftable_item['craft']['items']
        craft_skill = craftable_item['craft']['skill']
        craft_level = craftable_item['craft']['level']
        print(f'{character_object.name} > materials: {craft_materials} + skill {craft_skill} + level {craft_level}')
        character_skill_level = await character_object.get_skill_level(craft_skill)
        if character_skill_level >= craft_level:
          # Get a strategy to gather/collect the right portions of materials
          character_infos = await character_object.get_infos()
          total_inventory_size = int(character_infos['inventory_max_items'])
          total_nb_materials = sum([material['quantity'] for material in craft_materials])

          # If inventory is not empty > go and deposit all of it
          if await character_object.get_inventory_size() > 0:
            # Go to bank and deposit all objects
            # Move to the bank
            bank_coords = await character_object.get_nearest_coords(
              content_type='bank',
              content_code='bank'
              )
            cooldown = await character_object.move(*bank_coords)
            await asyncio.sleep(cooldown)

            # Deposit all the items in the inventory
            # TODO get a method for full deposit
            inventory_items = await character_object.get_inventory_items()
            for item_code, item_qty in inventory_items.items():
              cooldown = await character_object.bank_deposit(item_code, item_qty)
              await asyncio.sleep(cooldown)

          for material in craft_materials:

            # If character already got some
            qty_in_bag = await character_object.get_inventory_quantity(material["code"])


            qty_to_get = round(total_inventory_size * int(material['quantity']) / total_nb_materials) - qty_in_bag
            print(f'{character_object.name} > let gather {qty_to_get} {material["code"]}')
            # if not at bank in the correct qty, go and gather it
            # Check the bank
            bank_items = await get_bank_items(character_object.session)
            # print(f'bank_items > {bank_items}')
            if bank_items.get(material['code'], 0) >= qty_to_get:
              # Go and get the materials at the bank

              # Move to the bank
              bank_coords = await character_object.get_nearest_coords(
                content_type='bank',
                content_code='bank'
                )
              cooldown = await character_object.move(*bank_coords)
              await asyncio.sleep(cooldown)

              # Get the materials
              cooldown = await character_object.bank_withdraw(material['code'], qty_to_get)
              await asyncio.sleep(cooldown)

            else:
              # Go and gather the material

              # TODO get the currently available materials in bank, while on site?

              # TODO get the location specific to the target material
              # Look where the character can go to best express his skill
              location_details = await map_object.get_skill_location_details(level=skill_level, skill=character_object.gathering_skill_name)
              resource_location_code = location_details['code']
              print(f'{character_object.name} > location of {character_object.gathering_skill_name}: {resource_location_code}')

              # Go to the resource location
              nearest_resource_coords = await character_object.get_nearest_coords(
                  content_type='resource',
                  content_code=resource_location_code
                  )
              cooldown = await character_object.move(*nearest_resource_coords)
              await asyncio.sleep(cooldown)

              # Gather
              while await character_object.get_inventory_quantity(material['code']) < qty_to_get:
                cooldown = await character_object.perform_gathering()
                await asyncio.sleep(cooldown)

          item_to_craft = craftable_item['code']
          print(f'{character_object.name} > item to craft {item_to_craft}')
          break

    # TRANSFORM

    nb_items_to_craft = await character_object.get_nb_craftable_items(item_to_craft)


    # Move to the crafting location
    nearest_workshop_coords = await character_object.get_nearest_coords(
      content_type='workshop',
      content_code='cooking' if character_object.gathering_skill_name == 'fishing' else character_object.gathering_skill_name
      )
    print(f'{character_object.name} > location of {character_object.gathering_skill_name} workshop: {nearest_workshop_coords}')
    cooldown = await character_object.move(*nearest_workshop_coords)
    await asyncio.sleep(cooldown)

    cooldown = await character_object.perform_crafting(item_to_craft, nb_items_to_craft)
    await asyncio.sleep(cooldown)


    # TODO recycle

# personae
# ** WOOD CUTTER / PLANK CRAFTER
# ** FISHER / COOKING MASTER
# ** ORE MINER / BLACKSMITH
# ** WARRIOR
# + WEAPON CRAFTER
# + GEAR CRAFTER
# + JEWEL CRAFTER

  # TODO > recycle

async def main():
  async with aiohttp.ClientSession() as session:

    map_object = Map(session)

    # Create characters
    characters = [
        Character(session, 'Kersh', 'mining', 'weaponcrafting'),
        Character(session, 'Capu', 'woodcutting', 'gearcrafting'),
        Character(session, 'Crabex', 'woodcutting', 'jewelrycrafting'),
        Character(session, 'Brubu', 'mining', 'mining'),
        Character(session, 'JeaGa', 'fishing', 'cooking'),
    ]

    await asyncio.gather(*[run_bot(character, map_object) for character in characters])

await main()
