from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from enum import Enum
from datetime import datetime
from constants import EQUIPMENTS_TYPES


class TaskType(Enum):
    RESOURCES = "resources"
    MONSTERS = "monsters"
    ITEMS = "items"
    RECYCLE = "recycle"
    IDLE = "idle"


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

    # TODO use CharacterInfos
    def does_provide_xp(self, character_infos: dict, max_level: int) -> bool:
        character_level = character_infos["level"]
        return self.level >= (character_level - 10) and character_level < max_level


class Resource(BaseModel):
    name: str
    code: str
    skill: str
    level: int
    drops: list[Drop]

    def get_skill_name(self):
        return self.skill


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

    @staticmethod
    def get_event_craft_items():
        return ['strangold', 'obsidian', 'magical_plank']

    @staticmethod
    def get_event_gather_items():
        return ['strange_ore', 'piece_of_obsidian', 'magic_wood']

    def get_task_type(self) -> TaskType:
        if self.is_gatherable():
            return TaskType.RESOURCES
        return TaskType.ITEMS

    # TODO use CharacterInfos
    def get_max_taskable_quantity(self, inventory_max_size: int) -> int:
        return inventory_max_size // self.get_nb_ingredients()

    def has_protected_ingredients(self) -> bool:
        craft_recipee = self.get_craft_recipee()
        ingredients = list(craft_recipee.keys())
        return any([ingredient in ["jasper_crystal", "magical_cure"] for ingredient in ingredients])

    def get_nb_ingredients(self) -> int:
        craft_recipee = self.get_craft_recipee()
        return sum([qty for _, qty in craft_recipee.items()])

    def get_craft_recipee(self) -> dict[str, int]:
        if self.craft is not None:
            return {m['code']: m['quantity'] for m in self.craft.items}
        return {self.code: 1}

    def is_gatherable(self) -> bool:
        return self.craft is None and self.type == "resource" and self.subtype in ["mining", "woodcutting", "fishing"]

    def is_from_task(self) -> bool:
        return self.type == "resource" and self.subtype == "task"

    def is_from_event(self) -> bool:
        return self.code in Item.get_event_gather_items() or self.code in Item.get_event_craft_items()

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

    def get_skill_name(self) -> str:
        if self.craft:
            return self.craft.skill
        if self.is_gatherable():
            return self.subtype
        return ""

    def get_skill_level(self) -> int:
        if self.craft:
            return self.craft.level
        if self.is_gatherable():
            return self.level
        return self.level

    def does_provide_xp(self, character_infos: dict, max_level: int) -> bool:

        item_skill_name = self.get_skill_name()
        item_skill_level = self.get_skill_level()

        skill_level_key = f'{item_skill_name}_level' if item_skill_name else 'level'
        skill_level = character_infos.get(skill_level_key)

        return skill_level is not None and skill_level - 10 <= item_skill_level and skill_level < max_level


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


class BankDetails(BaseModel):
    slots: int = 0
    expansions: int = 0
    next_expansion_cost: int = 0
    gold: int = 0


class Environment(BaseModel):
    items: dict[str, Item]
    monsters: dict[str, Monster]
    resource_locations: dict[str, Resource]
    maps: list[dict]
    status: Status
    bank_details: BankDetails
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

    def get_obsolete_equipments(self, all_items_quantities: dict[str, int]) -> dict[str, Item]:
        equipment_groups = self.get_equipments_by_type()

        map_equipments = {}
        for equipment_type, equipments in equipment_groups.items():
            filtered_equipments = []
            for equipment in equipments:
                if all_items_quantities.get(equipment.code, 0) >= equipment.get_min_stock_qty():
                    filtered_equipments.append(equipment)
            map_equipments[equipment_type] = filtered_equipments

        obsolete_equipments = identify_obsolete_equipments(map_equipments)
        return {equipment.code: equipment for equipment in obsolete_equipments}

    def get_items_list_by_type(self, _item_type: str) -> list[Item]:
        return [
            item
            for _, item in self.items.items()
            if item.type == _item_type
        ]

    def get_equipments_by_type(self) -> dict[str, list[Item]]:
        return {
            equipment_type: self.get_items_list_by_type(equipment_type)
            for equipment_type in EQUIPMENTS_TYPES

        }

    def get_craftable_items(self, character_infos: dict) -> list[Item]:
        return [
            item
            for item in self.crafted_items
            if item.level <= character_infos[f'{item.craft.skill}_level']
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
