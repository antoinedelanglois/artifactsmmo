from constants import EQUIPMENTS_SLOTS
from models import Item
from typing import Optional
import logging


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
            pass
        elif effect_name == "hp":
            score += effect_value * 0.25
        elif effect_name == "defense":
            score += effect_value * 0.5
        else:
            score += effect_value
    return score


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

    return best_equipment


async def select_best_equipment_set(
        current_equipments: dict[str, Item],
        sorted_valid_equipments: dict[str, list[Item]],
        vulnerabilities: dict
) -> dict[str, Item]:
    selected_equipments = {}

    best_weapon = select_best_weapon(
        current_equipments.get('weapon', None),       # FIXME None is not a valid Item
        sorted_valid_equipments.get('weapon', []),
        vulnerabilities
    )
    selected_equipments['weapon'] = best_weapon

    for slot in EQUIPMENTS_SLOTS:
        if slot == 'weapon':
            continue
        current_item = current_equipments.get(slot, {})
        equipment_list = sorted_valid_equipments.get(slot, [])
        best_item = select_best_support_equipment(current_item, equipment_list, vulnerabilities, best_weapon)
        selected_equipments[slot] = best_item

    return selected_equipments
