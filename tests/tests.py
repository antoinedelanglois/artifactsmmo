import unittest
from main import Item, Effect, Monster, calculate_weapon_score


class TestEquipmentSelection(unittest.TestCase):
    def test_weapon_choice(self):
        test_weapon = Item(
            name="fire_sword",
            code="fire_sword",
            level=10,
            type="weapon",
            subtype="sword",
            description="hot",
            effects=[Effect(name="attack_fire", value=50)]
        )
        test_monster = Monster(
            name="Icy",
            code="ice_monster",
            level=10,
            hp=666,
            res_fire=-50,  # Vulnerable to fire
            res_earth=0,
            res_water=50,
            res_air=0,
            attack_fire=12,
            attack_earth=12,
            attack_water=12,
            attack_air=12,
            min_gold=1,
            max_gold=2,
            drops=[]
        )

        vulnerabilities = test_monster.get_vulnerabilities()
        weapon_effects = {effect.name: effect.value for effect in test_weapon.effects}
        weapon_score = calculate_weapon_score(weapon_effects, vulnerabilities)
        print(f"Weapon Score: {weapon_score}")


if __name__ == '__main__':
    unittest.main()