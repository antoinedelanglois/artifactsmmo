# Server url
SERVER = "https://api.artifactsmmo.com"
EQUIPMENTS_SLOTS = ['weapon', 'shield', 'helmet', 'body_armor', 'leg_armor', 'boots', 'ring1', 'ring2',
                    'amulet', 'artifact1', 'artifact2', 'artifact3']
EQUIPMENTS_TYPES = ['weapon', 'shield', 'helmet', 'body_armor', 'leg_armor', 'boots', 'ring',
                    'amulet', 'artifact']
SLOT_TYPE_MAPPING = {
    'weapon': ['weapon'],
    'shield': ['shield'],
    'helmet': ['helmet'],
    'body_armor': ['body_armor'],
    'leg_armor': ['leg_armor'],
    'boots': ['boots'],
    'ring': ['ring1', 'ring2'],
    'amulet': ['amulet'],
    'artifact': ['artifact1', 'artifact2', 'artifact3'],
    # Add other mappings if necessary
}
EXCLUDED_MONSTERS = ["cultist_acolyte", "cultist_emperor", "lich", "bat"]
SPAWN_COORDINATES = (0, 0)
BANK_COORDINATES = (4, 1)
STOCK_QTY_OBJECTIVE = 500
