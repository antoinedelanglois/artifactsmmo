# Server url
SERVER = "https://api.artifactsmmo.com"
SLOT_TYPE_MAPPING = {
    'weapon': ['weapon'],
    'shield': ['shield'],
    'helmet': ['helmet'],
    'body_armor': ['body_armor'],
    'leg_armor': ['leg_armor'],
    'boots': ['boots'],
    'ring': ['ring1', 'ring2'],
    'amulet': ['amulet'],
    # Manual management of artifacts
    # 'artifact': ['artifact1', 'artifact2', 'artifact3']
}
EQUIPMENTS_TYPES = list(SLOT_TYPE_MAPPING.keys())
EQUIPMENTS_SLOTS = [x for slot in SLOT_TYPE_MAPPING.values() for x in slot]
EXCLUDED_MONSTERS = ["cultist_emperor", "bat"]
SPAWN_COORDINATES = (0, 0)
BANK_COORDINATES = (4, 1)
STOCK_QTY_OBJECTIVE = 500
