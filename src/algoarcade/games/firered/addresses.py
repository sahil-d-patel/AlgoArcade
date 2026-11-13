from dataclasses import dataclass


@dataclass(frozen=True)
class Address:
    address: int
    source_commit: str


SOURCE_COMMIT = "3b401eaf14f193946899eb7c056d5ac02a470126"

# From pokefirered.sym and pokefirered_rev1.sym at SOURCE_COMMIT. The ROM functions, listed per
# revision at the bottom, are the only entries that differ between the revisions.
ADDRESSES = {
    sha1: {
        # The overworld for the agent's tools (games/firered/field.py).
        "gObjectEvents": Address(0x02036E38, SOURCE_COMMIT),
        "gStringVar4": Address(0x02021D18, SOURCE_COMMIT),
        "gDisplayedStringBattle": Address(0x0202298C, SOURCE_COMMIT),
        "gTasks": Address(0x03005090, SOURCE_COMMIT),
        **{name: Address(address, SOURCE_COMMIT) for name, address in functions.items()},
        "gSaveBlock1": Address(0x0202552C, SOURCE_COMMIT),
        "gSaveBlock1Ptr": Address(0x03005008, SOURCE_COMMIT),
        "gSaveBlock2": Address(0x02024588, SOURCE_COMMIT),
        "gSaveBlock2Ptr": Address(0x0300500C, SOURCE_COMMIT),
        "gPlayerPartyCount": Address(0x02024029, SOURCE_COMMIT),
        "gPlayerParty": Address(0x02024284, SOURCE_COMMIT),
        "gBattleMons": Address(0x02023BE4, SOURCE_COMMIT),
        "gBattlerPartyIndexes": Address(0x02023BCE, SOURCE_COMMIT),
        "gEnemyParty": Address(0x0202402C, SOURCE_COMMIT),
        "gMain": Address(0x030030F0, SOURCE_COMMIT),
        "sMessageBoxType": Address(0x0203709C, SOURCE_COMMIT),
        "sGlobalScriptContextStatus": Address(0x03000EA8, SOURCE_COMMIT),
        "sLockFieldControls": Address(0x03000F9C, SOURCE_COMMIT),
        "sTextPrinters": Address(0x02020034, SOURCE_COMMIT),
        "sGlobalScriptContext": Address(0x03000EB0, SOURCE_COMMIT),
        "VMap": Address(0x03005040, SOURCE_COMMIT),
        "gBackupMapData": Address(0x02031DFC, SOURCE_COMMIT),
    }
    for sha1, functions in (
        (
            "41cb23d8dccc8ebd7c649cd8fbb58eeace6e2fdc",
            {
                "WaitForAorBPress": 0x0806B898,
                "Task_YesNoMenu_HandleInput": 0x0809CE54,
                "Task_MultichoiceMenu_HandleInput": 0x0809CC98,
                "Hask_MultichoiceGridMenu_HandleInput": 0x0809CFDC,
            },
        ),
        (
            "dd5945db9b930750cb39d00c84da8571feebf417",
            {
                "WaitForAorBPress": 0x0806B8AC,
                "Task_YesNoMenu_HandleInput": 0x0809CE68,
                "Task_MultichoiceMenu_HandleInput": 0x0809CCAC,
                "Hask_MultichoiceGridMenu_HandleInput": 0x0809CFF0,
            },
        ),
    )
}


@dataclass(frozen=True)
class GameId:
    value: int
    source_file: str


LAYOUT_COMMIT = "e93eb12a336fb97df420b0961753d8a3713c9fcc"

# Flag, var and item ids, the same on both revisions. The key is the symbol in source_file of
# pret/pokefirered at LAYOUT_COMMIT.
IDS = {
    "FLAG_GOT_POTION_ON_ROUTE_1": GameId(0x230, "include/constants/flags.h"),
    "FLAG_GOT_TM39_FROM_BROCK": GameId(0x254, "include/constants/flags.h"),
    "FLAG_BEAT_RIVAL_IN_OAKS_LAB": GameId(0x258, "include/constants/flags.h"),
    "FLAG_VISITED_OAKS_LAB": GameId(0x2CF, "include/constants/flags.h"),
    "FLAG_DEFEATED_BROCK": GameId(0x4B0, "include/constants/flags.h"),
    "FLAG_BADGE01_GET": GameId(0x820, "include/constants/flags.h"),
    "FLAG_SYS_POKEMON_GET": GameId(0x828, "include/constants/flags.h"),
    "FLAG_SYS_POKEDEX_GET": GameId(0x829, "include/constants/flags.h"),
    "FLAG_WORLD_MAP_PEWTER_CITY": GameId(0x892, "include/constants/flags.h"),
    "VAR_MAP_SCENE_VIRIDIAN_CITY_MART": GameId(0x4057, "include/constants/vars.h"),
    "ITEM_OAKS_PARCEL": GameId(349, "include/constants/items.h"),
}
