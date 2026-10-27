from dataclasses import dataclass


@dataclass(frozen=True)
class Address:
    address: int
    source_commit: str


SOURCE_COMMIT = "3b401eaf14f193946899eb7c056d5ac02a470126"

# Verified independently against both revisions' symbol files at SOURCE_COMMIT.
ADDRESSES = {
    sha1: {
        "gSaveBlock1": Address(0x0202552C, SOURCE_COMMIT),
        "gSaveBlock1Ptr": Address(0x03005008, SOURCE_COMMIT),
        "gSaveBlock2Ptr": Address(0x0300500C, SOURCE_COMMIT),
        "gPlayerPartyCount": Address(0x02024029, SOURCE_COMMIT),
        "gPlayerParty": Address(0x02024284, SOURCE_COMMIT),
        "gBattleMons": Address(0x02023BE4, SOURCE_COMMIT),
        "gMain": Address(0x030030F0, SOURCE_COMMIT),
        "sMessageBoxType": Address(0x0203709C, SOURCE_COMMIT),
        "sGlobalScriptContextStatus": Address(0x03000EA8, SOURCE_COMMIT),
        "sLockFieldControls": Address(0x03000F9C, SOURCE_COMMIT),
    }
    for sha1 in (
        "41cb23d8dccc8ebd7c649cd8fbb58eeace6e2fdc",
        "dd5945db9b930750cb39d00c84da8571feebf417",
    )
}
