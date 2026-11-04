from dataclasses import dataclass


@dataclass(frozen=True)
class Address:
    address: int
    source_commit: str


SOURCE_COMMIT = "3b401eaf14f193946899eb7c056d5ac02a470126"

# From pokefirered.sym and pokefirered_rev1.sym at SOURCE_COMMIT. WaitForAorBPress is a ROM
# function, so it is the one entry that differs between the revisions.
ADDRESSES = {
    sha1: {
        "gSaveBlock1": Address(0x0202552C, SOURCE_COMMIT),
        "gSaveBlock1Ptr": Address(0x03005008, SOURCE_COMMIT),
        "gSaveBlock2": Address(0x02024588, SOURCE_COMMIT),
        "gSaveBlock2Ptr": Address(0x0300500C, SOURCE_COMMIT),
        "gPlayerPartyCount": Address(0x02024029, SOURCE_COMMIT),
        "gPlayerParty": Address(0x02024284, SOURCE_COMMIT),
        "gBattleMons": Address(0x02023BE4, SOURCE_COMMIT),
        "gMain": Address(0x030030F0, SOURCE_COMMIT),
        "sMessageBoxType": Address(0x0203709C, SOURCE_COMMIT),
        "sGlobalScriptContextStatus": Address(0x03000EA8, SOURCE_COMMIT),
        "sLockFieldControls": Address(0x03000F9C, SOURCE_COMMIT),
        "sTextPrinters": Address(0x02020034, SOURCE_COMMIT),
        "sGlobalScriptContext": Address(0x03000EB0, SOURCE_COMMIT),
        "WaitForAorBPress": Address(wait_for_a_or_b, SOURCE_COMMIT),
    }
    for sha1, wait_for_a_or_b in (
        ("41cb23d8dccc8ebd7c649cd8fbb58eeace6e2fdc", 0x0806B898),
        ("dd5945db9b930750cb39d00c84da8571feebf417", 0x0806B8AC),
    )
}
