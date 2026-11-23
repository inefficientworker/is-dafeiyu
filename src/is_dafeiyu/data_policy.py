"""Shared category policy for easy real-world aquatic negatives."""

REAL_AQUATIC_SUBCATEGORIES = frozenset(
    {
        "wiki_betta_blue",
        "wiki_blue_tang",
        "wiki_blue_whale",
        "wiki_clownfish",
        "wiki_coral_fish",
        "wiki_dolphin",
        "wiki_goldfish",
        "wiki_koi",
        "wiki_orca",
        "wiki_shark",
    }
)


def is_real_aquatic(row: dict) -> bool:
    return int(row["label"]) == 0 and row["subcategory"] in REAL_AQUATIC_SUBCATEGORIES
