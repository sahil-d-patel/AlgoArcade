import pyarrow as pa
import pytest

from algoarcade.logs.schema import REWARD_COMPONENTS, TABLES, SchemaError, validate


def valid_row(schema: pa.Schema) -> dict[str, list]:
    examples = {pa.string(): "x", pa.bool_(): False, pa.float64(): 0.5}
    return {f.name: [examples.get(f.type, 1)] for f in schema}


@pytest.mark.parametrize("table", sorted(TABLES))
def test_a_valid_batch_passes(table: str) -> None:
    schema = TABLES[table]
    validate(pa.table(valid_row(schema), schema=schema), table)


@pytest.mark.parametrize("table", sorted(TABLES))
def test_a_missing_field_is_rejected(table: str) -> None:
    batch = pa.table(valid_row(TABLES[table]), schema=TABLES[table])
    with pytest.raises(SchemaError, match="missing"):
        validate(batch.drop_columns([batch.column_names[0]]), table)


@pytest.mark.parametrize("table", sorted(TABLES))
def test_an_extra_field_is_rejected(table: str) -> None:
    batch = pa.table(valid_row(TABLES[table]), schema=TABLES[table])
    with pytest.raises(SchemaError, match="extra"):
        validate(batch.append_column("reward_bonus", pa.array([1.0])), table)


@pytest.mark.parametrize("table", sorted(TABLES))
def test_a_wrong_type_is_rejected(table: str) -> None:
    schema = TABLES[table]
    batch = pa.table(valid_row(schema), schema=schema)
    i = schema.get_field_index("run_id")
    with pytest.raises(SchemaError, match="run_id"):
        validate(batch.set_column(i, "run_id", pa.array([1], pa.int64())), table)


def test_a_wider_integer_is_rejected() -> None:
    schema = TABLES["steps"]
    batch = pa.table(valid_row(schema), schema=schema)
    i = schema.get_field_index("x")
    with pytest.raises(SchemaError, match="x"):
        validate(batch.set_column(i, "x", pa.array([1], pa.int64())), "steps")


def test_a_null_in_a_required_field_is_rejected() -> None:
    schema = TABLES["steps"]
    row = valid_row(schema)
    row["map_group"] = [None]
    all_nullable = pa.schema([f.with_nullable(True) for f in schema])
    with pytest.raises(SchemaError, match="map_group"):
        validate(pa.table(row, schema=all_nullable), "steps")


def test_column_order_does_not_matter() -> None:
    schema = TABLES["steps"]
    batch = pa.table(valid_row(schema), schema=schema)
    validate(batch.select(list(reversed(batch.column_names))), "steps")


def test_every_component_has_a_column_in_steps_and_episodes() -> None:
    for table in ("steps", "episodes"):
        names = TABLES[table].names
        assert [n for n in names if n.startswith("reward_") and n != "reward_total"] == [
            f"reward_{c}" for c in REWARD_COMPONENTS
        ]


def test_reward_components_match_the_technical_plan() -> None:
    assert REWARD_COMPONENTS == (
        "events",
        "explore",
        "level",
        "badge",
        "milestone",
        "heal",
        "blackout",
        "anti_loop",
        "anti_spam",
        "time",
    )
