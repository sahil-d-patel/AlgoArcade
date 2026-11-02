"""The four logged tables. Contract: docs/harness-spec.md, Logging schema."""

import pyarrow as pa

# The ten components in docs/technical-plan.md, Reward design. Each logs its raw value as
# reward_<name>, null when the run's config leaves it out.
REWARD_COMPONENTS = (
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


def _required(name: str, type_: pa.DataType) -> pa.Field:
    return pa.field(name, type_, nullable=False)


_REWARDS = [pa.field(f"reward_{c}", pa.float64()) for c in REWARD_COMPONENTS]

STEPS = pa.schema(
    [
        _required("run_id", pa.string()),
        _required("seed", pa.int32()),
        _required("env_idx", pa.int32()),
        _required("env_steps", pa.int64()),
        _required("wall_clock_s", pa.float64()),
        _required("episode_id", pa.int32()),
        _required("episode_step", pa.int32()),
        _required("action", pa.int16()),
        _required("map_group", pa.int16()),
        _required("map_num", pa.int16()),
        _required("x", pa.int16()),
        _required("y", pa.int16()),
        _required("in_battle", pa.bool_()),
        _required("milestone_count", pa.int16()),
        _required("reward_total", pa.float64()),
        *_REWARDS,
    ]
)

VISITS = pa.schema(
    [
        _required("run_id", pa.string()),
        _required("window_start_step", pa.int64()),
        _required("map_group", pa.int16()),
        _required("map_num", pa.int16()),
        _required("x", pa.int16()),
        _required("y", pa.int16()),
        _required("visits", pa.int32()),
    ]
)

EPISODES = pa.schema(
    [
        _required("run_id", pa.string()),
        _required("seed", pa.int32()),
        _required("env_idx", pa.int32()),
        _required("episode_id", pa.int32()),
        _required("env_steps", pa.int64()),
        _required("wall_clock_s", pa.float64()),
        _required("episode_return", pa.float64()),
        _required("episode_length", pa.int32()),
        *_REWARDS,
        _required("badge_count", pa.int16()),
        _required("party_level_sum", pa.int16()),
        _required("story_flags", pa.int16()),
        _required("unique_tiles", pa.int32()),
        _required("milestone_count", pa.int16()),
        _required("start_state", pa.string()),
        _required("archive_start", pa.bool_()),
        _required("end_reason", pa.string()),
        # Null for RL runs.
        pa.field("llm_model", pa.string()),
        pa.field("llm_tool_level", pa.int16()),
        pa.field("llm_tokens_in", pa.int64()),
        pa.field("llm_tokens_out", pa.int64()),
        pa.field("llm_cost_usd", pa.float64()),
    ]
)

RUNS = pa.schema(
    [
        _required("run_id", pa.string()),
        _required("experiment_id", pa.string()),
        _required("arm", pa.string()),
        _required("algorithm", pa.string()),
        _required("trainer", pa.string()),
        _required("seed", pa.int32()),
        _required("env_id", pa.string()),
        _required("backend", pa.string()),
        _required("rom_sha1", pa.string()),
        _required("start_state", pa.string()),
        _required("obs_mode", pa.string()),
        _required("frames_per_decision", pa.int16()),
        _required("num_envs", pa.int16()),
        _required("reward_config_hash", pa.string()),
        _required("git_commit", pa.string()),
        _required("hyperparameters", pa.string()),
        _required("env_steps", pa.int64()),
        _required("wall_clock_s", pa.float64()),
    ]
)

TABLES = {"steps": STEPS, "visits": VISITS, "episodes": EPISODES, "runs": RUNS}


class SchemaError(ValueError):
    pass


def validate(batch: pa.Table, table: str) -> None:
    """Raise SchemaError unless `batch` has exactly the columns of `table`, with their types,
    and no null in a required column. Column order is free."""
    schema = TABLES[table]
    missing = sorted(set(schema.names) - set(batch.column_names))
    extra = sorted(set(batch.column_names) - set(schema.names))
    if missing or extra:
        raise SchemaError(f"{table}: missing {missing}, extra {extra}")
    for field in schema:
        column = batch.column(field.name)
        if column.type != field.type:
            raise SchemaError(f"{table}.{field.name}: {column.type}, expected {field.type}")
        if not field.nullable and column.null_count:
            raise SchemaError(
                f"{table}.{field.name}: {column.null_count} nulls in a required field"
            )
