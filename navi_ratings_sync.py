#!/usr/bin/env python3

"""
navi_ratings_sync.py

Synchronizes ratings between local FLAC/MP3 files and a Navidrome SQLite database.

Version: 2.1.0
Author: Kevin Rogers
Last Updated: 2025-09-13

Usage:
    python3 navi_ratings_sync.py \
        --db-path   /path/to/navidrome.db \
        --music-dir /path/to/music \        
        --user      navidrome_username \
        [--preference file|database] \
        [--validate-only] \
        [--dry-run] \
        [--log-path ratings.log] \
        [--version]

Supports validation-only mode, dry-run simulation, and bidirectional sync based on user preference (defaults to 'database' if not specified).

Example usage: 
    python3 navi_ratings_sync.py --db-path /path/to/navidrome.db --music-dir /path/to/music --user navidrome_username [--preference file|database]
"""

# =============================================================================
# 1. Imports & Globals
# =============================================================================
# Standard libraries, third-party modules, and global constants like __version__

import os
import sys
import configparser
import logging
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field
from mutagen.flac import FLAC
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, POPM
from collections import Counter
from typing import Optional
from typing import ClassVar

__version__ = "2.1.0"

# =============================================================================
# 2. Runtime Context & Constants
# =============================================================================
# Defines the SyncContext dataclass for encapsulating runtime settings,
# along with default rating scales used for FLAC and MP3 normalization.
# These constants support unified conversion logic across file formats.

@dataclass
class SyncContext:
    DEFAULT_FLAC_SCALE: ClassVar[str] = "0-100"
    DEFAULT_MP3_SCALE: ClassVar[str] = "0-255"

    music_dir: Path
    db_path: Path
    log_path: Path
    user: str
    preference: str
    dry_run: bool
    validate_only: bool
    custom_mp3_tag: str = ""
    custom_mp3_scale: str = ""
    custom_mp3_email: str = ""
    default_mp3_tag: str = ""

    def __post_init__(self):
        if not self.default_mp3_tag:
            self.default_mp3_tag = f"{self.user}@navidrome"

    @property
    def flac_scale(self) -> tuple[int, int]:
        """Returns the FLAC rating scale as a (low, high) tuple, e.g., (0, 100)."""
        return tuple(map(int, self.DEFAULT_FLAC_SCALE.split("-")))

    @property
    def mp3_scale(self) -> tuple[int, int]:
        """Returns the MP3 rating scale as a (low, high) tuple, using custom or default."""
        scale_str = self.custom_mp3_scale or self.DEFAULT_MP3_SCALE
        return tuple(map(int, scale_str.split("-")))

    @property
    def mp3_tag(self) -> str:
        """Returns the tag name to use for MP3 rating sync (POPM frame)."""
        return self.custom_mp3_tag or self.default_mp3_tag

    @property
    def mp3_email(self) -> Optional[str]:
        """
        Returns the POPM email identifier to use when reading/writing MP3 ratings.
        If customization is active but email is blank, returns None to skip POPM.
        """
        if self.custom_mp3_tag and self.custom_mp3_scale:
            email = self.custom_mp3_email.strip()
            return email if email else None
        return self.default_mp3_email

    stats: Counter = field(default_factory=Counter)


# =============================================================================
# 3. Config File Loading & Validation
# =============================================================================
# Loads navi_ratings_sync.ini, validates required fields, and logs config summary

config_path = Path(__file__).parent / "navi_ratings_sync.ini"

REQUIRED_FIELDS = {
    "Paths": ["music_dir", "db_path", "log_path"],
    "User": ["user"],
    "SyncBehavior": ["preference", "validate_only", "dry_run"]
}

def load_config(path: Path) -> dict:
    """Loads config file and returns parsed dictionary."""

    logging.info("Loading config file: %s", path)
    if not path.exists():
        logging.error("Config file not found: %s", path)
        raise FileNotFoundError(f"Missing config file: {path}")

    parser = configparser.ConfigParser()
    parser.read(path)

    # Convert to nested dict
    cfg = {section: dict(parser.items(section)) for section in parser.sections()}
    logging.debug("Config loaded with sections: %s", list(cfg.keys()))
    return cfg

def validate_config(cfg: dict) -> None:
    """Validates presence and basic types of required config fields."""
    logging.info("Validating config structure and values")

    # Check required sections and keys
    for section, keys in REQUIRED_FIELDS.items():
        if section not in cfg:
            logging.error("Missing required section: [%s]", section)
            raise ValueError(f"Missing required section: [{section}]")
        for key in keys:
            if key not in cfg[section]:
                logging.error("Missing required key: [%s][%s]", section, key)
                raise ValueError(f"Missing required key: [{section}][{key}]")
            if cfg[section][key] in [None, ""]:
                logging.error("Empty value for key: [%s][%s]", section, key)
                raise ValueError(f"Empty value for key: [{section}][{key}]")

    # Type conversions
    try:
        cfg["SyncBehavior"]["validate_only"] = cfg["SyncBehavior"]["validate_only"].lower() == "true"
        cfg["SyncBehavior"]["dry_run"] = cfg["SyncBehavior"]["dry_run"].lower() == "true"
    except Exception as e:
        logging.exception("Failed to parse boolean values in SyncBehavior")
        raise TypeError("SyncBehavior booleans must be 'true' or 'false'") from e

    # Validate preference
    valid_preferences = {"database", "file"}
    pref = cfg["SyncBehavior"]["preference"].lower()
    if pref not in valid_preferences:
        logging.error("Invalid SyncBehavior.preference: %s", pref)
        raise ValueError(f"SyncBehavior.preference must be one of: {valid_preferences}")

    # Validate MP3 customization block
    tag = cfg.get("MP3 Customization", {}).get("custom_mp3_tag", "").strip()
    scale = cfg.get("MP3 Customization", {}).get("custom_mp3_scale", "").strip()

    if (tag and not scale) or (scale and not tag):
        logging.error("Both custom_mp3_tag and custom_mp3_scale must be set together")
        raise ValueError("Incomplete MP3 customization: both tag and scale are required")

def log_config_summary(cfg: dict) -> None:
    logging.info("Config Summary:")
    for section, values in cfg.items():
        for key, value in values.items():
            logging.info("  [%s] %s = %s", section, key, value)

# =============================================================================
# 4. Logging Setup
# =============================================================================
# Initializes temporary logging (bootstrap) and full logging after config load

def bootstrap_logging() -> None:
    """Temporary logging setup used before config is loaded and validated."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logging.info("Temporary logging initialized")

def setup_logging(log_path: Path, dry_run: bool):
    """
    Configure logging to file and console with timestamps and log levels.
    Ensures log file exists and is writable.
    """

    # Resolve absolute path
    log_path = log_path.resolve()

    # Create log file if it doesn't exist
    if not log_path.exists():
        try:
            log_path.touch()
        except Exception as e:
            raise PermissionError(f"Cannot create log file: {log_path}") from e

    # Check write access
    if not os.access(log_path, os.W_OK):
        raise PermissionError(f"Log file is not writable: {log_path}")

    # Clear any existing handlers to avoid conflicts with bootstrap logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='w'),  # Overwrite each run
            logging.StreamHandler()
        ]
    )

    logging.info("Logging initialized at: %s", log_path)
    if dry_run:
        logging.info("DRY-RUN mode enabled: no changes will be written")

# =============================================================================
# 5. Environment Validation
# =============================================================================
# Checks DB schema, song entries, and file accessibility before sync

def run_validation(ctx) -> bool:
    """
    Performs environment checks to ensure sync can proceed safely.
    This always runs before dry-run or full sync.
    """
    logging.info("Starting validation")

    # 1. DB file exists
    if not ctx.db_path.exists():
        logging.error("Database file not found: %s", ctx.db_path)
        return False

    # 2. Schema check
    if not validate_db_schema(ctx.db_path):
        return False

    # 3. Songs exist
    songs = get_all_songs(ctx.db_path)
    if not songs:
        logging.error("No songs found in media_file table")
        return False

    # 4. File checks
    validate_song_files(songs, ctx.music_dir)

    logging.info("Validation summary: DB schema intact, %d songs checked, file checks completed.", len(songs))
    return True

def validate_db_schema(db_path: Path) -> bool:
    """
    Validates required tables and their expected columns.
    Returns True if all checks pass.
    """
    schema = {
        "user": ["id", "user_name"],
        "media_file": ["id", "path"],
        "annotation": [
            "user_id", "item_id", "item_type", "rating",
            "play_count", "play_date", "starred", "starred_at"
        ]
    }

    for table, required_cols in schema.items():
        # Check table exists
        rows = db_query(db_path, f"""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='{table}'
        """)
        if not rows:
            logging.error("Missing required table: %s", table)
            return False

        # Check required columns
        col_rows = db_query(db_path, f"PRAGMA table_info({table})")
        existing_cols = {row[1] for row in col_rows}
        missing = [col for col in required_cols if col not in existing_cols]
        if missing:
            logging.error("Table '%s' missing columns: %s", table, ", ".join(missing))
            return False

    logging.info("Database schema validation passed")
    return True

def validate_song_files(songs: list[tuple[int, str]], music_dir: Path) -> None:
    """
    Validates existence, type, and write access of song files.
    Logs warnings but does not fail validation.
    """
    for song_id, raw_path in songs:
        try:
            path = resolve_song_path(raw_path, music_dir)

            if not path.exists():
                logging.warning("Missing file: %s", path)
                continue

            if path.suffix.lower() not in [".flac", ".mp3"]:
                logging.info("Unsupported file type: %s", path)
                continue

            if not os.access(path, os.W_OK):
                logging.warning("No write permission for file: %s", path)

        except Exception:
            logging.exception("Validation error for song_id %s (%s)", song_id, raw_path)

# =============================================================================
# 6. Sync Execution
# =============================================================================
# Main sync loop and summary reporting, with dry-run support

def run_sync(ctx: SyncContext) -> bool:
    """
    Executes sync operation. Behavior depends on ctx.dry_run.
    Returns True if operation completes successfully, False otherwise.
    """
    prefix = "[DRY-RUN] " if ctx.dry_run else ""

    logging.info("%sStarting sync operation", prefix)

    user_id = get_user_id(ctx.db_path, ctx.user)
    if user_id is None:
        logging.error("%sUser ID lookup failed — aborting", prefix)
        return False

    songs = get_all_songs(ctx.db_path)
    if not songs:
        logging.error("%sNo songs found in media_file table", prefix)
        return False

    logging.info("%sFound %d songs to process", prefix, len(songs))
    ctx.stats = Counter()
    ctx.stats['total_songs'] = len(songs)

    for song_id, raw_path in songs:
        try:
            path = resolve_song_path(raw_path, ctx.music_dir)

            if not path.exists():
                logging.warning("%sSkipping missing file: %s", prefix, path)
                ctx.stats['missing_files'] += 1
                continue

            if path.suffix.lower() not in [".flac", ".mp3"]:
                logging.info("%sUnsupported file type: %s", prefix, path)
                ctx.stats['unsupported_file_types'] += 1
                continue

            process_song(
                path, song_id, user_id, ctx
            )

        except Exception:
            logging.exception("%sError processing song_id %s (%s)", prefix, song_id, raw_path)
            ctx.stats['malformed_tags'] += 1

    summarize_sync(ctx)
    logging.info("%sSync operation complete", prefix)
    return True

def summarize_sync(ctx: SyncContext):
    prefix = "[DRY-RUN] " if ctx.dry_run else ""
    stats = ctx.stats

    lines = [
        f"{prefix}Summary:",
        f"{prefix}  Total songs scanned: {stats['total_songs']}",
        f"{prefix}  Matched ratings: {stats['matched_ratings']}",
        f"{prefix}  Zero ratings: {stats['zero_ratings']}",
        f"{prefix}  Conversions: {stats['conversions']}",
        f"{prefix}  Missing files: {stats['missing_files']}",
        f"{prefix}  Unsupported file types: {stats['unsupported_file_types']}",
        f"{prefix}  Malformed tags: {stats['malformed_tags']}",
    ]
    for line in lines:
        logging.info(line)

# =============================================================================
# 7. SQLite Access Helpers
# =============================================================================
# Safe query/commit functions and lookup utilities for users, songs, and ratings

def db_query(db_path: Path, query: str, params=()):
    """
    Safe, performant SELECT using context manager.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(query, params)
            return cur.fetchall()
    except sqlite3.Error as e:
        logging.error("DB query failed [%s]: %s", query.strip().split()[0], e)
        return []

def db_commit(db_path: Path, query: str, params=(), dry_run=False, context=""):
    """
    Safe INSERT/UPDATE with auto-commit. Returns rows affected.
    Logs dry-run and actual writes with context.
    """
    action = query.strip().split()[0].upper()
    if dry_run:
        logging.info("[DRY-RUN] Would %s → %s: %s", action, context, params)
        return 0
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(query, params)
            conn.commit()
            logging.info("%s → %s: %s", action, context, params)
            return cur.rowcount
    except sqlite3.Error as e:
        logging.error("DB %s failed on %s: %s", action, db_path, e)
        return 0

def get_user_id(db_path: Path, username: str):
    """
    Fetch user ID for given username. Returns None if not found.
    """
    rows = db_query(
        db_path,
        "SELECT id FROM user WHERE user_name = ?",
        (username,)
    )
    if rows:
        return rows[0][0]
    logging.error("User '%s' not found in DB: %s", username, db_path)
    return None

def get_all_songs(db_path: Path):
    """
    Retrieve all media_file entries: returns list of (id, path) tuples.
    """
    return db_query(db_path, "SELECT id, path FROM media_file")

def get_nav_rating(db_path: Path, user_id: int, song_id: int):
    """
    Fetch rating from annotation table (1-5). Returns 0 if none.
    """
    rows = db_query(
        db_path,
        "SELECT rating FROM annotation "
        "WHERE user_id = ? AND item_id = ? AND item_type = 'media_file'",
        (user_id, song_id)
    )
    return int(rows[0][0]) if rows else 0

def create_annotation(db_path: Path, user_id: int, song_id: int, rating: int, dry_run=False):
    context = f"annotation[user={user_id}, song={song_id}]"
    return db_commit(
        db_path,
        "INSERT INTO annotation (user_id, item_id, item_type, play_count, play_date, rating, starred, starred_at) "
        "VALUES (?, ?, 'media_file', 0, NULL, ?, 0, NULL)",
        (user_id, song_id, rating),
        dry_run=dry_run,
        context=context
    )

def update_annotation(db_path: Path, user_id: int, song_id: int, rating: int, dry_run=False):
    context = f"annotation[user={user_id}, song={song_id}]"
    return db_commit(
        db_path,
        "UPDATE annotation SET rating = ? WHERE user_id = ? AND item_id = ? AND item_type = 'media_file'",
        (rating, user_id, song_id),
        dry_run=dry_run,
        context=context
    )
    
def annotation_exists(db_path: Path, user_id: int, song_id: int) -> bool:
    rows = db_query(
        db_path,
        "SELECT 1 FROM annotation WHERE user_id = ? AND item_id = ? AND item_type = 'media_file'",
        (user_id, song_id)
    )
    return bool(rows)

# =============================================================================
# 8. Rating Conversion Utilities
# =============================================================================
# Converts ratings between Navidrome, FLAC, and MP3 formats using percentage-based anchors

def convert_to_navidrome(value: int, scale: tuple[int, int], round_mode="round") -> int:
    """
    Converts a rating from a custom scale to Navidrome's 1-5 rating format.

    Parameters:
        value (int): The input rating from the target format (e.g., 0-100 or 0-255).
        scale (tuple[int, int]): The source rating scale as a (low, high) tuple.
                                 Examples: (0, 100) for FLAC, (0, 255) for MP3.
        round_mode (str): Rounding strategy to apply. Options:
                          - 'round' (default): standard rounding
                          - 'floor': always round down
                          - 'ceil': always round up

    Returns:
        int: The converted Navidrome rating (range: 1–5), clamped to valid bounds.
             Uses percentage anchors to ensure full-star fidelity across formats.
    """
    import math
    low, high = scale

    if high == 5 and low == 1:
        return value  # already in Navidrome format

    # Convert value to percentage of scale
    percent = (value - low) / (high - low)

    # Map percentage to Navidrome rating (1–5)
    scaled = percent * 5

    if round_mode == "floor":
        return max(1, min(5, math.floor(scaled)))
    elif round_mode == "ceil":
        return max(1, min(5, math.ceil(scaled)))
    return max(1, min(5, round(scaled)))

def convert_from_navidrome(rating: int, scale: tuple[int, int], round_mode="round") -> int:
    """
    Converts a Navidrome rating (1–5) to a target format-specific rating scale.

    Parameters:
        rating (int): The Navidrome rating to convert (expected range: 1–5).
        scale (tuple[int, int]): The target rating scale as a (low, high) tuple.
                                 Examples: (0, 100) for FLAC, (0, 255) for MP3.
        round_mode (str): Rounding strategy to apply. Options:
                          - 'round' (default): standard rounding
                          - 'floor': always round down
                          - 'ceil': always round up

    Returns:
        int: The converted rating value within the target scale bounds.
             Anchors each star rating to a percentage of the scale:
             1★ = 20%, 2★ = 40%, ..., 5★ = 100%
    """
    import math
    low, high = scale

    # Define percentage anchors for full-star ratings
    percent = {1: 0.20, 2: 0.40, 3: 0.60, 4: 0.80, 5: 1.00}.get(rating, 0.0)

    # Scale percentage to target range
    scaled = percent * (high - low) + low

    if round_mode == "floor":
        return max(low, min(high, math.floor(scaled)))
    elif round_mode == "ceil":
        return max(low, min(high, math.ceil(scaled)))
    return max(low, min(high, round(scaled)))

# =============================================================================
# 9. File I/O: FLAC & MP3
# =============================================================================
# Reads and writes ratings to audio files, with error handling and dry-run logic

def flac_get_rating(path: Path, ctx: SyncContext) -> int:
    try:
        audio = FLAC(path)
        raw = audio.get("RATING", [])
        if raw:
            rating = int(raw[0])
            if rating < 0:
                logging.warning("Improper FLAC rating tag (negative value: %d) in file: %s", rating, path)
                ctx.stats['malformed_tags'] += 1
                return -1
            return convert_to_navidrome(rating, ctx.flac_scale)
    except Exception as e:
        logging.error("FLAC read error %s: %s", path, e)
    return 0

def flac_write_rating(path: Path, rating: int, ctx: SyncContext):    
    desc = f"FLAC rating {rating} → {path}"
    if ctx.dry_run:
        logging.info("[DRY-RUN] Would write " + desc)
        return
    try:
        audio = FLAC(path)
        audio["RATING"] = str(convert_from_navidrome(rating, ctx.flac_scale))
        audio.save()
        logging.info("Wrote " + desc)
    except Exception as e:
        logging.error("FLAC write error %s: %s", path, e)

def mp3_get_rating(path: Path, ctx: SyncContext) -> int:
    try:
        audio = MP3(path, ID3=ID3)
        popms = audio.tags.getall("POPM")
        if not popms:
            return 0

        # Choose correct POPM frame
        target_email = ctx.mp3_email
        for frame in popms:
            if frame.email == target_email:
                if frame.rating < 0:
                    logging.warning("Improper MP3 rating tag (negative value: %d) in file: %s", frame.rating, path)
                    ctx.stats['malformed_tags'] += 1
                    return -1
                scale = ctx.mp3_scale
                return convert_to_navidrome(frame.rating, scale)
    except Exception as e:
        logging.error("MP3 read error %s: %s", path, e)
    return 0

def mp3_write_rating(path: Path, rating: int, ctx: SyncContext):
    desc = f"MP3  rating {rating} → {path}"
    if ctx.dry_run:
        logging.info("[DRY-RUN] Would write " + desc)
        return
    try:
        audio = MP3(path, ID3=ID3)
        target_email = ctx.mp3_tag
        popm = POPM(email=target_email, rating=convert_from_navidrome(rating, ctx.mp3_scale))

        # Remove only the POPM frame with matching email
        popms = audio.tags.getall("POPM")
        for frame in popms:
            if frame.email == target_email:
                audio.tags.delall("POPM:" + frame.email)

        audio.tags.add(popm)
        audio.save()
        logging.info("Wrote " + desc)
    except Exception as e:
        logging.error("MP3 write error %s: %s", path, e)

# =============================================================================
# 10. Sync Decision Logic
# =============================================================================
# Determines sync direction based on rating values and user preference

def decide_action(
        nav_rating: int,
        file_rating: int,
        preference: str
    ):
    """
    Determine whether to write to file, to DB, or skip.
    Returns tuple: (action, rating)
    - action: "write_file", "write_db", or None
    - rating: the rating value to apply
    """
    # If only one rating exists, sync it to the other
    if nav_rating and not file_rating:
        return ("write_file", nav_rating)
    if file_rating and not nav_rating:
        return ("write_db", file_rating)

    # If both exist and differ, use preference
    if nav_rating and file_rating and nav_rating != file_rating:
        if preference == "database":
            return ("write_file", nav_rating)
        elif preference == "file":
            return ("write_db", file_rating)

    return (None, None)

def resolve_song_path(raw_path: str, music_dir: Path) -> Path:
    """
    Safely resolve a song's full filesystem path based on music_dir.
    Handles leading slashes and ensures absolute resolution.
    """
    return (music_dir / raw_path.lstrip("/")).resolve()

def process_song(path: Path, song_id: int, user_id: int, ctx: SyncContext):
    """
    Compare ratings and simulate or apply sync action.
    Returns tuple: (conflict_found: bool, malformed_tag: bool)
    """
    ext = path.suffix.lower()
    nav = get_nav_rating(ctx.db_path, user_id, song_id)
    file = 0
    malformed = False

    if ext == ".flac":
        file = flac_get_rating(path, ctx)
        if file == -1:
            malformed = True
            return (False, malformed)
    elif ext == ".mp3":
        file = mp3_get_rating(path, ctx)
        if file == -1:
            malformed = True
            return (False, malformed)

    if nav == 0 and file == 0:
        ctx.stats['zero_ratings'] += 1
    elif nav == file:
        ctx.stats['matched_ratings'] += 1
    else:
        ctx.stats['conversions'] += 1

    conflict = nav and file and nav != file
    if conflict:
        logging.info("Rating conflict: DB=%d, File=%d → %s", nav, file, path)

    action, rating = decide_action(nav, file, ctx.preference)

    if action == "write_file":
        if ctx.dry_run:
            logging.info("[DRY-RUN] Would write rating %d to file: %s", rating, path)
        else:
            if ext == ".flac":
                flac_write_rating(path, rating, ctx)
            else:
                mp3_write_rating(path, rating, ctx)

    elif action == "write_db":
        if ctx.dry_run:
            logging.info("[DRY-RUN] Would update DB rating to %d for file: %s", rating, path)
        else:
            if annotation_exists(ctx.db_path, user_id, song_id):
                update_annotation(ctx.db_path, user_id, song_id, file)
            else:
                create_annotation(ctx.db_path, user_id, song_id, file)

    return (conflict, malformed)

# =============================================================================
# 11. Main Execution Flow
# =============================================================================
# Orchestrates config loading, validation, and sync execution

def main():
    # Step 1: Setup temporary logging to catch early issues
    bootstrap_logging()

    # Step 2: Load config file
    try:
        cfg = load_config(config_path)
    except Exception as e:
        logging.error("Failed to load config: %s", e)
        return 1

    # Step 3: Validate config structure and values
    try:
        validate_config(cfg)
    except Exception as e:
        logging.error("Config validation failed: %s", e)
        return 1

    # Step 4: Extract config values into SyncContext
    ctx = SyncContext(
        music_dir=Path(cfg["Paths"]["music_dir"]),
        db_path=Path(cfg["Paths"]["db_path"]),
        log_path=Path(cfg["Paths"]["log_path"]),
        user=cfg["User"]["user"],
        preference=cfg["SyncBehavior"]["preference"],
        dry_run=cfg["SyncBehavior"]["dry_run"],
        validate_only=cfg["SyncBehavior"]["validate_only"],
        custom_mp3_tag=cfg.get("MP3 Customization", {}).get("custom_mp3_tag", "").strip(),
        custom_mp3_scale=cfg.get("MP3 Customization", {}).get("custom_mp3_scale", "").strip(),
        custom_mp3_email=cfg.get("MP3 Customization", {}).get("custom_mp3_email", "").strip()
    )

    # Step 5: Setup full logging now that we have log_path
    try:
        setup_logging(ctx.log_path, ctx.dry_run)
    except Exception as e:
        logging.error("Failed to initialize logging: %s", e)
        return 1

    # Step 6: Run environment validation
    success = run_validation(ctx)
    if not success:
        logging.error("Validation failed — aborting")
        return 1

    logging.info("Environment validated — ready to proceed with sync logic")

    # Step 7: Run Validation mode
    if ctx.validate_only:
        logging.info("Running in validate-only mode")
        success = run_validation(ctx)
        return 10 if success else 1

    # Step 8: Dry-run or Full Sync
    logging.info("Running %s", "dry-run sync" if ctx.dry_run else "full sync")
    success = run_sync(ctx)
    return (20 if ctx.dry_run else 0) if success else 1

if __name__ == "__main__":
    exit_code = main()
    logging.info("Exiting with code %d", exit_code)
    sys.exit(exit_code)