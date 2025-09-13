# 🎵 Navidrome Ratings Sync Script

This Python script syncs music ratings between a Navidrome database and local FLAC or MP3 files

## 🚀 Features

- A Validation mode to check that the database exists, can be accessed, and has the correct tables and columns in place to pull the data from. It also verifies that music files exist and are writable  
- A Dry-run mode to check what changes the script would make without actually making any changes
- Ratings are synchronized between the database and the music files, if the rating exists only in the database or only in the file, it will be written to the other
- When both database and music file contain ratings that do not match, a configurable preference determines whether to use the rating from the database or from the music file
- Logging  
- Explicit exit codes for CI or automation use

## 📦 Requirements

- Python 3.10+
- Mutagen library (used for FLAC/MP3 tag access)
- Navidrome instance

## 📁 Directory Placement

Place both the script and the navi_ratings_sync.ini config file in the same directory as the Navidrome database (navidrome.db).

## 🛠️ Usage

The script now uses a config file (navi_ratings_sync.ini) to define paths, user settings, sync behavior, and optional MP3 customization. CLI arguments are no longer required or supported.

```ini
[Paths]
music_dir = /path/to/music
db_path = /path/to/navidrome.db
log_path = ratings.log

[User]
user = navidrome_username

[SyncBehavior]
preference = database
validate_only = false
dry_run = true

[MP3 Customization]
custom_mp3_tag = CustomPOPM
custom_mp3_scale = 0-255
custom_mp3_email = user@example.com
```

## 📤 Exit Codes

These exit codes are designed for use in CI pipelines, wrapper scripts, or cron jobs to detect success, failure, or dry-run outcomes.

0: ✅ Success — Full sync completed without errors 
1: ❌ Failure — Validation failed, user not found, or critical error  
10: 🔍 Validation-only success — All checks passed, no sync performed  
20: 🧪 Dry-run success — Simulated sync completed without errors

## 🎚️ Rating Scale Expectations

This tool assumes the following default rating scales unless overridden in the config file:

- **FLAC files**: Ratings are expected to use a 0–100 scale (e.g., `RATING=80` for 4 stars).  
  If your FLAC files use a 1–5 or 1–10 scale (e.g., RATING=5 to mean 5 stars), they must be normalized to match this model before syncing. The script does not auto-detect alternate scales.

If you're unsure how your files are tagged, consider auditing them before syncing.  
This tool does not currently auto-detect rating scales.

## 🧪 Experimental MP3 Support

Support for MP3 files is still in the testing phase. While the script can read and write ratings using `POPM` frames and custom tags, behavior may vary depending on how your files were originally tagged. Feedback from users with MP3 libraries is especially welcome.

### 🔍 Default Behavior

- Ratings are read from the `POPM` frame, which typically uses a 0–255 scale.
- By default, the script maps MP3 ratings to a 1–5 scale using percentage anchors:
  1★ = 51, 2★ = 102, 3★ = 153, 4★ = 204, 5★ = 255.
  This ensures consistent full-star behavior across players that interpret ratings using half-star bins.
- If no custom tag is specified, the script uses a `POPM` frame associated with your Navidrome username.

### ⚙️ Custom Tag Support

If your MP3 files contain ratings from another program — like MusicBee, MediaMonkey, or a custom tool — you can configure the script to use a custom tag and scale. This allows you to migrate legacy ratings or maintain a separate tagging scheme.

```ini
[MP3 Customization]
custom_mp3_tag = my_rating
custom_mp3_scale = 0-255
custom_mp3_email = legacy@example.com
```

- **`custom_mp3_tag`**: The name of the tag to use (not limited to POPM).
- **`custom_mp3_scale`**: The rating scale used by your custom tag (e.g., 0–255).
- **`custom_mp3_email`**: The identifier stored in the tag — usually an email-style string.  
  This is optional and only required for POPM frames that use email identifiers.

>⚠️ `custom_mp3_tag` and `custom_mp3_scale` must be set together. If either is missing, the script will halt with an error to prevent misconfiguration.

>⚠️ Note: Not all music players use percentage-based rating scales. This script assumes ratings represent percentage thresholds (e.g., 80 = 4 stars on a 0–100 scale).

### ✅ Common Use Cases

- Migrating ratings from another music app into Navidrome  
- Testing sync behavior without affecting your main tags  
- Using separate rating tags for different users or setups

### ⚠️ Known Limitations

- Multiple `POPM` frames with different identifiers may cause conflicts.
- Non-standard ID3 versions or corrupted tags may not be handled gracefully.
- The script does not auto-detect rating scales — manual normalization may be required.

## ⚠️ Common Issues

- Duplicate albums in Navidrome
If you see duplicate albums after syncing, run a full rescan in Navidrome. This rebuilds its internal index and resolves inconsistencies caused by tag changes or file moves.

## 📦 Latest Release
<!-- VERSION_BLOCK_START -->
Current version: v2.1.0
<!-- VERSION_BLOCK_END -->
Get it at https://github.com/funsizerobots/Navi_Ratings_Sync/releases

## 📝 Notes
These notes document the background, design choices, and known behaviors of the script

- This script has been my first experience with Python.
- I originally wrote it to write the song ratings to the song files from the Navidrome database. Then realized it might be a good idea to keep them in sync and go both ways. The need to resolve the conflict if the ratings differed between song file and database is where the preference option came from.
- I did attempt to use the Navidrome API at first, instead of directly accessing the database, however, it was not giving me proper information back regarding song locations and so I decided it was just easier at that point to go with accessing the database directly.
- I wrote the original Python script myself and it was functional, but later I used Microsoft Copilot (because it was there and easy to access) to help me add the things I wanted to add to make the script more robust and safer. This includes the validation and dry-run modes and many other changes.
- I tried to make the logic as least destructive as I could. It only adds information to either the tag or the database when it's missing from one and it only makes a change to the tag or database when there is a rating in each that doesn't match (set by preference). Neither the rating tags in the files or the ratings in the database are deleted/removed in this process.
- If a music file contains a rating tag with a negative value (e.g., RATING=-20), it will be skipped entirely. No changes will be made to the file or the database. These files are logged as warnings and counted in the summary under “Malformed tags.” This applies to both FLAC and MP3 formats.
- Files with no rating in either the database or the music file are counted as “zero ratings” and skipped. This helps distinguish between missing data and intentional unrated tracks.
- In dry-run mode, all sync actions are simulated but not applied. Malformed tags and other warnings are still logged, allowing safe diagnostics before committing changes.
- All log messages are written to the path specified in log_path. This includes validation results, sync actions, skipped files, and any detected conflicts. Exit codes are designed for use in CI pipelines, wrapper scripts, or cron jobs.