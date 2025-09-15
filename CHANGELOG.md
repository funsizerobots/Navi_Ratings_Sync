# Changelog

All notable changes to this project will be documented in this file.  
This project adheres to [Semantic Versioning](https://semver.org/).

---

## [2.1.0] - 2025-09-13
### Fixed
- Resolved logging overwrite issue caused by conflicting handlers from bootstrap setup. Logging now correctly writes to file and console during sync runs.

### Changed
- Updated rating conversion logic to use percentage-based anchors instead of scaled 10-point values. This improves compatibility with players that use half-stars within a 5-star rating system.

---

## [2.0.1] - 2025-09-10
### Changed
- Factored in 10 point star ratings when converting from Navidrome's 5 star scale to avoid half-star ratings when music files loaded in other music programs

---

## [2.0.0] - 2025-08-23
### Added
- Updated summary output to reflect new counter structure and sync outcome  
- Enhanced error handling for config and logging setup  
- Implemented a config-driven execution flow with centralized validation, logging, and branching logic  
- A configuration file (navi_ratings_sync.ini) to store values and set runtime flags and optional settings  
- ~~Pre-validation for required CLI arguments with clear error messages shown in both console and log~~  
- ~~Startup logging of all CLI arguments for reproducibility and debugging support~~  
- Logging for when changes are made to the database data  
- Logs now distinguish between real and dry-run rating updates for FLAC and MP3 files  
- Support for custom POPM tags in MP3 files via config setting  

### Changed
- Dry-run mode now correctly identifies malformed tags and prevents unintended sync actions
- Added safeguard for negative rating values in FLAC and MP3 files; these are now logged and skipped
- Fixed discrepancy between legacy FLAC ratings and new scale conversion logic
- Merged dry-run and full sync logic into a unified function  
- Improved dry-run behavior with consistent prefixing and clearer messaging  
- Improved error handling and logging throughout  
- Validation now runs before any sync, but can be run separately by setting config flag  
- When updating MP3 POPM ratings tags, use <user>@navidrome instead of just navidrome for POPM email  
- Refactored FLAC and MP3 rating conversion to use a single formula-based function  
- Updated `mp3_get_rating` and `mp3_write_rating` to use centralized conversion logic  
- Removed semi-hardcoded FLAC rating thresholds in favor of formulaic rounding (`round(flac_rating / 20)`), with clamping to valid range  

### Removed
- CLI argument to show version information  
- CLI argument parsing  
- Startup logging of CLI arguments (no longer applicable)  
- Pre-validation of CLI arguments (replaced by config validation)  
- CLI argument logging and validation logic

---

## [1.1.1] - 2025-08-18
### Added  
- CLI argument to show version information
- Script versioning via `__version__` and header comment updates  
- Split validation into modular functions for readability and reuse
- Improved validation logic with explicit schema checks    

### Changed   
- Updated header comment information and formatting for consistency
- Cleaned up argument help descriptions and defaults

---

## [1.1.0] - 2025-08-17  
### Added  
- First commit and push to private Gitea instance
- `.gitignore` rules for cache, logs, and tooling artifacts
- Initial Visual Studio project setup
- CLI argument parsing with support for dry-run and validation modes
- Added dry-run and validation mode logic
- MP3 file support
- Bidirectional rating sync: music file → database or database → music file

---

## [1.0.0]  
### Added  
- Basic logging to a log file
- Transfer ratings from Navidrome database to music files
