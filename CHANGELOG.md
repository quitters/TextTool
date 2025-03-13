# Changelog

## [1.1.0] - 2025-03-13

### Fixed
- Increased text size in the renderer to make text much more visible (default font size increased from 24 to 72)
- Fixed the export functionality by adding the missing `show_export_options` method
- Fixed indentation issues in the main window code
- Restored the `run_application` function that was accidentally removed

### Changed
- Improved text rendering quality by increasing the minimum scale factor from 1 to 2
- Added better error handling for text rendering and export operations

### Added
- Added debug print statements to help diagnose text rendering issues
- Created utility scripts for fixing and maintaining the codebase
