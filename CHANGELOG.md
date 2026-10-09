# Changelog

This file records user-visible and repository-governance changes.

## Unreleased

### Added

- Optional Gemini Robotics ER 2 orchestration above the RTC VLA client, with a
  closed skill catalog, latest-only camera observations, versioned prompt
  switching, and offline CPU tests.
- GitHub issue forms and a pull request template for public contributions.
- Software-only CPU continuous integration checks for repository readiness and
  the `openpi-client` image and msgpack utilities.
- Private security reporting guidance using the approved repository contact.
- Network deployment boundary documentation for the controlled robot LAN and
  current transport limitations.
- Reproducible external example dependencies for ALOHA and LIBERO, with pinned
  source checkouts that are not bundled in the repository snapshot.
- License and provenance documentation for the OpenPI working baseline,
  inherited components, local modifications, and Gemma model-asset boundary.
- English and Chinese user guides for the public Cloth task.
- YAML task configuration support for selecting the number of FSDP devices.
- ModelScope as an additional public mirror for TRON2 OpenPI model weights.

### Changed

- Standardized the English and Chinese README language navigation.
- Normalized public TRON2 task config names to start their task suffix with a
  lowercase letter.
