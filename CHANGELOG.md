# Changelog

This file records user-visible and repository-governance changes.

## Unreleased

### Added

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
- Modular TRON2 training and deployment for ServoJ/ServoP arms,
  gripper/BrainCo2 paired hands, head, chassis, and lifter layouts from 16D to
  31D.
- State/action dimension negotiation between task configs, policy serving, and
  deployment clients.
- `--physical-dim` support in the portable one-command training entrypoint.
- Public deployment profiles for the 26D ServoP/BrainCo2 layout and the 21D
  ServoJ/gripper/head/chassis/lifter layout.

### Changed

- Standardized the English and Chinese README language navigation.
- Normalized public TRON2 task config names to start their task suffix with a
  lowercase letter.
- Made RTC action postprocessing and recording layout-aware, including safe
  handling for ServoP quaternion actions and BrainCo2 hand slices.
- Updated the BrainCo and chassis server profiles with their deployment
  checkpoint paths.
- Both READMEs name the deploy profile pair for every public task, list all six
  in the deployment section instead of only Candy, and mark which profiles use
  hardware-based names.

### Fixed

- Stabilized quantile normalization for nearly constant state and action
  dimensions to prevent amplified training targets and gradient spikes.
- ModelScope links in both READMEs and in the model badge pointed at the retired
  `limx-tron2` namespace, which returns 404; they now use `limxdynamics`.
- `policy.repo_id` is documented as the asset directory that has to exist inside
  the downloaded checkpoint, and the config table is marked as the in-code
  defaults rather than the published layout.
- `INSTALL.md` and `INSTALL_CN.md` cloned from the retired `limx-tron2`
  organisation; both now point at `limxdynamics`, matching the repository's own
  URLs.
