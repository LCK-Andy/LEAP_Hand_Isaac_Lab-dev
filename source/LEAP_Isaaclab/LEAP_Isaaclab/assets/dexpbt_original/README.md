# Generated original DexPBT assets

Build this directory with `scripts/assets/build_dexpbt_original_assets.py` and an IsaacGymEnvs checkout pinned to `665c32170d84b4be66722eea405a1e08b6e7f761`.

See [`docs/dexpbt-original.md`](../../../../../docs/dexpbt-original.md) for preparation, provenance and validation limits. Generated USDs, imported meshes and manifests are intentionally ignored; the build script records their hashes and preserves the upstream license files. The task refuses to launch if the built robot or provenance manifest is missing.

Do not substitute the modern Dexsuite KUKA–Allegro asset: its dynamics and collision geometry are different from the released DexPBT robot.
