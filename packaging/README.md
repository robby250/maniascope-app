# Releasing

1. Windows build without CI: `packaging/build_windows_wine.sh ../maniascope-app /tmp/winbuild`, then upload
   the zip to the release (this is how v0.1.0 was built). With CI (public repo; `packaging/windows-workflow.yml` is exported as `.github/workflows/windows.yml`):
   push a `v*` tag or run the `windows` workflow. The artifact is
   `ManiaScope-windows.zip`; the job runs `--self-test` both from source and frozen.
2. Population evidence for friends (derived statistics only, owner's scores stripped):
   `python3 -c "import recdata,pickle; p=recdata.load_public(); pickle.dump(recdata.release_public(p), open(f'public-{p[\"calc\"]}.pkl','wb'))"`
   Attach `public-<calc>.pkl` and the zip to a GitHub release of `recdata.RELEASE_REPO`.
3. Public source: `python3 packaging/export_public.py ../maniascope-app`, commit, push.

Every calculator change needs a new `public-<calc>.pkl` (and `calib/dan_table.py` rerun).
