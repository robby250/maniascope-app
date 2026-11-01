# Releasing

Everything lands in `dist/` (gitignored). Nothing here uploads.

1. Linux: `cargo build --release -p ms-app`, copy `target/release/maniascope` to `dist/maniascope-linux-x86_64`.
2. Windows: `cargo build --release -p ms-app --target x86_64-pc-windows-gnu` (needs `gcc-mingw-w64-x86-64`), copy the exe
   to `dist/ManiaScope.exe`. CI does the same: `packaging/windows-workflow.yml`, exported as `.github/workflows/windows.yml`.
3. Population data: `cargo run --release -p ms-tools --bin release_public -- <public.msp> dist` writes
   `dist/public-<calc>.msp` without the builder's own scores.
4. Public source: `cargo run --release -p ms-tools --bin export_public -- <a clone of the public repo>`.

Attach the two exes and `public-<calc>.msp` to a release of `robby250/maniascope`. Every calculator change needs a new
`public-<calc>.msp`.
