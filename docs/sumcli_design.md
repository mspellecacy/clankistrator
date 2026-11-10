# sumcli: add numbers from the command line

A deliberately tiny design, useful as a cheap first run of Clankistrator.

## Vision
A small Rust command-line program. `sumcli 1 2 3.5` prints `6.5`.

## Requirements
- Rust, edition 2021, no dependencies.
- A library function `pub fn sum(args: &[String]) -> Result<f64, String>` in `src/lib.rs` that parses each argument as `f64` and returns the total, or an error naming the first argument that is not a number.
- `src/main.rs` calls it with the command-line arguments, prints the total, and exits with status 1 and a message on stderr on error.
- Unit tests in `src/lib.rs` for: empty input (the total is 0), several numbers, and an invalid argument.

## Roadmap
- [ ] Create the Cargo project with the library and the binary.
- [ ] Implement `sum` with its tests.
- [ ] Wire up `main` and check `cargo test` and `cargo build` pass.
