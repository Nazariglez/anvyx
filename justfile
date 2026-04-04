run target:
    cargo run -- run {{target}}

check target:
    cargo run -- check {{target}}

tests target="tests":
    cargo run --package test-runner -- {{target}} --quiet

full-tests target="tests":
    cargo test --workspace
    cargo run --package test-runner -- {{target}} --quiet

full-tests-release target="tests":
    cargo test --workspace --release
    cargo run --package test-runner -- {{target}} --release

install:
    cargo install --path crates/anvyx --force

miri:
    MIRIFLAGS="-Zmiri-strict-provenance" cargo +nightly miri test -p anvyx-lang --all-targets

scan-tests threshold="75":
    python3 ./scripts/scan_tests.py -t {{threshold}}

fmt:
    cargo +nightly fmt