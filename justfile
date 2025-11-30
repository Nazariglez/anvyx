run target:
    cargo run -- run {{target}}

check target:
    cargo run -- check {{target}}

tests target="tests":
    cargo test --workspace
    cargo run --package test-runner -- {{target}} --quiet

tests-release target="tests":
    cargo test --workspace --release
    cargo run --package test-runner -- {{target}} --release