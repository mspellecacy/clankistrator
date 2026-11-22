//! Tells the program which commit it was built from, for the usage log and
//! the transcripts: a run is easier to account for when it is known which
//! build ran it. A build outside a git checkout, or without git, goes without.

use std::process::Command;

fn main() {
    for watched in [".git/HEAD", ".git/index", ".git/refs/heads", "src"] {
        println!("cargo:rerun-if-changed={watched}");
    }
    let git = |args: &[&str]| {
        let output = Command::new("git").args(args).output().ok().filter(|output| output.status.success())?;
        Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
    };
    if let Some(commit) = git(&["rev-parse", "--short=12", "HEAD"]) {
        let changed = git(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|status| !status.is_empty());
        println!("cargo:rustc-env=CLANKISTRATOR_COMMIT={commit}{}", if changed { "-dirty" } else { "" });
    }
}
