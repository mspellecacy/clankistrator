//! An operating-system jail for the commands an agent runs in a project that
//! already exists, built with bubblewrap (`bwrap`, Linux only).
//!
//! The path checks in [`Sandbox`](super::Sandbox) bind only the file tools. A
//! command is a program: it opens whatever the user can open, and what it
//! prints goes to a model provider. The jail is the boundary for commands:
//!
//! - the project is the only part of the user's files that is there, read-only
//!   unless the agent was given write access;
//! - the home directory is empty, apart from what the server's owner listed;
//! - environment files and the paths the owner hid are not there;
//! - there is no network, unless the owner allowed it;
//! - the process namespace is private, so the server's own environment (which
//!   holds the provider keys) cannot be read through /proc;
//! - the environment is a short list of harmless variables.
//!
//! Everything a jail allows is decided when the server starts. Nothing in a
//! tool call can loosen it.

use std::collections::VecDeque;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use tokio::process::Command;

use super::{RUNS_UNSEEN, is_env_file};

const BWRAP: &str = "bwrap";

/// Directories that hold other programs' sockets and scratch files. Each is
/// replaced by an empty one.
const EMPTIED: &[&str] = &["/tmp", "/var/tmp", "/run", "/mnt", "/media"];

/// The only variables a jailed command inherits. Everything else, including
/// whatever the server read from its settings file, stays outside.
const ENV_ALLOWED: &[&str] = &[
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "TZ",
    "NO_COLOR", "CI",
    // Where toolchains live. The directories still have to be made readable.
    "CARGO_HOME", "RUSTUP_HOME", "RUSTUP_TOOLCHAIN", "NVM_DIR", "PNPM_HOME", "BUN_INSTALL", "VOLTA_HOME",
    "PYENV_ROOT", "GOPATH", "GOROOT", "JAVA_HOME",
];

/// Directories the search for environment files does not enter: they are
/// large, and what is in them was put there by a tool.
const NOT_SEARCHED: &[&str] = &[".git", "node_modules", "target", "dist", "build", ".next", "vendor", ".venv"];
const SEARCH_DEPTH: usize = 6;
const SEARCH_LIMIT: usize = 50_000;

/// What the server's owner allows jailed commands.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JailPolicy {
    /// Whether commands share the machine's network.
    pub network: bool,
    /// Paths outside the project that commands may read, such as a toolchain
    /// installed under the home directory.
    pub readable: Vec<PathBuf>,
    /// The home directory to empty. `None` leaves no home to hide.
    pub home: Option<PathBuf>,
}

impl JailPolicy {
    /// The policy for the user running the server: nothing extra allowed.
    pub fn for_current_user() -> Self {
        Self { home: std::env::var_os("HOME").map(PathBuf::from).filter(|home| home.is_absolute()), ..Self::default() }
    }
}

/// One project's jail: the policy, plus what was found in the project when
/// the agent's run was prepared.
#[derive(Debug)]
pub struct Jail {
    policy: JailPolicy,
    writable: bool,
    /// Environment files in the project, each covered by an unreadable file.
    env_files: Vec<PathBuf>,
    /// For a worktree or submodule: the git directory its `.git` file points to.
    git_dir: Option<PathBuf>,
}

/// Check that bubblewrap can build a jail on this machine, by building one.
/// Being installed is not enough: unprivileged user namespaces can be off.
pub fn probe() -> Result<(), String> {
    if !cfg!(target_os = "linux") {
        return Err("commands are jailed with bubblewrap, which exists only on Linux".to_string());
    }
    let output = std::process::Command::new(BWRAP)
        .args(["--unshare-all", "--die-with-parent", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "true"])
        .stdin(std::process::Stdio::null())
        .output()
        .map_err(|e| format!("could not start bubblewrap (`{BWRAP}`): {e}. Is it installed?"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "bubblewrap (`{BWRAP}`) could not build a jail: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

/// The git directory a `.git` file names, with the repository it belongs to:
/// a worktree's git directory is `<repository>/.git/worktrees/<name>`, and git
/// needs the whole of `<repository>/.git`.
fn linked_git_dir(root: &Path) -> Option<PathBuf> {
    let text = std::fs::read_to_string(root.join(".git")).ok()?;
    let named = text.lines().find_map(|line| line.strip_prefix("gitdir:"))?.trim();
    let git_dir = root.join(named).canonicalize().ok()?;
    let common = std::fs::read_to_string(git_dir.join("commondir"))
        .ok()
        .and_then(|common| git_dir.join(common.trim()).canonicalize().ok());
    Some(common.unwrap_or(git_dir))
}

/// The environment files in a project, found by walking it. The walk is
/// bounded, so a file nested very deep or inside a build directory is missed.
fn env_files(root: &Path, hidden: &[PathBuf]) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut queue: VecDeque<(PathBuf, usize)> = VecDeque::from([(root.to_path_buf(), 0)]);
    let mut seen = 0;
    while let Some((dir, depth)) = queue.pop_front() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            seen += 1;
            if seen > SEARCH_LIMIT {
                return found;
            }
            let path = entry.path();
            let Ok(kind) = entry.file_type() else { continue };
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if kind.is_dir() {
                if depth < SEARCH_DEPTH && !NOT_SEARCHED.contains(&name.as_ref()) && !hidden.contains(&path) {
                    queue.push_back((path, depth + 1));
                }
            } else if kind.is_file() && is_env_file(&name) {
                found.push(path);
            }
        }
    }
    found
}

impl Jail {
    /// Work out the jail for one agent run in the project at `root`.
    pub(super) fn new(policy: JailPolicy, writable: bool, root: &Path, hidden: &[PathBuf]) -> Result<Self, String> {
        if let Some(home) = &policy.home
            && home.starts_with(root)
        {
            return Err(format!(
                "commands cannot be jailed in {}: it contains your home directory, so nothing would be hidden",
                root.display()
            ));
        }
        Ok(Self { policy, writable, env_files: env_files(root, hidden), git_dir: linked_git_dir(root) })
    }

    pub fn network(&self) -> bool {
        self.policy.network
    }

    /// The bubblewrap arguments that build the jail, up to the program to run.
    /// Later mounts cover earlier ones, so the order is: empty what is not the
    /// project's, put the project in, then cover the parts of it to hide.
    pub(super) fn arguments(&self, root: &Path, hidden: &[PathBuf], cwd: &Path) -> Vec<OsString> {
        let mut args: Vec<OsString> = Vec::new();
        let mut add = |parts: &[&std::ffi::OsStr]| args.extend(parts.iter().map(|part| part.to_os_string()));
        let flag = |text: &'static str| std::ffi::OsStr::new(text);
        let project_bind = flag(if self.writable { "--bind" } else { "--ro-bind" });

        add(&[flag("--unshare-all")]);
        if self.policy.network {
            add(&[flag("--share-net")]);
        }
        // A new session has no controlling terminal to type into. Its
        // processes leave our process group, so they are stopped through
        // --die-with-parent instead: killing bwrap takes the namespace down.
        add(&[flag("--die-with-parent"), flag("--new-session")]);
        add(&[flag("--ro-bind"), flag("/"), flag("/"), flag("--dev"), flag("/dev"), flag("--proc"), flag("/proc")]);

        for dir in EMPTIED.iter().map(Path::new).filter(|dir| dir.is_dir()) {
            add(&[flag("--tmpfs"), dir.as_os_str()]);
        }
        if let Some(home) = self.policy.home.as_deref().filter(|home| home.is_dir()) {
            add(&[flag("--tmpfs"), home.as_os_str()]);
        }
        // With a network, names still have to resolve, and the resolver's
        // file usually lives under /run.
        if self.policy.network
            && let Ok(resolv) = Path::new("/etc/resolv.conf").canonicalize()
            && let Some(dir) = resolv.parent().filter(|dir| dir.starts_with("/run"))
        {
            add(&[flag("--ro-bind"), dir.as_os_str(), dir.as_os_str()]);
        }
        for path in self.policy.readable.iter().filter(|path| path.exists()) {
            add(&[flag("--ro-bind"), path.as_os_str(), path.as_os_str()]);
        }

        if let Some(git_dir) = self.git_dir.as_deref().filter(|dir| dir.is_dir()) {
            add(&[project_bind, git_dir.as_os_str(), git_dir.as_os_str()]);
        }
        add(&[project_bind, root.as_os_str(), root.as_os_str()]);

        if self.writable {
            // A command may change the project, but not the files that run
            // code later without ever showing in a diff.
            let git_dir = self.git_dir.clone().unwrap_or_else(|| root.join(".git"));
            let mut unseen: Vec<PathBuf> = vec![git_dir.join("hooks"), git_dir.join("config")];
            unseen.extend(RUNS_UNSEEN.iter().filter(|path| !path.starts_with(".git/")).map(|path| root.join(path)));
            // Mount points cannot be renamed or removed, but the directory
            // above one can: `mv .git aside`, and a writable copy takes its
            // place. So the directory holding a covered file is bound onto
            // itself first, which makes it a mount point as well.
            let mut anchored: Vec<PathBuf> = vec![root.to_path_buf()];
            anchored.extend(self.git_dir.clone());
            for path in unseen {
                let Some(parent) = path.parent().filter(|parent| parent.is_dir()) else { continue };
                if !anchored.iter().any(|dir| dir == parent) {
                    add(&[flag("--bind"), parent.as_os_str(), parent.as_os_str()]);
                    anchored.push(parent.to_path_buf());
                }
                if path.exists() {
                    add(&[flag("--ro-bind"), path.as_os_str(), path.as_os_str()]);
                } else if parent != root && parent != git_dir {
                    // A file that is not there yet cannot be covered, so the
                    // directory it would be created in is read-only instead.
                    add(&[flag("--ro-bind"), parent.as_os_str(), parent.as_os_str()]);
                }
            }
        }

        for path in hidden {
            match std::fs::symlink_metadata(path) {
                Ok(meta) if meta.is_dir() => add(&[flag("--tmpfs"), path.as_os_str()]),
                Ok(meta) if meta.is_file() => add(&[flag("--ro-bind"), flag("/dev/null"), path.as_os_str()]),
                _ => {}
            }
        }
        for file in self.env_files.iter().filter(|file| file.is_file()) {
            add(&[flag("--ro-bind"), flag("/dev/null"), file.as_os_str()]);
        }

        add(&[flag("--chdir"), cwd.as_os_str(), flag("--")]);
        args
    }

    /// The command that runs `program` inside the jail.
    pub(super) fn command(&self, root: &Path, hidden: &[PathBuf], program: &str, args: &[String], cwd: &Path) -> Command {
        let mut cmd = Command::new(BWRAP);
        cmd.args(self.arguments(root, hidden, cwd)).arg(program).args(args).current_dir(cwd);
        cmd.env_clear();
        for key in ENV_ALLOWED {
            if let Some(value) = std::env::var_os(key) {
                cmd.env(key, value);
            }
        }
        cmd
    }
}
