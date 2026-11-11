use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use rig::tool::ToolDyn;

use crate::context::STATE_DIR;

pub mod file_system;
pub use file_system::{EditFile, ListFiles, ReadFile, WriteFile};

pub mod command_executor;
pub use command_executor::ExecuteCommand;

pub mod jail;
pub use jail::{Jail, JailPolicy, Writes};

pub mod probe;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolError(pub String);
impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for ToolError {}

/// The directory every agent tool is confined to.
///
/// File tools refuse any path that resolves outside the root. Commands run with
/// the root as their working directory and with a timeout. Unless the sandbox
/// has a [`Jail`], they are ordinary processes running as the current user:
/// the path checks are a guard rail against a model wandering off, and the
/// jail is the security boundary.
///
/// What is hidden is hidden twice, because the file tools run in this process
/// and commands run in the jail: [`resolve`](Self::resolve) refuses the path,
/// and the jail covers it.
#[derive(Debug, Clone)]
pub struct Sandbox {
    root: PathBuf,
    pub command_timeout: Duration,
    /// Whether the file tools refuse environment files (see [`is_env_file`]).
    env_files_hidden: bool,
    /// Paths inside the root that agents may not see, as given and as they resolve.
    hidden: Arc<Vec<PathBuf>>,
    /// Whether the file tools refuse to write the paths in [`RUNS_UNSEEN`].
    unseen_guarded: bool,
    /// The jail commands run in, if any.
    jail: Option<Arc<Jail>>,
    /// The commands started through this sandbox and its clones.
    commands: Arc<CommandScope>,
}

/// The commands running in one sandbox, so that they can be stopped together
/// without touching the commands of other agents.
#[derive(Debug, Default)]
pub(crate) struct CommandScope {
    stopped: AtomicBool,
    pub(crate) pids: std::sync::Mutex<Vec<u32>>,
}

/// Files in a project that run code later, outside any jail, and that a review
/// of the agent's diff never shows: git runs hooks and the programs its config
/// names, and Claude Code runs the hooks in its local settings file.
pub(crate) const RUNS_UNSEEN: &[&str] = &[".git/hooks", ".git/config", ".claude/settings.local.json"];

/// The path with the symlinks in its existing part followed.
fn real_path(path: &Path) -> PathBuf {
    let mut existing = path;
    while !existing.exists() {
        match existing.parent() {
            Some(parent) => existing = parent,
            None => return path.to_path_buf(),
        }
    }
    match (existing.canonicalize(), path.strip_prefix(existing)) {
        (Ok(real), Ok(rest)) => real.join(rest),
        _ => path.to_path_buf(),
    }
}

/// Whether a file name is an environment file that may hold credentials:
/// `.env`, `.envrc` and `.env.<anything>` except the templates projects commit.
pub fn is_env_file(name: &str) -> bool {
    const TEMPLATES: &[&str] = &[".env.example", ".env.sample", ".env.template"];
    name == ".env" || name == ".envrc" || (name.starts_with(".env.") && !TEMPLATES.contains(&name))
}

impl Sandbox {
    /// Creates the root directory if needed and canonicalises it.
    pub fn new(root: impl AsRef<Path>, command_timeout: Duration) -> std::io::Result<Self> {
        std::fs::create_dir_all(root.as_ref())?;
        Ok(Self {
            root: root.as_ref().canonicalize()?,
            command_timeout,
            env_files_hidden: false,
            hidden: Arc::default(),
            unseen_guarded: false,
            jail: None,
            commands: Arc::default(),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Make the file tools refuse environment files. For work in a project
    /// that already exists, where such a file holds real credentials and
    /// whatever a tool returns is sent to the model's provider. Commands are
    /// not covered: a command can still read any file.
    pub fn hiding_env_files(mut self) -> Self {
        self.env_files_hidden = true;
        self
    }

    /// Hide paths inside the root from the agent: the file tools refuse them,
    /// and a jail covers them. Paths are relative to the root.
    pub fn hiding(mut self, paths: &[PathBuf]) -> Result<Self, String> {
        let mut hidden = Vec::new();
        for path in paths {
            let inside = path.components().all(|part| matches!(part, Component::Normal(_) | Component::CurDir));
            if !inside || path.as_os_str().is_empty() {
                return Err(format!(
                    "hidden path {:?} must be relative to the project directory and stay inside it",
                    path.display()
                ));
            }
            let full = self.root.join(path);
            let real = real_path(&full);
            if real != full {
                hidden.push(real);
            }
            hidden.push(full);
        }
        self.hidden = Arc::new(hidden);
        Ok(self)
    }

    /// Make the file tools refuse to write the files in [`RUNS_UNSEEN`]. For
    /// work in a project that already exists.
    pub fn guarding_unseen_code(mut self) -> Self {
        self.unseen_guarded = true;
        self
    }

    /// Run this sandbox's commands in a jail. `writes` says what their writes
    /// to the project do. Call it after [`hiding`](Self::hiding).
    pub fn jailed(mut self, policy: JailPolicy, writes: Writes) -> Result<Self, String> {
        self.jail = Some(Arc::new(Jail::new(policy, writes, &self.root, &self.hidden)?));
        Ok(self)
    }

    /// Delete the files a jailed run kept to itself: its /tmp, and what a
    /// scratch run's commands wrote. Called when the run ends.
    pub fn discard_run_files(&self) {
        if let Some(jail) = &self.jail {
            jail.discard_run_files();
        }
    }

    pub fn jail(&self) -> Option<&Jail> {
        self.jail.as_deref()
    }

    pub(crate) fn hidden(&self) -> &[PathBuf] {
        &self.hidden
    }

    pub(crate) fn commands(&self) -> &Arc<CommandScope> {
        &self.commands
    }

    /// Kill the commands running in this sandbox and refuse to start more.
    ///
    /// Dropping an agent's future is not enough to stop its command: rig runs
    /// each tool call in a task of its own, which carries on without the agent.
    pub fn stop_commands(&self) {
        self.commands.stopped.store(true, Ordering::SeqCst);
        let pids: Vec<u32> =
            self.commands.pids.lock().map(|mut pids| pids.drain(..).collect()).unwrap_or_default();
        for pid in pids {
            command_executor::kill_process_group(Some(pid));
        }
    }

    pub fn commands_stopped(&self) -> bool {
        self.commands.stopped.load(Ordering::SeqCst)
    }

    /// Resolve a model-supplied path to an absolute path inside the root.
    ///
    /// Relative paths are taken relative to the root. Absolute paths are accepted
    /// only if they already point inside the root. `..` may not climb out, and a
    /// symlink inside the root may not lead out of it either.
    pub fn resolve(&self, raw: &str) -> Result<PathBuf, String> {
        let raw = raw.trim();
        let given = Path::new(raw);
        let relative = if given.is_absolute() {
            given
                .strip_prefix(&self.root)
                .map_err(|_| format!("path {raw:?} is outside the project directory; use a path relative to the project root"))?
        } else {
            given
        };

        let mut clean = PathBuf::new();
        for component in relative.components() {
            match component {
                Component::Normal(part) => clean.push(part),
                Component::CurDir => {}
                Component::ParentDir => {
                    if !clean.pop() {
                        return Err(format!("path {raw:?} escapes the project directory"));
                    }
                }
                Component::RootDir | Component::Prefix(_) => {
                    return Err(format!("path {raw:?} is outside the project directory"));
                }
            }
        }
        let full = self.root.join(&clean);

        // Follow symlinks on the part of the path that already exists.
        let mut existing = full.as_path();
        while !existing.exists() {
            match existing.parent() {
                Some(parent) => existing = parent,
                None => break,
            }
        }
        let real = existing
            .canonicalize()
            .map_err(|e| format!("cannot resolve {raw:?}: {e}"))?;
        if !real.starts_with(&self.root) {
            return Err(format!("path {raw:?} leads outside the project directory through a symlink"));
        }
        let real = real_path(&full);
        if self.hidden.iter().any(|hidden| full.starts_with(hidden) || real.starts_with(hidden)) {
            return Err(format!("path {raw:?} is hidden from agents on this server"));
        }
        if self.env_files_hidden {
            // The name as given, and the name a symlink to it really has.
            let named = |path: &Path| path.file_name().and_then(|name| name.to_str()).is_some_and(is_env_file);
            if named(&full) || named(&real) {
                return Err(format!(
                    "path {raw:?} is an environment file, which may hold credentials; it cannot be read or written here"
                ));
            }
        }
        Ok(full)
    }

    /// Project markers found in directories above the root. Build tools and
    /// version control search parent directories, so a root nested inside
    /// another project is worth warning about.
    pub fn enclosing_projects(&self) -> Vec<PathBuf> {
        const MARKERS: &[&str] = &[".git", "Cargo.toml", "package.json", "pyproject.toml", "go.mod"];
        self.root
            .ancestors()
            .skip(1)
            .flat_map(|dir| MARKERS.iter().map(move |marker| dir.join(marker)))
            .filter(|path| path.exists())
            .collect()
    }

    /// Like [`resolve`](Self::resolve), but also refuses Clankistrator's own state directory.
    pub fn resolve_for_write(&self, raw: &str) -> Result<PathBuf, String> {
        let full = self.resolve(raw)?;
        if full.starts_with(self.root.join(STATE_DIR)) {
            return Err(format!("{STATE_DIR}/ is reserved for the orchestrator and cannot be written"));
        }
        if self.unseen_guarded {
            let real = real_path(&full);
            let unseen = RUNS_UNSEEN.iter().map(|path| (path, self.root.join(path)));
            if let Some((path, _)) = unseen.into_iter().find(|(_, at)| full.starts_with(at) || real.starts_with(at)) {
                return Err(format!(
                    "{path} runs code outside the project without showing in a diff, so it cannot be written here"
                ));
            }
        }
        Ok(full)
    }
}

/// Which tools an agent is given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolAccess {
    /// No tools (planning).
    None,
    /// Read files, list directories, run commands (review).
    Inspect,
    /// Inspect plus writing and editing files (workers).
    Full,
    /// Only the harmless probe tool, which returns the code for this token.
    Probe(u32),
}

impl ToolAccess {
    pub fn build(self, sandbox: &Sandbox) -> Vec<Box<dyn ToolDyn>> {
        let mut tools: Vec<Box<dyn ToolDyn>> = Vec::new();
        match self {
            ToolAccess::None => return tools,
            ToolAccess::Probe(token) => {
                tools.push(Box::new(probe::GetCode { token }));
                return tools;
            }
            ToolAccess::Inspect | ToolAccess::Full => {}
        }
        tools.push(Box::new(ReadFile { sandbox: sandbox.clone() }));
        tools.push(Box::new(ListFiles { sandbox: sandbox.clone() }));
        tools.push(Box::new(ExecuteCommand { sandbox: sandbox.clone() }));
        if self == ToolAccess::Full {
            tools.push(Box::new(WriteFile { sandbox: sandbox.clone() }));
            tools.push(Box::new(EditFile { sandbox: sandbox.clone() }));
        }
        tools
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sandbox() -> (tempfile::TempDir, Sandbox) {
        let dir = tempfile::tempdir().unwrap();
        let sb = Sandbox::new(dir.path().join("out"), Duration::from_secs(5)).unwrap();
        (dir, sb)
    }

    #[test]
    fn relative_paths_resolve_inside_the_root() {
        let (_dir, sb) = sandbox();
        assert_eq!(sb.resolve("src/main.rs").unwrap(), sb.root().join("src/main.rs"));
        assert_eq!(sb.resolve("./a/../b.txt").unwrap(), sb.root().join("b.txt"));
        assert_eq!(sb.resolve(".").unwrap(), sb.root());
    }

    #[test]
    fn absolute_paths_are_accepted_only_inside_the_root() {
        let (_dir, sb) = sandbox();
        let inside = sb.root().join("Cargo.toml");
        assert_eq!(sb.resolve(inside.to_str().unwrap()).unwrap(), inside);
        assert!(sb.resolve("/etc/passwd").is_err());
    }

    #[test]
    fn parent_traversal_is_rejected() {
        let (_dir, sb) = sandbox();
        assert!(sb.resolve("../secret").is_err());
        assert!(sb.resolve("a/../../secret").is_err());
        let sneaky = format!("{}/../secret", sb.root().display());
        assert!(sb.resolve(&sneaky).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_cannot_lead_out_of_the_root() {
        let (dir, sb) = sandbox();
        std::fs::write(dir.path().join("outside.txt"), "secret").unwrap();
        std::os::unix::fs::symlink(dir.path(), sb.root().join("link")).unwrap();
        assert!(sb.resolve("link/outside.txt").is_err());
        assert!(sb.resolve("link/new-file.txt").is_err());
    }

    #[test]
    fn enclosing_projects_are_detected() {
        let (dir, sb) = sandbox();
        let before = sb.enclosing_projects();
        std::fs::write(dir.path().join("Cargo.toml"), "").unwrap();
        std::fs::write(sb.root().join("Cargo.toml"), "").unwrap(); // the project's own manifest does not count
        let after = sb.enclosing_projects();
        assert_eq!(after.len(), before.len() + 1);
        assert!(after.contains(&dir.path().canonicalize().unwrap().join("Cargo.toml")));
    }

    #[test]
    fn state_directory_is_read_only_for_agents() {
        let (_dir, sb) = sandbox();
        assert!(sb.resolve_for_write(".clankistrator/state.json").is_err());
        assert!(sb.resolve_for_write("src/lib.rs").is_ok());
    }

    #[test]
    fn environment_files_are_refused_only_when_hidden() {
        let (_dir, sb) = sandbox();
        assert!(sb.resolve(".env.local").is_ok(), "a generated project may have its own");
        let hidden = sb.clone().hiding_env_files();
        for name in [".env", ".env.local", ".env.production", "apps/web/.env.staging", ".env.local.bak", ".envrc"] {
            assert!(hidden.resolve(name).is_err(), "{name}");
            assert!(hidden.resolve_for_write(name).is_err(), "{name}");
        }
        for name in [".env.example", "src/env.ts", ".environment/notes.md", "docs/.envoy.yaml"] {
            assert!(hidden.resolve(name).is_ok(), "{name}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_does_not_reveal_an_environment_file() {
        let (_dir, sb) = sandbox();
        std::fs::write(sb.root().join(".env.production"), "KEY=1").unwrap();
        std::os::unix::fs::symlink(sb.root().join(".env.production"), sb.root().join("settings.txt")).unwrap();
        assert!(sb.clone().hiding_env_files().resolve("settings.txt").is_err());
    }

    #[test]
    fn hidden_paths_are_refused_and_must_stay_inside_the_project() {
        let (_dir, sb) = sandbox();
        std::fs::create_dir_all(sb.root().join("tmp/private")).unwrap();
        let hidden = sb.clone().hiding(&[PathBuf::from("tmp/private"), PathBuf::from("notes.md")]).unwrap();
        for path in ["tmp/private", "tmp/private/a.txt", "./tmp/../tmp/private/deep/b.txt", "notes.md"] {
            assert!(hidden.resolve(path).unwrap_err().contains("hidden"), "{path}");
            assert!(hidden.resolve_for_write(path).is_err(), "{path}");
        }
        for path in ["tmp", "tmp/public.txt", "notes.md.bak", "src/notes.md"] {
            assert!(hidden.resolve(path).is_ok(), "{path}");
        }
        assert!(sb.resolve("tmp/private/a.txt").is_ok(), "nothing is hidden unless asked for");

        for outside in ["../elsewhere", "/etc", ""] {
            assert!(sb.clone().hiding(&[PathBuf::from(outside)]).is_err(), "{outside:?}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_does_not_reveal_a_hidden_path() {
        let (_dir, sb) = sandbox();
        std::fs::create_dir_all(sb.root().join("private")).unwrap();
        std::os::unix::fs::symlink(sb.root().join("private"), sb.root().join("door")).unwrap();
        let hidden = sb.hiding(&[PathBuf::from("private")]).unwrap();
        assert!(hidden.resolve("door/notes.txt").is_err());
    }

    #[test]
    fn files_that_run_unseen_cannot_be_written_once_guarded() {
        let (_dir, sb) = sandbox();
        std::fs::create_dir_all(sb.root().join(".git/hooks")).unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(sb.root().join(".git/hooks"), sb.root().join("scripts")).unwrap();
        let unseen = [".git/hooks/pre-commit", ".git/config", ".claude/settings.local.json"];
        for path in unseen {
            assert!(sb.resolve_for_write(path).is_ok(), "a generated project is the agents' own: {path}");
        }
        let guarded = sb.guarding_unseen_code();
        for path in unseen {
            assert!(guarded.resolve_for_write(path).unwrap_err().contains("without showing in a diff"), "{path}");
            assert!(guarded.resolve(path).is_ok(), "reading is fine: {path}");
        }
        #[cfg(unix)]
        assert!(guarded.resolve_for_write("scripts/pre-commit").is_err(), "not through a symlink either");
        for path in [".git/HEAD", ".claude/settings.json", ".claude/agents/reviewer.md", "src/config"] {
            assert!(guarded.resolve_for_write(path).is_ok(), "{path}");
        }
    }

    #[test]
    fn tool_sets_match_access_level() {
        let (_dir, sb) = sandbox();
        let names = |a: ToolAccess| a.build(&sb).iter().map(|t| t.name()).collect::<Vec<_>>();
        assert!(names(ToolAccess::None).is_empty());
        assert_eq!(names(ToolAccess::Inspect), ["read_file", "list_files", "execute_command"]);
        assert_eq!(
            names(ToolAccess::Full),
            ["read_file", "list_files", "execute_command", "write_file", "edit_file"],
            "only workers get the tools that change files"
        );
        assert_eq!(names(ToolAccess::Probe(1)), ["get_code"], "the probe gets no file or command tools");
    }
}
