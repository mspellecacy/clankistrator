use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use rig::tool::ToolDyn;

use crate::context::STATE_DIR;

pub mod file_system;
pub use file_system::{ListFiles, ReadFile, WriteFile};

pub mod command_executor;
pub use command_executor::ExecuteCommand;

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
/// the root as their working directory and with a timeout, but they are ordinary
/// processes running as the current user: this is a guard rail against a model
/// wandering off, not a security boundary.
#[derive(Debug, Clone)]
pub struct Sandbox {
    root: PathBuf,
    pub command_timeout: Duration,
}

impl Sandbox {
    /// Creates the root directory if needed and canonicalises it.
    pub fn new(root: impl AsRef<Path>, command_timeout: Duration) -> std::io::Result<Self> {
        std::fs::create_dir_all(root.as_ref())?;
        Ok(Self { root: root.as_ref().canonicalize()?, command_timeout })
    }

    pub fn root(&self) -> &Path {
        &self.root
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
    /// Inspect plus writing files (workers).
    Full,
}

impl ToolAccess {
    pub fn build(self, sandbox: &Sandbox) -> Vec<Box<dyn ToolDyn>> {
        let mut tools: Vec<Box<dyn ToolDyn>> = Vec::new();
        if self == ToolAccess::None {
            return tools;
        }
        tools.push(Box::new(ReadFile { sandbox: sandbox.clone() }));
        tools.push(Box::new(ListFiles { sandbox: sandbox.clone() }));
        tools.push(Box::new(ExecuteCommand { sandbox: sandbox.clone() }));
        if self == ToolAccess::Full {
            tools.push(Box::new(WriteFile { sandbox: sandbox.clone() }));
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
    fn tool_sets_match_access_level() {
        let (_dir, sb) = sandbox();
        let names = |a: ToolAccess| a.build(&sb).iter().map(|t| t.name()).collect::<Vec<_>>();
        assert!(names(ToolAccess::None).is_empty());
        assert!(!names(ToolAccess::Inspect).contains(&"write_file".to_string()));
        assert!(names(ToolAccess::Full).contains(&"write_file".to_string()));
    }
}
