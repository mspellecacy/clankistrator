use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use std::fs;

use crate::tools::{Sandbox, ToolError};

/// Files larger than this are truncated when read, so one file cannot swamp the context.
const MAX_READ_BYTES: usize = 100_000;

// Tool failures are returned as `Ok(message)` on purpose: the model should see
// what went wrong and correct itself, rather than the whole agent run aborting.

#[derive(Deserialize, Serialize)]
pub struct ReadFileArgs {
    pub path: String,
}

pub struct ReadFile {
    pub sandbox: Sandbox,
}

impl Tool for ReadFile {
    const NAME: &'static str = "read_file";
    type Error = ToolError;
    type Args = ReadFileArgs;
    type Output = String;

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        tracing::debug!("Reading file: {}", args.path);
        let path = match self.sandbox.resolve(&args.path) {
            Ok(p) => p,
            Err(e) => return Ok(format!("Error: {e}")),
        };
        match fs::read(&path) {
            Ok(bytes) => {
                let total = bytes.len();
                let shown = &bytes[..total.min(MAX_READ_BYTES)];
                let mut text = String::from_utf8_lossy(shown).into_owned();
                if total > MAX_READ_BYTES {
                    text.push_str(&format!(
                        "\n[... truncated: showing the first {MAX_READ_BYTES} of {total} bytes ...]"
                    ));
                }
                Ok(text)
            }
            Err(e) => Ok(format!("Error reading file {}: {}", args.path, e)),
        }
    }

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Read a file. Paths are relative to the project root.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path of the file, relative to the project root (e.g. 'src/main.rs')."
                    }
                },
                "required": ["path"]
            }),
        }
    }
}

#[derive(Deserialize, Serialize)]
pub struct WriteFileArgs {
    pub path: String,
    pub content: String,
}

pub struct WriteFile {
    pub sandbox: Sandbox,
}

impl Tool for WriteFile {
    const NAME: &'static str = "write_file";
    type Error = ToolError;
    type Args = WriteFileArgs;
    type Output = String;

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        tracing::info!("Writing file: {}", args.path);
        let path = match self.sandbox.resolve_for_write(&args.path) {
            Ok(p) => p,
            Err(e) => return Ok(format!("Error: {e}")),
        };
        if let Some(parent) = path.parent()
            && let Err(e) = fs::create_dir_all(parent)
        {
            return Ok(format!("Failed to create directory for {}: {}", args.path, e));
        }
        match fs::write(&path, &args.content) {
            Ok(_) => Ok(format!("Wrote {} bytes to {}", args.content.len(), args.path)),
            Err(e) => Ok(format!("Failed to write to file {}: {}", args.path, e)),
        }
    }

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Create or overwrite a file with the given content. Parent directories are created. Paths are relative to the project root.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path of the file, relative to the project root (e.g. 'src/main.rs')."
                    },
                    "content": {
                        "type": "string",
                        "description": "The complete new content of the file."
                    }
                },
                "required": ["path", "content"]
            }),
        }
    }
}

#[derive(Deserialize, Serialize)]
pub struct ListFilesArgs {
    #[serde(default = "default_directory")]
    pub directory: String,
}

fn default_directory() -> String {
    ".".to_string()
}

pub struct ListFiles {
    pub sandbox: Sandbox,
}

impl Tool for ListFiles {
    const NAME: &'static str = "list_files";
    type Error = ToolError;
    type Args = ListFilesArgs;
    type Output = String;

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let dir = match self.sandbox.resolve(&args.directory) {
            Ok(p) => p,
            Err(e) => return Ok(format!("Error: {e}")),
        };
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(e) => return Ok(format!("Failed to list directory {}: {}", args.directory, e)),
        };

        let mut files: Vec<String> = entries
            .flatten()
            .map(|entry| {
                let name = entry.file_name().to_string_lossy().into_owned();
                if entry.path().is_dir() { format!("{name}/") } else { name }
            })
            .collect();
        files.sort();

        if files.is_empty() {
            Ok("(empty directory)".to_string())
        } else {
            Ok(files.join("\n"))
        }
    }

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "List the files and directories in a directory. Paths are relative to the project root; use '.' for the root itself.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory to list, relative to the project root. Use '.' for the root."
                    }
                },
                "required": ["directory"]
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn sandbox() -> (tempfile::TempDir, Sandbox) {
        let dir = tempfile::tempdir().unwrap();
        let sb = Sandbox::new(dir.path().join("out"), Duration::from_secs(5)).unwrap();
        (dir, sb)
    }

    #[tokio::test]
    async fn write_then_read_and_list() {
        let (_dir, sb) = sandbox();
        let write = WriteFile { sandbox: sb.clone() };
        let out = write
            .call(WriteFileArgs { path: "src/main.rs".into(), content: "fn main() {}".into() })
            .await
            .unwrap();
        assert!(out.starts_with("Wrote"), "{out}");
        assert!(sb.root().join("src/main.rs").is_file());

        let read = ReadFile { sandbox: sb.clone() };
        let content = read.call(ReadFileArgs { path: "src/main.rs".into() }).await.unwrap();
        assert_eq!(content, "fn main() {}");

        let list = ListFiles { sandbox: sb.clone() };
        assert_eq!(list.call(ListFilesArgs { directory: ".".into() }).await.unwrap(), "src/");
    }

    #[tokio::test]
    async fn writes_outside_the_root_are_refused() {
        let (dir, sb) = sandbox();
        let write = WriteFile { sandbox: sb.clone() };
        for path in ["../escaped.txt", "/tmp/clankistrator-escaped.txt", ".clankistrator/state.json"] {
            let out = write
                .call(WriteFileArgs { path: path.into(), content: "x".into() })
                .await
                .unwrap();
            assert!(out.starts_with("Error:"), "{path}: {out}");
        }
        assert!(!dir.path().join("escaped.txt").exists());

        let read = ReadFile { sandbox: sb };
        let out = read.call(ReadFileArgs { path: "/etc/hostname".into() }).await.unwrap();
        assert!(out.starts_with("Error:"), "{out}");
    }
}
