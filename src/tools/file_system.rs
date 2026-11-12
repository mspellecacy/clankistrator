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
pub struct EditFileArgs {
    pub path: String,
    pub old_text: String,
    pub new_text: String,
    #[serde(default)]
    pub replace_all: bool,
}

/// Changes part of a file, so that a small change to a large file does not
/// mean writing the whole file out again (and getting the rest of it wrong).
pub struct EditFile {
    pub sandbox: Sandbox,
}

impl EditFile {
    fn edit(&self, args: &EditFileArgs) -> Result<String, String> {
        let path = self.sandbox.resolve_for_write(&args.path)?;
        if args.old_text.is_empty() {
            return Err("old_text is empty. To create a file, use write_file.".to_string());
        }
        if args.old_text == args.new_text {
            return Err("old_text and new_text are the same, so nothing would change.".to_string());
        }
        let content =
            fs::read_to_string(&path).map_err(|e| format!("could not read {} as text: {e}", args.path))?;
        let found = content.matches(&args.old_text).count();
        match found {
            0 => Err(format!(
                "old_text was not found in {}. It must match the file exactly, including indentation \
                 and line breaks. Read the file again and copy the text from it.",
                args.path
            )),
            1 => Ok(()),
            _ if args.replace_all => Ok(()),
            _ => Err(format!(
                "old_text occurs {found} times in {}. Include more of the surrounding lines so that it \
                 matches one place, or set replace_all to change every occurrence.",
                args.path
            )),
        }?;
        fs::write(&path, content.replace(&args.old_text, &args.new_text))
            .map_err(|e| format!("failed to write {}: {e}", args.path))?;
        Ok(format!("Replaced {found} occurrence{} in {}", if found == 1 { "" } else { "s" }, args.path))
    }
}

impl Tool for EditFile {
    const NAME: &'static str = "edit_file";
    type Error = ToolError;
    type Args = EditFileArgs;
    type Output = String;

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        tracing::info!("Editing file: {}", args.path);
        Ok(self.edit(&args).unwrap_or_else(|e| format!("Error: {e}")))
    }

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Change part of an existing file by replacing one piece of text with another. \
                          Prefer this to write_file for a change to an existing file. \
                          Paths are relative to the project root."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path of the file, relative to the project root (e.g. 'src/main.rs')."
                    },
                    "old_text": {
                        "type": "string",
                        "description": "The text to replace, copied exactly from the file, including indentation. It must occur exactly once unless replace_all is set."
                    },
                    "new_text": {
                        "type": "string",
                        "description": "The text to put in its place. Empty removes old_text."
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence of old_text instead of requiring exactly one. Defaults to false."
                    }
                },
                "required": ["path", "old_text", "new_text"]
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
        for path in ["../escaped.txt", "/tmp/clankistrator-escaped.txt"] {
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

    #[tokio::test]
    async fn edit_replaces_one_place_and_says_why_when_it_cannot() {
        let (_dir, sandbox) = sandbox();
        let path = sandbox.root().join("src/lib.rs");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "fn a() {\n    1\n}\nfn b() {\n    1\n}\n").unwrap();
        let tool = EditFile { sandbox: sandbox.clone() };
        let edit = |old: &str, new: &str, replace_all: bool| {
            tool.call(EditFileArgs {
                path: "src/lib.rs".into(),
                old_text: old.into(),
                new_text: new.into(),
                replace_all,
            })
        };

        let done = edit("fn a() {\n    1", "fn a() {\n    2", false).await.unwrap();
        assert_eq!(done, "Replaced 1 occurrence in src/lib.rs");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "fn a() {\n    2\n}\nfn b() {\n    1\n}\n");

        let unchanged = std::fs::read_to_string(&path).unwrap();
        assert!(edit("    3", "    4", false).await.unwrap().contains("was not found"));
        assert!(edit("fn ", "pub fn ", false).await.unwrap().contains("occurs 2 times"));
        assert!(edit("", "x", false).await.unwrap().starts_with("Error"));
        assert!(edit("fn a", "fn a", false).await.unwrap().starts_with("Error"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), unchanged, "a refused edit changes nothing");

        assert_eq!(edit("fn ", "pub fn ", true).await.unwrap(), "Replaced 2 occurrences in src/lib.rs");
        assert!(std::fs::read_to_string(&path).unwrap().starts_with("pub fn a() {"));

        let outside = tool
            .call(EditFileArgs {
                path: "../lib.rs".into(),
                old_text: "a".into(),
                new_text: "b".into(),
                replace_all: false,
            })
            .await
            .unwrap();
        assert!(outside.starts_with("Error"), "{outside}");
    }
}
