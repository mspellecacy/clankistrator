use serde::{Deserialize, Serialize};
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use crate::tools::ToolError;
use std::fs;
use std::path::Path;

#[derive(Deserialize, Serialize)]
pub struct ReadFileArgs {
    pub path: String,
}

pub struct ReadFile;

impl Tool for ReadFile {
    const NAME: &'static str = "read_file";
    type Error = ToolError;
    type Args = ReadFileArgs;
    type Output = String;

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        tracing::debug!("Reading file: {}", args.path);
        fs::read_to_string(&args.path)
            .map_err(|e| ToolError(format!("Failed to read file {}: {}", args.path, e)))
    }

    async fn definition(&self, _path: String) -> ToolDefinition {
        serde_json::from_value(serde_json::json!({
            "name": "read_file",
            "description": "Read the content of a file from the disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to read."
                    }
                },
                "required": ["path"]
            }
        })).expect("Invalid tool definition")
    }
}

#[derive(Deserialize, Serialize)]
pub struct WriteFileArgs {
    pub path: String,
    pub content: String,
}

pub struct WriteFile;

impl Tool for WriteFile {
    const NAME: &'static str = "write_file";
    type Error = ToolError;
    type Args = WriteFileArgs;
    type Output = String;

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        tracing::info!("Writing file: {}", args.path);
        let path = Path::new(&args.path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| ToolError(format!("Failed to create directory {}: {}", parent.display(), e)))?;
        }
        fs::write(&args.path, &args.content)
            .map_err(|e| ToolError(format!("Failed to write to file {}: {}", args.path, e)))?;
        Ok(format!("Successfully wrote to {}", args.path))
    }

    async fn definition(&self, _path: String) -> ToolDefinition {
        serde_json::from_value(serde_json::json!({
            "name": "write_file",
            "description": "Write content to a file on the disk. Creates directories if they don't exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to write."
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file."
                    }
                },
                "required": ["path", "content"]
            }
        })).expect("Invalid tool definition")
    }
}

#[derive(Deserialize, Serialize)]
pub struct ListFilesArgs {
    pub directory: String,
}

pub struct ListFiles;

impl Tool for ListFiles {
    const NAME: &'static str = "list_files";
    type Error = ToolError;
    type Args = ListFilesArgs;
    type Output = String;

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let entries = fs::read_dir(&args.directory)
            .map_err(|e| ToolError(format!("Failed to list directory {}: {}", args.directory, e)))?;

        let mut files = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| ToolError(format!("Error reading directory entry: {}", e)))?;
            let path = entry.path();
            let file_name = path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();
            let is_dir = path.is_dir();
            files.push(format!("{}{}", file_name, if is_dir { "/" } else { "" }));
        }

        Ok(files.join("\n"))
    }

    async fn definition(&self, _path: String) -> ToolDefinition {
        serde_json::from_value(serde_json::json!({
            "name": "list_files",
            "description": "List files and directories in a given directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "The directory to list files from."
                    }
                },
                "required": ["directory"]
            }
        })).expect("Invalid tool definition")
    }
}
