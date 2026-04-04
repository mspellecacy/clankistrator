#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolError(pub String);
impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for ToolError {}

pub mod spawn_agent;

pub mod file_system;
pub use file_system::{ReadFile, WriteFile, ListFiles};

pub mod command_executor;
pub use command_executor::ExecuteCommand;

pub mod rag;
pub use rag::QueryDesignDoc;

pub mod memory;
pub use memory::SearchProjectMemory;
