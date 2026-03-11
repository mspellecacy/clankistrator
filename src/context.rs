use std::path::PathBuf;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum TaskStatus {
    Pending,
    InProgress(String), // Agent ID or Name
    Completed { output_summary: String },
    Failed { error: String },
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub enum AgentRole {
    Orchestrator,
    Infrastructure,
    Backend,
    Frontend,
    QA,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProjectTask {
    pub id: String,
    pub title: String,
    pub description: String,
    pub dependencies: Vec<String>,
    pub status: TaskStatus,
    pub assigned_role: AgentRole,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TaskResult {
    pub task_id: String,
    pub status: String, // "Completed" | "Failed"
    pub summary: String,
    pub files_created: Vec<String>,
    pub feedback: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProjectContext {
    pub project_name: String,
    pub tasks: Vec<ProjectTask>,
    pub working_directory: PathBuf,
}

impl ProjectContext {
    pub fn new(name: String, output_dir: PathBuf) -> Self {
        Self {
            project_name: name,
            tasks: Vec::new(),
            working_directory: output_dir,
        }
    }

    pub fn get_task_mut(&mut self, task_id: &str) -> Option<&mut ProjectTask> {
        self.tasks.iter_mut().find(|t| t.id == task_id)
    }

    pub fn get_task(&self, task_id: &str) -> Option<&ProjectTask> {
        self.tasks.iter().find(|t| t.id == task_id)
    }

    pub fn is_task_completed(&self, task_id: &str) -> bool {
        self.get_task(task_id)
            .map(|t| matches!(t.status, TaskStatus::Completed { .. }))
            .unwrap_or(false)
    }

    pub fn get_ready_tasks(&self) -> Vec<&ProjectTask> {
        self.tasks
            .iter()
            .filter(|task| {
                if !matches!(task.status, TaskStatus::Pending) {
                    return false;
                }
                task.dependencies
                    .iter()
                    .all(|dep_id| self.is_task_completed(dep_id))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_ready_tasks() {
        let mut context = ProjectContext::new("Test".into(), PathBuf::from("."));
        context.tasks.push(ProjectTask {
            id: "TASK-1".into(),
            title: "Task 1".into(),
            description: "Task 1 description".into(),
            dependencies: vec![],
            status: TaskStatus::Pending,
            assigned_role: AgentRole::Backend,
        });
        context.tasks.push(ProjectTask {
            id: "TASK-2".into(),
            title: "Task 2".into(),
            description: "Task 2 description".into(),
            dependencies: vec!["TASK-1".into()],
            status: TaskStatus::Pending,
            assigned_role: AgentRole::Frontend,
        });

        let ready = context.get_ready_tasks();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, "TASK-1");

        // Complete TASK-1
        context.get_task_mut("TASK-1").unwrap().status = TaskStatus::Completed {
            output_summary: "Done".into(),
        };

        let ready = context.get_ready_tasks();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, "TASK-2");
    }
}
