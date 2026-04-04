use std::path::PathBuf;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "PascalCase")]
pub enum TaskStatus {
    #[serde(alias = "pending", alias = "PENDING")]
    Pending,
    InProgress(String), // Agent ID or Name
    ReadyForReview { output_summary: String },
    Reviewing(String),  // QA Agent ID or Name
    RequiresHumanApproval { output_summary: String, feedback: Option<String> },
    Completed { output_summary: String },
    Failed { error: String },
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
pub enum AgentRole {
    Orchestrator,
    #[serde(alias = "infrastructure", alias = "INFRASTRUCTURE")]
    Infrastructure,
    #[serde(alias = "backend", alias = "BACKEND")]
    Backend,
    #[serde(alias = "frontend", alias = "FRONTEND")]
    Frontend,
    #[serde(alias = "qa", alias = "QA")]
    QA,
    #[serde(alias = "critic", alias = "CRITIC")]
    Critic,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProjectTask {
    pub id: String,
    pub title: String,
    pub description: String,
    pub dependencies: Vec<String>,
    pub status: TaskStatus,
    pub assigned_role: AgentRole,
    pub feedback: Option<String>,
    #[serde(default)]
    pub retries: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NewTaskDef {
    pub title: String,
    pub description: String,
    pub assigned_role: AgentRole,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TaskResult {
    pub task_id: String,
    pub status: String, // "Completed" | "Failed"
    pub summary: String,
    pub files_created: Vec<String>,
    pub feedback: Option<String>,
    #[serde(default)]
    pub new_tasks: Option<Vec<NewTaskDef>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProjectContext {
    pub project_name: String,
    pub tasks: Vec<ProjectTask>,
    pub working_directory: PathBuf,
    pub token_usage: u64,
    pub total_cost: f64,
    pub max_steps: usize,
    pub brave_mode: bool,
    pub model: String,
}

impl ProjectContext {
    pub fn new(name: String, output_dir: PathBuf, max_steps: usize, brave_mode: bool, model: String) -> Self {
        Self {
            project_name: name,
            tasks: Vec::new(),
            working_directory: output_dir,
            token_usage: 0,
            total_cost: 0.0,
            max_steps,
            brave_mode,
            model,
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

    pub fn get_tasks_for_review(&self) -> Vec<&ProjectTask> {
        self.tasks
            .iter()
            .filter(|task| matches!(task.status, TaskStatus::ReadyForReview { .. }))
            .collect()
    }

    pub fn get_tasks_for_approval(&self) -> Vec<&ProjectTask> {
        self.tasks
            .iter()
            .filter(|task| matches!(task.status, TaskStatus::RequiresHumanApproval { .. }))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_ready_tasks() {
        let mut context = ProjectContext::new("Test".into(), PathBuf::from("."), 50, false, "gpt-4o".to_string());
        context.tasks.push(ProjectTask {
            id: "TASK-1".into(),
            title: "Task 1".into(),
            description: "Task 1 description".into(),
            dependencies: vec![],
            status: TaskStatus::Pending,
            assigned_role: AgentRole::Backend,
            feedback: None,
            retries: 0,
        });
        context.tasks.push(ProjectTask {
            id: "TASK-2".into(),
            title: "Task 2".into(),
            description: "Task 2 description".into(),
            dependencies: vec!["TASK-1".into()],
            status: TaskStatus::Pending,
            assigned_role: AgentRole::Frontend,
            feedback: None,
            retries: 0,
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
