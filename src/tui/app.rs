use crate::context::ProjectContext;
use std::sync::Arc;
use tokio::sync::Mutex;
use crossterm::event::{Event, KeyCode};
use anyhow::Result;

#[derive(Debug, PartialEq, Eq)]
pub enum View {
    Tasks,
    Logs,
    HITL,
    Stats,
}

pub struct App {
    pub context: Arc<Mutex<ProjectContext>>,
    pub current_view: View,
    pub running: bool,
    pub task_list_state: usize,
    pub log_scroll: u16,
    pub log_state: tui_logger::TuiWidgetState,
    pub show_modal: bool,
    pub modal_selection: usize,
}

impl App {
    pub fn new(context: Arc<Mutex<ProjectContext>>) -> Self {
        Self {
            context,
            current_view: View::Tasks,
            running: true,
            task_list_state: 0,
            log_scroll: 0,
            log_state: tui_logger::TuiWidgetState::new(),
            show_modal: false,
            modal_selection: 0,
        }
    }

    pub async fn handle_event(&mut self, event: Event) -> Result<()> {
        if let Event::Key(key) = event {
            if self.show_modal {
                match key.code {
                    KeyCode::Esc | KeyCode::Char('m') => self.show_modal = false,
                    KeyCode::Up => {
                        self.modal_selection = self.modal_selection.saturating_sub(1);
                    }
                    KeyCode::Down => {
                        let max_selection = 2; // Brave, Model, Max Steps
                        if self.modal_selection < max_selection {
                            self.modal_selection += 1;
                        }
                    }
                    KeyCode::Enter | KeyCode::Char(' ') | KeyCode::Right => {
                        let mut ctx = self.context.lock().await;
                        match self.modal_selection {
                            0 => ctx.brave_mode = !ctx.brave_mode,
                            1 => {
                                // Cycle models
                                let current_model = ctx.model.clone();
                                let models = vec!["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "claude-3-5-sonnet-20240620"];
                                if let Some(idx) = models.iter().position(|&m| m == current_model) {
                                    ctx.model = models[(idx + 1) % models.len()].to_string();
                                } else {
                                    ctx.model = "gpt-4o".to_string();
                                }
                            },
                            2 => ctx.max_steps += 5,
                            _ => {}
                        }
                    }
                    KeyCode::Left => {
                        let mut ctx = self.context.lock().await;
                        match self.modal_selection {
                            0 => ctx.brave_mode = !ctx.brave_mode,
                            1 => {
                                // Cycle models reverse
                                let current_model = ctx.model.clone();
                                let models = vec!["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "claude-3-5-sonnet-20240620"];
                                if let Some(idx) = models.iter().position(|&m| m == current_model) {
                                    ctx.model = models[(idx + models.len() - 1) % models.len()].to_string();
                                } else {
                                    ctx.model = "gpt-4o".to_string();
                                }
                            },
                            2 => ctx.max_steps = ctx.max_steps.saturating_sub(5),
                            _ => {}
                        }
                    }
                    _ => {}
                }
                return Ok(());
            }

            match key.code {
                KeyCode::Char('q') => self.running = false,
                KeyCode::Char('m') => self.show_modal = true,
                KeyCode::Char('1') => self.current_view = View::Tasks,
                KeyCode::Char('2') => self.current_view = View::Logs,
                KeyCode::Char('3') => self.current_view = View::HITL,
                KeyCode::Char('4') => self.current_view = View::Stats,
                KeyCode::Up => {
                    if self.current_view == View::Tasks {
                        self.task_list_state = self.task_list_state.saturating_sub(1);
                    }
                },
                KeyCode::Down => {
                    if self.current_view == View::Tasks {
                        let task_count = self.context.lock().await.tasks.len();
                        if self.task_list_state + 1 < task_count {
                            self.task_list_state += 1;
                        }
                    }
                },
                KeyCode::Char('a') => {
                    // Quick Approve in HITL view
                    if self.current_view == View::HITL {
                        let mut ctx = self.context.lock().await;
                        let approval_tasks: Vec<String> = ctx.get_tasks_for_approval().iter().map(|t| t.id.clone()).collect();
                        if let Some(task_id) = approval_tasks.first() {
                            if let Some(task) = ctx.get_task_mut(task_id) {
                                if let crate::context::TaskStatus::RequiresHumanApproval { output_summary, .. } = &task.status {
                                    task.status = crate::context::TaskStatus::Completed {
                                        output_summary: output_summary.clone(),
                                    };
                                }
                            }
                        }
                    }
                },
                _ => {}
            }
        }
        Ok(())
    }
}
