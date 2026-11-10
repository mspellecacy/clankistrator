use anyhow::Result;
use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use tui_logger::TuiWidgetEvent;

use crate::llm::check_credentials;
use crate::orchestrator::SharedContext;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    Tasks,
    Logs,
    Hitl,
    Stats,
}

impl View {
    const ALL: [View; 4] = [View::Tasks, View::Logs, View::Hitl, View::Stats];

    pub fn index(self) -> usize {
        Self::ALL.iter().position(|v| *v == self).unwrap_or(0)
    }
}

/// A one-line text prompt shown over the current view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Input {
    /// Feedback for the task being rejected.
    Reject { task_id: String, text: String },
    /// A new model name.
    Model { text: String },
}

impl Input {
    fn text_mut(&mut self) -> &mut String {
        match self {
            Input::Reject { text, .. } | Input::Model { text } => text,
        }
    }
}

pub const MODAL_ROWS: usize = 3; // Brave, Model, Max Steps

pub struct App {
    pub context: SharedContext,
    pub current_view: View,
    pub running: bool,
    pub selected_task: usize,
    pub log_state: tui_logger::TuiWidgetState,
    pub show_modal: bool,
    pub modal_selection: usize,
    pub input: Option<Input>,
}

impl App {
    pub fn new(context: SharedContext) -> Self {
        Self {
            context,
            current_view: View::Tasks,
            running: true,
            selected_task: 0,
            log_state: tui_logger::TuiWidgetState::new(),
            show_modal: false,
            modal_selection: 0,
            input: None,
        }
    }

    pub async fn handle_event(&mut self, event: Event) -> Result<()> {
        let Event::Key(key) = event else { return Ok(()) };
        if key.kind == KeyEventKind::Release {
            return Ok(());
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            self.running = false;
            return Ok(());
        }

        if self.input.is_some() {
            self.handle_input_key(key.code).await;
        } else if self.show_modal {
            self.handle_modal_key(key.code).await;
        } else {
            self.handle_view_key(key.code).await;
        }
        Ok(())
    }

    async fn handle_input_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Esc => self.input = None,
            KeyCode::Backspace => {
                if let Some(input) = &mut self.input {
                    input.text_mut().pop();
                }
            }
            KeyCode::Char(c) => {
                if let Some(input) = &mut self.input {
                    input.text_mut().push(c);
                }
            }
            KeyCode::Enter => match self.input.take() {
                Some(Input::Reject { task_id, text }) => {
                    let mut ctx = self.context.lock().await;
                    if ctx.reject(&task_id, &text) {
                        tracing::info!("TUI [{task_id}]: rejected, sent back to the worker");
                        let _ = ctx.save();
                    }
                }
                Some(Input::Model { text }) => {
                    let model = text.trim().to_string();
                    match check_credentials(&model) {
                        Ok(()) => {
                            tracing::info!("TUI: model changed to {model}; it applies to agents started from now on");
                            self.context.lock().await.model = model;
                        }
                        Err(e) => tracing::warn!("TUI: model not changed: {e}"),
                    }
                }
                None => {}
            },
            _ => {}
        }
    }

    async fn handle_modal_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Esc | KeyCode::Char('m') => self.show_modal = false,
            KeyCode::Up => self.modal_selection = self.modal_selection.saturating_sub(1),
            KeyCode::Down => self.modal_selection = (self.modal_selection + 1).min(MODAL_ROWS - 1),
            KeyCode::Enter | KeyCode::Char(' ') | KeyCode::Right | KeyCode::Left => {
                let mut ctx = self.context.lock().await;
                match self.modal_selection {
                    0 => ctx.brave_mode = !ctx.brave_mode,
                    1 => {
                        if code != KeyCode::Left && code != KeyCode::Right {
                            self.input = Some(Input::Model { text: ctx.model.clone() });
                        }
                    }
                    2 => {
                        if code == KeyCode::Left {
                            ctx.max_steps = ctx.max_steps.saturating_sub(5);
                        } else {
                            ctx.max_steps += 5;
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    async fn handle_view_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Char('q') => self.running = false,
            KeyCode::Char('m') => self.show_modal = true,
            KeyCode::Char('1') => self.current_view = View::Tasks,
            KeyCode::Char('2') => self.current_view = View::Logs,
            KeyCode::Char('3') => self.current_view = View::Hitl,
            KeyCode::Char('4') => self.current_view = View::Stats,
            KeyCode::Tab => {
                self.current_view = View::ALL[(self.current_view.index() + 1) % View::ALL.len()];
            }
            KeyCode::Up if self.current_view == View::Tasks => {
                self.selected_task = self.selected_task.saturating_sub(1);
            }
            KeyCode::Down if self.current_view == View::Tasks => {
                let task_count = self.context.lock().await.tasks.len();
                if self.selected_task + 1 < task_count {
                    self.selected_task += 1;
                }
            }
            KeyCode::PageUp if self.current_view == View::Logs => {
                self.log_state.transition(TuiWidgetEvent::PrevPageKey);
            }
            KeyCode::PageDown if self.current_view == View::Logs => {
                self.log_state.transition(TuiWidgetEvent::NextPageKey);
            }
            KeyCode::Esc if self.current_view == View::Logs => {
                self.log_state.transition(TuiWidgetEvent::EscapeKey);
            }
            KeyCode::Char('a') if self.current_view == View::Hitl => {
                let mut ctx = self.context.lock().await;
                let first = ctx.get_tasks_for_approval().first().map(|t| t.id.clone());
                if let Some(task_id) = first
                    && ctx.approve(&task_id)
                {
                    tracing::info!("TUI [{task_id}]: approved");
                    let _ = ctx.save();
                }
            }
            KeyCode::Char('r') if self.current_view == View::Hitl => {
                let ctx = self.context.lock().await;
                if let Some(task) = ctx.get_tasks_for_approval().first() {
                    self.input = Some(Input::Reject { task_id: task.id.clone(), text: String::new() });
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::tests::task;
    use crate::context::{ProjectContext, TaskStatus};
    use crossterm::event::KeyEvent;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    fn app() -> (tempfile::TempDir, App) {
        let dir = tempfile::tempdir().unwrap();
        let mut ctx =
            ProjectContext::new("t".into(), dir.path().to_path_buf(), 50, 40, false, "ollama:test".into());
        let mut waiting = task("TASK-1", &[]);
        waiting.status = TaskStatus::RequiresHumanApproval { output_summary: "done".into() };
        ctx.tasks = vec![waiting, task("TASK-2", &["TASK-1"])];
        (dir, App::new(Arc::new(Mutex::new(ctx))))
    }

    async fn press(app: &mut App, code: KeyCode) {
        app.handle_event(Event::Key(KeyEvent::new(code, KeyModifiers::NONE))).await.unwrap();
    }

    async fn type_text(app: &mut App, text: &str) {
        for c in text.chars() {
            press(app, KeyCode::Char(c)).await;
        }
    }

    #[tokio::test]
    async fn approve_from_the_hitl_view() {
        let (_dir, mut app) = app();
        press(&mut app, KeyCode::Char('a')).await;
        assert!(!app.context.lock().await.is_task_completed("TASK-1"), "'a' only works in the HITL view");

        press(&mut app, KeyCode::Char('3')).await;
        press(&mut app, KeyCode::Char('a')).await;
        assert!(app.context.lock().await.is_task_completed("TASK-1"));
    }

    #[tokio::test]
    async fn reject_with_typed_feedback() {
        let (_dir, mut app) = app();
        press(&mut app, KeyCode::Char('3')).await;
        press(&mut app, KeyCode::Char('r')).await;
        // 'q' and 'a' are text now, not commands.
        type_text(&mut app, "quit using a tab").await;
        assert!(app.running);
        press(&mut app, KeyCode::Enter).await;

        let ctx = app.context.lock().await;
        let t = ctx.get_task("TASK-1").unwrap();
        assert_eq!(t.status, TaskStatus::Pending);
        assert!(t.feedback.as_deref().unwrap().contains("quit using a tab"));
        assert!(app.input.is_none());
    }

    #[tokio::test]
    async fn escape_cancels_a_rejection() {
        let (_dir, mut app) = app();
        press(&mut app, KeyCode::Char('3')).await;
        press(&mut app, KeyCode::Char('r')).await;
        type_text(&mut app, "never mind").await;
        press(&mut app, KeyCode::Esc).await;
        let ctx = app.context.lock().await;
        assert!(matches!(ctx.get_task("TASK-1").unwrap().status, TaskStatus::RequiresHumanApproval { .. }));
    }

    #[tokio::test]
    async fn menu_edits_brave_steps_and_model() {
        let (_dir, mut app) = app();
        press(&mut app, KeyCode::Char('m')).await;
        press(&mut app, KeyCode::Enter).await;
        assert!(app.context.lock().await.brave_mode);

        press(&mut app, KeyCode::Down).await;
        press(&mut app, KeyCode::Down).await;
        press(&mut app, KeyCode::Right).await;
        assert_eq!(app.context.lock().await.max_steps, 55);
        press(&mut app, KeyCode::Left).await;
        press(&mut app, KeyCode::Left).await;
        assert_eq!(app.context.lock().await.max_steps, 45);

        press(&mut app, KeyCode::Up).await;
        press(&mut app, KeyCode::Enter).await;
        assert_eq!(app.input, Some(Input::Model { text: "ollama:test".into() }));
        for _ in 0.."test".len() {
            press(&mut app, KeyCode::Backspace).await;
        }
        type_text(&mut app, "other").await;
        press(&mut app, KeyCode::Enter).await;
        assert_eq!(app.context.lock().await.model, "ollama:other");

        press(&mut app, KeyCode::Esc).await;
        assert!(!app.show_modal);
    }

    #[tokio::test]
    async fn ctrl_c_quits_even_while_typing() {
        let (_dir, mut app) = app();
        press(&mut app, KeyCode::Char('3')).await;
        press(&mut app, KeyCode::Char('r')).await;
        app.handle_event(Event::Key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)))
            .await
            .unwrap();
        assert!(!app.running);
    }
}
