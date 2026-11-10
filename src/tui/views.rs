use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Gauge, List, ListItem, ListState, Paragraph, Tabs, Wrap},
};

use crate::context::{ProjectContext, ProjectTask, RunPhase, TaskStatus};
use crate::tui::app::{App, Input, View};

fn bold() -> Style {
    Style::default().add_modifier(Modifier::BOLD)
}

fn status_marker(status: &TaskStatus) -> (&'static str, Color) {
    match status {
        TaskStatus::Completed { .. } => ("✅", Color::Green),
        TaskStatus::Failed { .. } => ("❌", Color::Red),
        TaskStatus::InProgress => ("🔄", Color::Blue),
        TaskStatus::Pending => ("⏳", Color::Gray),
        TaskStatus::ReadyForReview { .. } => ("👀", Color::Yellow),
        TaskStatus::Reviewing { .. } => ("🔍", Color::Magenta),
        TaskStatus::RequiresHumanApproval { .. } => ("👤", Color::LightRed),
    }
}

pub fn draw(frame: &mut Frame, app: &mut App, ctx: &ProjectContext) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0), Constraint::Length(1)])
        .split(frame.area());

    let waiting = ctx.get_tasks_for_approval().len();
    let hitl_title = if waiting > 0 { format!("[3] HITL ({waiting})") } else { "[3] HITL".to_string() };
    let titles: Vec<Line> = vec!["[1] Tasks".to_string(), "[2] Logs".to_string(), hitl_title, "[4] Stats".to_string()]
        .into_iter()
        .map(Line::from)
        .collect();

    let phase_color = match ctx.phase {
        RunPhase::Finished if !ctx.succeeded() => Color::Red,
        RunPhase::Finished => Color::Green,
        RunPhase::Error(_) => Color::Red,
        RunPhase::AwaitingApproval | RunPhase::StepLimit => Color::Yellow,
        _ => Color::Cyan,
    };
    let title = Line::from(vec![
        Span::raw(format!(" Clankistrator: {} | ", ctx.project_name)),
        Span::styled(ctx.phase.label(), Style::default().fg(phase_color).add_modifier(Modifier::BOLD)),
        Span::raw(" "),
    ]);

    let tabs = Tabs::new(titles)
        .block(Block::default().borders(Borders::ALL).title(title))
        .select(app.current_view.index())
        .style(Style::default().fg(Color::Cyan))
        .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));
    frame.render_widget(tabs, chunks[0]);

    match app.current_view {
        View::Tasks => draw_tasks(frame, app, ctx, chunks[1]),
        View::Logs => draw_logs(frame, app, chunks[1]),
        View::Hitl => draw_hitl(frame, ctx, chunks[1]),
        View::Stats => draw_stats(frame, ctx, chunks[1]),
    }

    let help = match app.current_view {
        View::Tasks => "↑/↓ select task",
        View::Logs => "PgUp/PgDn scroll, Esc follow",
        View::Hitl => "a approve, r reject with feedback",
        View::Stats => "",
    };
    let footer = Paragraph::new(format!(" 1-4/Tab views | m menu | q quit | {help}"))
        .style(Style::default().fg(Color::DarkGray));
    frame.render_widget(footer, chunks[2]);

    if app.show_modal {
        draw_config_modal(frame, app, ctx);
    }
    if let Some(input) = &app.input {
        draw_input(frame, input);
    }
}

fn draw_config_modal(frame: &mut Frame, app: &App, ctx: &ProjectContext) {
    let block = Block::default()
        .title(" Configuration (m or Esc to close) ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow));

    let area = centered_rect(60, 35, frame.area());
    frame.render_widget(Clear, area);

    let row_style = |row: usize| {
        if app.modal_selection == row {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        }
    };
    let brave = if ctx.brave_mode {
        "[X] Brave mode (tasks are approved automatically)"
    } else {
        "[ ] Brave mode (you approve each task)"
    };
    let max_steps = if ctx.max_steps == 0 { "unlimited".to_string() } else { ctx.max_steps.to_string() };

    let items = vec![
        ListItem::new(Span::styled(brave, row_style(0))),
        ListItem::new(Span::styled(format!("Model:     {}  (Enter to edit)", ctx.model), row_style(1))),
        ListItem::new(Span::styled(format!("Max steps: < {max_steps} >"), row_style(2))),
        ListItem::new(""),
        ListItem::new(Span::styled(
            "↑/↓ select, Enter or ←/→ to change. Changes apply to agents started from now on.",
            Style::default().fg(Color::Gray).add_modifier(Modifier::ITALIC),
        )),
    ];
    frame.render_widget(List::new(items).block(block), area);
}

fn draw_input(frame: &mut Frame, input: &Input) {
    let (title, text) = match input {
        Input::Reject { task_id, text } => (format!(" Why is {task_id} rejected? (Enter to send, Esc to cancel) "), text),
        Input::Model { text } => (" Model, e.g. gpt-4o or ollama:llama3.2 (Enter to apply, Esc to cancel) ".to_string(), text),
    };
    let area = centered_rect(70, 20, frame.area());
    // A fixed-height box, kept inside the screen however small the terminal is.
    let screen = frame.area();
    let height = 5.min(screen.height);
    let y = area.y.min(screen.bottom().saturating_sub(height));
    let area = Rect { y, height, ..area };
    frame.render_widget(Clear, area);
    let paragraph = Paragraph::new(format!("{text}█"))
        .wrap(Wrap { trim: false })
        .block(Block::default().borders(Borders::ALL).title(title).border_style(Style::default().fg(Color::Yellow)));
    frame.render_widget(paragraph, area);
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

fn field(label: &'static str, value: String) -> Line<'static> {
    Line::from(vec![Span::styled(label, bold()), Span::raw(value)])
}

fn section(lines: &mut Vec<Line<'static>>, heading: &'static str, body: &str) {
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(heading, bold())));
    lines.extend(body.lines().map(|l| Line::from(l.to_string())));
}

fn task_details(task: &ProjectTask) -> Vec<Line<'static>> {
    let mut lines = vec![
        field("Task:    ", format!("{} - {}", task.id, task.title)),
        field("Role:    ", format!("{:?}", task.assigned_role)),
        field("Status:  ", task.status.label().to_string()),
        field("Retries: ", task.retries.to_string()),
        field(
            "Depends: ",
            if task.dependencies.is_empty() { "-".into() } else { task.dependencies.join(", ") },
        ),
    ];
    section(&mut lines, "Description", &task.description);
    match &task.status {
        TaskStatus::Completed { output_summary }
        | TaskStatus::ReadyForReview { output_summary }
        | TaskStatus::Reviewing { output_summary }
        | TaskStatus::RequiresHumanApproval { output_summary } => {
            section(&mut lines, "Worker summary", output_summary);
        }
        TaskStatus::Failed { error } => section(&mut lines, "Error", error),
        TaskStatus::Pending | TaskStatus::InProgress => {}
    }
    if !task.files.is_empty() {
        section(&mut lines, "Files", &task.files.join("\n"));
    }
    if !task.acceptance_criteria.is_empty() {
        let criteria: Vec<String> = task.acceptance_criteria.iter().map(|c| format!("- {c}")).collect();
        section(&mut lines, "Acceptance criteria", &criteria.join("\n"));
    }
    if !task.review_notes.is_empty() {
        section(&mut lines, "The reviewer could not verify (yours to check)", &task.review_notes.join("\n"));
    }
    if let Some(feedback) = &task.feedback {
        section(&mut lines, "Feedback for the next attempt", feedback);
    }
    lines
}

fn draw_tasks(frame: &mut Frame, app: &mut App, ctx: &ProjectContext, area: Rect) {
    if ctx.tasks.is_empty() {
        let text = match &ctx.phase {
            RunPhase::Error(e) => format!("No tasks. {e}"),
            _ => "Planning: the model is breaking the design document into tasks...".to_string(),
        };
        let p = Paragraph::new(text)
            .wrap(Wrap { trim: true })
            .block(Block::default().borders(Borders::ALL).title("Tasks"));
        frame.render_widget(p, area);
        return;
    }
    app.selected_task = app.selected_task.min(ctx.tasks.len() - 1);

    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(area);

    let items: Vec<ListItem> = ctx
        .tasks
        .iter()
        .map(|t| {
            let (marker, color) = status_marker(&t.status);
            ListItem::new(Line::from(vec![
                Span::styled(format!(" {:<8} ", t.id), Style::default().fg(Color::Cyan)),
                Span::styled(marker, Style::default().fg(color)),
                Span::raw(format!(" {}", t.title)),
            ]))
        })
        .collect();

    let title = format!("Tasks ({}/{} completed)", ctx.count_completed(), ctx.tasks.len());
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(title))
        .highlight_style(Style::default().bg(Color::DarkGray).add_modifier(Modifier::BOLD));
    let mut state = ListState::default().with_selected(Some(app.selected_task));
    frame.render_stateful_widget(list, columns[0], &mut state);

    let details = Paragraph::new(task_details(&ctx.tasks[app.selected_task]))
        .wrap(Wrap { trim: false })
        .block(Block::default().borders(Borders::ALL).title("Details"));
    frame.render_widget(details, columns[1]);
}

fn draw_logs(frame: &mut Frame, app: &mut App, area: Rect) {
    let tui_w = tui_logger::TuiLoggerWidget::default()
        .block(Block::default().title("Logs").borders(Borders::ALL))
        .style_error(Style::default().fg(Color::Red))
        .style_debug(Style::default().fg(Color::Green))
        .style_warn(Style::default().fg(Color::Yellow))
        .style_trace(Style::default().fg(Color::Gray))
        .style_info(Style::default().fg(Color::Blue))
        .output_separator('|')
        .output_timestamp(Some("%H:%M:%S".to_string()))
        .output_level(Some(tui_logger::TuiLoggerLevelOutput::Abbreviated))
        .output_target(false)
        .output_file(false)
        .output_line(false)
        .state(&app.log_state);

    frame.render_widget(tui_w, area);
}

fn draw_hitl(frame: &mut Frame, ctx: &ProjectContext, area: Rect) {
    let pending = ctx.get_tasks_for_approval();
    let Some(task) = pending.first() else {
        let text = if ctx.brave_mode {
            "Brave mode is on: tasks that pass review are approved automatically."
        } else {
            "No tasks are waiting for your approval."
        };
        let p = Paragraph::new(text)
            .block(Block::default().borders(Borders::ALL).title("Human-in-the-Loop"))
            .style(Style::default().fg(Color::Gray));
        frame.render_widget(p, area);
        return;
    };

    let mut lines = task_details(task);
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("a", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        Span::raw(" approve    "),
        Span::styled("r", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
        Span::raw(" reject and send feedback to the worker"),
    ]));

    let title = format!("Approval required: {} ({} waiting)", task.id, pending.len());
    let p = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(title))
        .wrap(Wrap { trim: false });
    frame.render_widget(p, area);
}

fn draw_stats(frame: &mut Frame, ctx: &ProjectContext, area: Rect) {
    let total = ctx.tasks.len();
    let completed = ctx.count_completed();
    let percent = if total > 0 { (completed * 100 / total) as u16 } else { 0 };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0)])
        .split(area);

    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title("Overall progress"))
        .gauge_style(Style::default().fg(Color::Green))
        .percent(percent);
    frame.render_widget(gauge, chunks[0]);

    let max_steps = if ctx.max_steps == 0 { "unlimited".to_string() } else { ctx.max_steps.to_string() };
    let mut lines = vec![
        field("Tasks:         ", format!("{completed} completed, {} failed, {total} total", ctx.count_failed())),
        field("Steps:         ", format!("{} of {max_steps}", ctx.steps_taken)),
        field("Model:         ", ctx.model.clone()),
        field("Brave mode:    ", if ctx.brave_mode { "on".into() } else { "off".into() }),
        field("Output:        ", ctx.working_directory.display().to_string()),
        Line::from(""),
        field("Model calls:   ", ctx.llm_calls.to_string()),
        field(
            "Input tokens:  ",
            if ctx.cached_input_tokens > 0 {
                format!("{} ({} read from the provider's cache)", ctx.input_tokens, ctx.cached_input_tokens)
            } else {
                ctx.input_tokens.to_string()
            },
        ),
        field("Output tokens: ", ctx.output_tokens.to_string()),
        Line::from(Span::styled(
            "Token counts are the totals reported by the provider for finished agent runs.",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(""),
    ];
    let design_check = if !ctx.design_check_enabled {
        "switched off".to_string()
    } else if let Some(error) = &ctx.design_check_error {
        format!("could not be completed: {error}")
    } else if let Some(findings) = &ctx.design_check {
        format!(
            "{} of {} requirements met, {} unmet, {} could not be verified ({} fix round(s) used)",
            findings.len() - ctx.unmet_requirements().len() - ctx.unverified_requirements().len(),
            findings.len(),
            ctx.unmet_requirements().len(),
            ctx.unverified_requirements().len(),
            ctx.fix_rounds_done
        )
    } else {
        "runs when all tasks are complete".to_string()
    };
    lines.push(field("Design check:  ", design_check));
    for (heading, findings, color) in [
        ("Unmet", ctx.unmet_requirements(), Color::Red),
        ("Could not be verified (yours to check)", ctx.unverified_requirements(), Color::Yellow),
    ] {
        if findings.is_empty() {
            continue;
        }
        lines.push(Line::from(Span::styled(heading, Style::default().fg(color).add_modifier(Modifier::BOLD))));
        for finding in findings {
            lines.extend(finding.line().lines().map(|l| Line::from(l.to_string())));
        }
    }
    let p = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .block(Block::default().borders(Borders::ALL).title("Run"));
    frame.render_widget(p, chunks[1]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::tests::task;
    use ratatui::{Terminal, backend::TestBackend};
    use std::sync::Arc;
    use tokio::sync::Mutex;

    fn render(app: &mut App, ctx: &ProjectContext) -> String {
        let mut terminal = Terminal::new(TestBackend::new(110, 30)).unwrap();
        terminal.draw(|f| draw(f, app, ctx)).unwrap();
        let buffer = terminal.backend().buffer().clone();
        buffer
            .content()
            .chunks(buffer.area.width as usize)
            .map(|row| row.iter().map(|cell| cell.symbol()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn every_view_renders() {
        let mut ctx = ProjectContext::new("calc".into(), "/tmp/out".into(), 50, 40, false, "gpt-4o".into());
        let mut app = App::new(Arc::new(Mutex::new(ctx.clone())));

        // Before planning has produced anything.
        assert!(render(&mut app, &ctx).contains("Planning"));

        let mut waiting = task("TASK-1", &[]);
        waiting.status = TaskStatus::RequiresHumanApproval { output_summary: "Wrote the parser".into() };
        waiting.files = vec!["src/parser.rs".into()];
        waiting.review_notes = vec!["- The cursor blinks\n  needs a terminal".into()];
        let mut failed = task("TASK-2", &["TASK-1"]);
        failed.status = TaskStatus::Failed { error: "cargo build exploded".into() };
        ctx.tasks = vec![waiting, failed];
        ctx.phase = RunPhase::AwaitingApproval;
        app.selected_task = 99; // stale selection must be clamped, not panic

        let tasks = render(&mut app, &ctx);
        assert!(tasks.contains("Waiting for your approval"));
        assert!(tasks.contains("[3] HITL (1)"));
        assert!(tasks.contains("cargo build exploded"), "details of the selected (failed) task:\n{tasks}");

        app.current_view = View::Hitl;
        let hitl = render(&mut app, &ctx);
        assert!(hitl.contains("Wrote the parser"));
        assert!(hitl.contains("src/parser.rs"));
        assert!(hitl.contains("The cursor blinks"), "what the reviewer could not verify is shown to the approver");
        assert!(hitl.contains("reject"));

        app.current_view = View::Stats;
        let stats = render(&mut app, &ctx);
        assert!(stats.contains("0 completed, 1 failed, 2 total"));
        assert!(stats.contains("Input tokens"));
        assert!(stats.contains("runs when all tasks are complete"));

        ctx.design_check = Some(vec![
            crate::context::Finding {
                requirement: "Enter evaluates the expression".into(),
                status: crate::context::FindingStatus::Unmet,
                evidence: "only '=' is handled".into(),
            },
            crate::context::Finding {
                requirement: "Borders are cyan".into(),
                status: crate::context::FindingStatus::CannotVerify,
                evidence: "needs a terminal".into(),
            },
        ]);
        let stats = render(&mut app, &ctx);
        assert!(stats.contains("0 of 2 requirements met, 1 unmet, 1 could not be verified"), "{stats}");
        assert!(stats.contains("only '=' is handled"));
        assert!(stats.contains("Borders are cyan"));

        app.current_view = View::Logs;
        render(&mut app, &ctx);

        app.show_modal = true;
        assert!(render(&mut app, &ctx).contains("Brave mode"));
        app.input = Some(Input::Reject { task_id: "TASK-1".into(), text: "needs tests".into() });
        assert!(render(&mut app, &ctx).contains("needs tests"));
    }

    #[test]
    fn tiny_terminals_do_not_panic() {
        let ctx = ProjectContext::new("calc".into(), "/tmp/out".into(), 50, 40, false, "gpt-4o".into());
        let mut app = App::new(Arc::new(Mutex::new(ctx.clone())));
        app.show_modal = true;
        app.input = Some(Input::Model { text: "x".into() });
        for (w, h) in [(1, 1), (10, 3), (20, 5)] {
            let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
            terminal.draw(|f| draw(f, &mut app, &ctx)).unwrap();
        }
    }
}
