use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Tabs, Gauge, Clear},
    Frame,
};
use crate::tui::app::{App, View};
use crate::context::{TaskStatus, ProjectContext};

pub fn draw(frame: &mut Frame, app: &mut App, ctx: &mut ProjectContext) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0)])
        .split(frame.area());

    let titles: Vec<Line> = vec!["[1] Tasks", "[2] Logs", "[3] HITL", "[4] Stats"]
        .into_iter()
        .map(Line::from)
        .collect();

    let index = match app.current_view {
        View::Tasks => 0,
        View::Logs => 1,
        View::HITL => 2,
        View::Stats => 3,
    };

    let tabs = Tabs::new(titles)
        .block(Block::default().borders(Borders::ALL).title("Clankistrator Dashboard"))
        .select(index)
        .style(Style::default().fg(Color::Cyan))
        .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));

    frame.render_widget(tabs, chunks[0]);

    match app.current_view {
        View::Tasks => draw_tasks(frame, app, ctx, chunks[1]),
        View::Logs => draw_logs(frame, app, chunks[1]),
        View::HITL => draw_hitl(frame, app, ctx, chunks[1]),
        View::Stats => draw_stats(frame, app, ctx, chunks[1]),
    }

    if app.show_modal {
        draw_config_modal(frame, app, ctx);
    }
}

fn draw_config_modal(frame: &mut Frame, app: &mut App, ctx: &ProjectContext) {
    let block = Block::default()
        .title(" Configuration Menu (Press 'm' or ESC to close) ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow));
    
    let area = centered_rect(60, 20, frame.area());
    frame.render_widget(Clear, area); // This clears the background
    
    let brave_status = if ctx.brave_mode {
        "[X] Brave Mode (Autonomous Execution)"
    } else {
        "[ ] Brave Mode (Human-in-the-Loop)"
    };

    let items = vec![
        ListItem::new(Line::from(vec![
            Span::styled(brave_status, if app.modal_selection == 0 {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            })
        ])),
    ];

    let list = List::new(items)
        .block(block)
        .highlight_style(Style::default().bg(Color::DarkGray));
    
    frame.render_widget(list, area);
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

fn draw_tasks(frame: &mut Frame, _app: &mut App, ctx: &mut ProjectContext, area: Rect) {
    let tasks: Vec<ListItem> = ctx.tasks.iter().map(|t| {
        let status_text = match &t.status {
            TaskStatus::Completed { .. } => ("✅", Color::Green),
            TaskStatus::Failed { .. } => ("❌", Color::Red),
            TaskStatus::InProgress(_) => ("🔄", Color::Blue),
            TaskStatus::Pending => ("⏳", Color::Gray),
            TaskStatus::ReadyForReview { .. } => ("👀", Color::Yellow),
            TaskStatus::Reviewing(_) => ("🔍", Color::Magenta),
            TaskStatus::RequiresHumanApproval { .. } => ("👤", Color::LightRed),
        };

        ListItem::new(vec![
            Line::from(vec![
                Span::styled(format!(" {:<8} ", t.id), Style::default().fg(Color::Cyan)),
                Span::styled(status_text.0, Style::default().fg(status_text.1)),
                Span::raw(format!(" - {}", t.title)),
            ]),
        ])
    }).collect();

    let tasks_list = List::new(tasks)
        .block(Block::default().borders(Borders::ALL).title("Tasks"))
        .highlight_style(Style::default().bg(Color::DarkGray).add_modifier(Modifier::BOLD));
    
    frame.render_widget(tasks_list, area);
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
        .output_target(true)
        .output_file(false)
        .output_line(false)
        .state(&mut app.log_state);
    
    frame.render_widget(tui_w, area);
}

fn draw_hitl(frame: &mut Frame, _app: &mut App, ctx: &mut ProjectContext, area: Rect) {
    let pending_approval = ctx.get_tasks_for_approval();

    if pending_approval.is_empty() {
        let p = Paragraph::new("No tasks pending human approval.")
            .block(Block::default().borders(Borders::ALL).title("Human-in-the-Loop"))
            .style(Style::default().fg(Color::Gray));
        frame.render_widget(p, area);
    } else {
        let task = pending_approval[0];
        let summary = if let TaskStatus::RequiresHumanApproval { output_summary, .. } = &task.status {
            output_summary
        } else {
            "N/A"
        };

        let text = vec![
            Line::from(vec![Span::styled("Task ID: ", Style::default().add_modifier(Modifier::BOLD)), Span::raw(&task.id)]),
            Line::from(vec![Span::styled("Title:   ", Style::default().add_modifier(Modifier::BOLD)), Span::raw(&task.title)]),
            Line::from(""),
            Line::from(Span::styled("Description:", Style::default().add_modifier(Modifier::BOLD))),
            Line::from(task.description.as_str()),
            Line::from(""),
            Line::from(Span::styled("Implementation Summary:", Style::default().add_modifier(Modifier::BOLD))),
            Line::from(summary.as_ref()),
            Line::from(""),
            Line::from(vec![
                Span::styled("Press 'a' to APPROVE this task", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            ]),
        ];

        let p = Paragraph::new(text)
            .block(Block::default().borders(Borders::ALL).title(format!("Approval Required: {}", task.id)))
            .wrap(ratatui::widgets::Wrap { trim: true });
        frame.render_widget(p, area);
    }
}

fn draw_stats(frame: &mut Frame, _app: &mut App, ctx: &mut ProjectContext, area: Rect) {
    let total_tasks = ctx.tasks.len();
    let completed_tasks = ctx.tasks.iter().filter(|t| matches!(t.status, TaskStatus::Completed { .. })).count();
    
    let progress = if total_tasks > 0 {
        completed_tasks as f64 / total_tasks as f64
    } else {
        0.0
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(0)
        ])
        .split(area);

    let progress_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title("Overall Progress"))
        .gauge_style(Style::default().fg(Color::Green))
        .percent((progress * 100.0) as u16);
    frame.render_widget(progress_gauge, chunks[0]);

    let token_usage = Paragraph::new(format!("Token Usage: {}", ctx.token_usage))
        .block(Block::default().borders(Borders::ALL).title("Resources"));
    frame.render_widget(token_usage, chunks[1]);

    let cost = Paragraph::new(format!("Estimated Cost: ${:.4}", ctx.total_cost))
        .block(Block::default().borders(Borders::ALL).title("Resources"));
    frame.render_widget(cost, chunks[2]);

    let max_steps_str = if ctx.max_steps == 0 { "Unlimited".to_string() } else { ctx.max_steps.to_string() };
    let steps_info = Paragraph::new(format!("Max Steps: {}", max_steps_str))
        .block(Block::default().borders(Borders::ALL).title("Configuration"));
    frame.render_widget(steps_info, chunks[3]);
}
