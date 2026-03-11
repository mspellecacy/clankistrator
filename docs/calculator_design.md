# TUI Calculator: A simple Terminal Calculator

## 1. Vision & Core Objectives
The primary goal is to provide a fast, keyboard-driven, and intuitive terminal-based calculator. It should support basic arithmetic operations, maintain a calculation history, and have a clean, modern terminal interface.

- **Fast & Responsive**: Immediate calculation results and fluid TUI updates.
- **Keyboard First**: Optimized for typing expressions directly with standard keybindings.
- **History Tracking**: Keep track of previous results for quick reference.
- **Type-Safe & Robust**: Written in Rust, ensuring reliability and graceful handling of edge cases like division by zero.

## 2. Features
- **Standard Arithmetic**: Addition (+), Subtraction (-), Multiplication (*), Division (/).
- **Expression Parsing**: Support for full expression evaluation (e.g., "15 + 3 * (10 / 2)").
- **Clear & Backspace**: Functions to reset the current input or delete the last character.
- **History Pane**: A scrollable view of past calculations and their results.
- **Error Feedback**: Clear error messages for syntax errors or invalid operations.
- **Persistence**: (Optional) Save history to a local file for subsequent sessions.

## 3. Infrastructure
- **Language**: Rust (Edition 2024).
- **TUI Framework**: `ratatui` for UI components.
- **Terminal Backend**: `crossterm` for terminal interaction and event handling.
- **Expression Evaluation**: `evalexpr` crate for flexible mathematical expression parsing.
- **State Management**: Simple application state struct with immediate-mode UI rendering.

## 4. DataSchema (Proposed Rust Types)

```rust
pub enum AppStatus {
    Ready,
    Error(String),
}

pub struct Calculation {
    pub expression: String,
    pub result: String,
}

pub struct CalculatorApp {
    pub current_input: String,
    pub history: Vec<Calculation>,
    pub status: AppStatus,
    pub scroll_offset: usize,
}
```

## 5. UI Layout
The interface is structured into three primary blocks:
1.  **Main Display (Center)**: A large block showing the current input string and the result of the calculation.
2.  **History Panel (Right/Bottom)**: A scrollable list of previous `Calculation` entries.
3.  **Status/Keybindings (Footer)**: Displays available keys (e.g., `[Enter] Calc`, `[Esc] Exit`, `[C] Clear`).

## 6. Implementation Roadmap

### Phase 1: Foundations
- [ ] Initialize project structure and `Cargo.toml` with `ratatui` and `evalexpr`.
- [ ] Create core `CalculatorApp` struct and basic event loop.
- [ ] Implement keyboard input handling for numbers and operators.

### Phase 2: Logic & UI
- [ ] Integrate `evalexpr` for real-time expression evaluation.
- [ ] Design and implement the main display and status footer components.
- [ ] Implement the history logic and display panel.

### Phase 3: Refinement
- [ ] Add comprehensive error handling and visual error states.
- [ ] Implement scrolling for the history panel.
- [ ] Fine-tune UI aesthetics (colors, borders, alignment).
