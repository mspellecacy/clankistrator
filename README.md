# Clankistrator: AI Orchestration Framework

Clankistrator is a high-performance, type-safe Rust framework for multi-agent AI orchestration. It decomposes high-level design documents (Markdown) into actionable project tasks, which are then assigned to and executed by specialized AI agents.

## 🚀 Vision & Core Objectives

The primary goal is to provide a **"Chief Architect"** agent that can read a comprehensive project design, understand its technical requirements, and orchestrate a fleet of **"Specialized Worker"** agents to implement the project from start to finish.

- **Rust-Centric**: Leverage Rust's safety and concurrency for reliable agent execution.
- **Agentic Workflows**: Support hand-offs, reviews, and iterative refinement.
- **Provider Agnostic**: Seamlessly switch between OpenAI, Anthropic, Gemini, or local models (Ollama).
- **Persistent State**: Maintain a clear "Project Context" representing the project's current state.

## 🛠️ Key Features

- **Design-First Orchestration**: Automatically parse Markdown design documents into task graphs with dependency resolution.
- **Specialized Worker Agents**: Dedicated personas for Infrastructure, Backend, Frontend, and QA.
- **Rich Tooling**: Agents have access to the file system, command execution (e.g., `cargo test`), and RAG-based document querying.
- **Review & Approval Loop**: Multi-stage execution with automated QA reviews and Human-in-the-Loop (HITL) approval.
- **Environment Management**: Securely manage API keys and local settings via `.env` and `.env.local` files.

## 🏗️ System Architecture

### 1. The Orchestrator (Master Agent)
The central brain that parses requirements, generates a `TaskGraph` (DAG), assigns workers, and monitors progress.

### 2. Worker Agents (Specialized Personas)
- **Infrastructure Architect**: Manages file systems, Docker, and environment setup.
- **Backend Architect**: Handles database schemas, API design, and core logic.
- **Frontend/Design Architect**: Focuses on UI structure and styling logic.
- **QA Reviewer**: Validates task outputs, finds edge cases, and provides feedback.

### 3. The Project Context (State)
A shared state containing the design source of truth, task list with statuses (Pending, InProgress, ReadyForReview, RequiresHumanApproval, Completed, Failed), and file metadata.

## 🏁 Getting Started

### Prerequisites
- [Rust](https://rustup.rs/) (2024 Edition)
- API Keys for your preferred LLM provider (OpenAI, Anthropic, or Gemini)

### Installation
Clone the repository and build the project:
```bash
git clone https://github.com/your-repo/clankistrator.git
cd clankistrator
cargo build --release
```

### Configuration
Create a `.env` or `.env.local` file in the project root:
```bash
cp .env.example .env
```
Fill in your API keys:
```env
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GEMINI_API_KEY=your_gemini_key_here
```

### Usage
Run Clankistrator by providing a design document and an output directory:
```bash
cargo run -- --design docs/calculator_design.md --output ./my-new-project
```

## 🗺️ Implementation Roadmap

### Phase 1: Core Foundation ✅
- Project initialization and basic data structures.
- Markdown parsing for design documents.
- Environment management and basic LLM connectivity.

### Phase 2: Agent Orchestration ✅
- Orchestrator logic and worker abstractions.
- Task management and dependency resolution.
- Dynamic agent spawning.

### Phase 3: Tooling & Execution ✅
- File system and command execution tools.
- Automated review loop between workers and QA.

### Phase 4: Refinement & UI ✅
- Rich logging with `tracing`.
- RAG support for large document querying.
- Human-in-the-Loop (HITL) approval phase.

## 📂 Project Structure
```text
clankistrator/
├── src/
│   ├── main.rs                 # CLI entry point
│   ├── context.rs              # ProjectContext & Task types
│   ├── orchestrator/           # Orchestrator agent logic
│   ├── agents/                 # Specialized worker implementations
│   ├── tools/                  # Shared tool implementations
│   └── parser/                 # Design doc parsing logic
├── docs/
│   ├── calculator_design.md    # TUI Calculator design example
│   └── clankistrator_design.md # Framework design document
└── tests/                      # Integration and unit tests
```

## ⚖️ License
This project is currently private.
