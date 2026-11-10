# Clankistrator

Clankistrator turns a Markdown design document into a working project. A planner model breaks the design into a small task graph, worker agents implement the tasks with file and command tools, a reviewer agent inspects the result, and you approve each task (or let it run unattended).

It is a Rust binary built on [rig](https://github.com/0xPlaygrounds/rig).

## How a run works

1. **Plan.** The model decomposes the design into at most 5 tasks with roles, dependencies and concrete acceptance criteria copied from the design. The plan is validated (unique ids, known dependencies, no cycles) and retried once if it is unusable.
2. **Execute.** Every task whose dependencies are complete goes to a worker agent (Infrastructure, Backend, Frontend or QA persona). Up to four agents run at once. Workers see the whole design document, summaries of completed tasks, and any feedback from an earlier attempt.
3. **Review.** A QA reviewer reads the files, builds and tests the project, and answers a list of items one by one: a fixed baseline (it builds, the logic has passing tests, the files are in the project root) plus the task's acceptance criteria. Each answer is `met`, `unmet` or `cannot verify`, with evidence. The reviewer is never asked for an overall verdict; Clankistrator rejects the task if any item is unmet and sends the unmet items and their evidence back to the worker. A `met` only counts if the reviewer quotes a line of code that Clankistrator then finds in the named file; an answer that fails this is sent back once for correction and otherwise counted as unmet. Items that can be decided by reading code cannot be answered `cannot verify`.
4. **Approve.** You approve or reject each reviewed task, in the TUI or on stdin. Items the reviewer could not verify are shown to you at this point. `--brave` approves automatically.
5. **Design check.** When all tasks are complete, the design document is turned into a requirement checklist (without looking at the code), and the whole project is verified against it the same way. Unmet requirements become one fix task, up to `--fix-rounds` times. If requirements are still unmet at the end, the run says so and exits non-zero.

To judge a project that already exists, without running any tasks:

```bash
cargo run --release -- --check --design docs/sumcli_design.md --output ../sumcli
```

A provider call that fails (rate limit, overload, network) is retried twice with a growing delay before it counts against the task. A task gets three retries, counted across failed attempts and failed reviews. After that it is marked failed, and every task that depends on it is marked failed too, so a run always ends. State is saved to `<output>/.clankistrator/state.json` after every task result, and `--resume` continues from it.

## Safety

Agents write files and run programs on your machine. Know what the guard rails are and are not:

- File tools are confined to the output directory. Absolute paths outside it, `..` escapes and symlinks leading out are refused.
- Commands run with the output directory as the working directory, without a shell, with no stdin, with your provider API keys removed from their environment, and are killed (with their child processes) after `--command-timeout` seconds. Process-group cleanup is implemented for Unix only.
- Choose an output directory that is not inside another project (the examples use `../sumcli`, next to this repository). Build tools and version control search parent directories for their project files. Clankistrator refuses a `cargo` command that would fall through to a `Cargo.toml` outside the output directory, stops `git` from seeing an enclosing repository, and warns at startup when the output directory is nested in a project, but other tools (npm, go, pytest) are not guarded.
- **Commands are not sandboxed.** They are ordinary processes running as you, and a program can touch anything you can. Building generated code also runs its build scripts. Use a container or a throwaway account if that matters to you.
- Without `--brave` nothing counts as done until you approve it, but the work has already been written and run by then. Approval gates progress, not execution.

## Getting started

Requirements: [Rust](https://rustup.rs/) 1.88 or newer and an API key for a provider, or a local Ollama.

```bash
git clone https://github.com/mspellecacy/clankistrator.git
cd clankistrator
cargo build --release
cp .env.example .env    # then put your key in it
```

Start with the smallest bundled example, which keeps a first run cheap:

```bash
cargo run --release -- --design docs/sumcli_design.md --output ../sumcli
```

This opens the TUI. For plain log output, add `--headless`; for an unattended run, add `--brave` as well:

```bash
cargo run --release -- --design docs/sumcli_design.md --output ../sumcli --headless --brave
```

`docs/calculator_design.md` (a terminal calculator) and `docs/spinning_cube_design.md` are larger examples.

A run makes real, billed model calls: one for planning, then at least one worker run and one review per task, each of which can be many turns.

### What a run costs

An agent sends its whole conversation again on every turn, so input tokens dominate: one run of the calculator example, before caching was added, took about 700,000 input tokens and 43,000 output tokens. Prompt caching is what keeps that affordable. With Anthropic models Clankistrator asks for it (turn it off with `--no-prompt-cache`); OpenAI applies it on its own. Tokens read from the cache are billed at a fraction of the normal input price, and with Anthropic the tokens written to the cache cost somewhat more than normal input, so caching pays off from an agent's second turn on.

The summary reports input tokens as one total, with the part read from the cache in brackets, in the form `12000 in (9000 from cache) / 800 out`. The total includes cached reads and, for Anthropic, cache writes. There is no cost estimate in currency; multiply by your provider's prices.

## Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `-d, --design <FILE>` | `design.md` | Markdown design document |
| `-o, --output <DIR>` | `$WORKING_DIRECTORY`, else `output` | Directory the agents work in |
| `-m, --model <MODEL>` | `$DEFAULT_MODEL`, else `gpt-4o` | Model, see below |
| `--headless` | off | Log output instead of the TUI; approvals on stdin |
| `-b, --brave` | off | Approve reviewed tasks automatically |
| `-S, --max-steps <N>` | 50 | Orchestration passes that ran or reviewed a task; 0 is unlimited. Waiting for approval costs nothing |
| `--max-turns <N>` | 40 | Model turns (tool-call rounds) per worker attempt |
| `--command-timeout <SECS>` | 300 | Time limit for a command started by an agent |
| `--review-model <MODEL>` | same as `--model` | Model for reviews and the design check. A stronger model here is the cheapest way to a stricter gate |
| `--fix-rounds <N>` | 2 | How many times unmet design requirements are turned into a fix task |
| `--skip-design-check` | off | Do not check the finished project against the design |
| `--no-prompt-cache` | off | Do not ask Anthropic to cache prompts between an agent's turns |
| `--check` | off | Only check the project in `--output` against `--design` and print the findings. Exits 1 if anything is unmet |
| `--resume` | off | Continue the run saved in the output directory. Interrupted and failed tasks are queued again |
| `--probe <MODEL>...` | | Test the given models (one request and one tool call each) instead of running |
| `--mcp-server` | off | Run as an MCP server instead (see below) |

Headless exit codes: 0 if every task completed and the design check found nothing unmet; 1 if a task failed, a design requirement is unmet, the design check could not be completed, or the run could not start; 2 if the step limit stopped it; 130 on Ctrl-C.

Every worker and reviewer reply is kept in `<output>/.clankistrator/transcripts/`, so you can see afterwards why a review passed or failed.

### Models and providers

A model is named as `prefix:model`. For three providers the prefix can be left out: names starting with `claude` go to Anthropic, names starting with `gemini` go to Gemini, and any other bare name goes to OpenAI. The rest are only ever chosen by prefix.

| Provider | Prefix | Needs | Run live |
| --- | --- | --- | --- |
| OpenAI | `openai:` (optional) | `OPENAI_API_KEY` | yes: probe and full runs |
| Anthropic | `anthropic:` (optional) | `ANTHROPIC_API_KEY` | yes: probe and full runs |
| Gemini | `gemini:` (optional) | `GEMINI_API_KEY` | no |
| Ollama | `ollama:` | a running server; `OLLAMA_BASE_URL` if it is not on `http://localhost:11434` | no |
| OpenRouter | `openrouter:` | `OPENROUTER_API_KEY` | no |
| OpenAI-compatible server | `compatible:` | `OPENAI_COMPATIBLE_BASE_URL` (for example `http://localhost:1234/v1`); `OPENAI_COMPATIBLE_API_KEY` if the server wants one | no |

"Run live" says whether that path has been exercised against the real service by the maintainers. The rows marked "no" compile and are wired up the same way, but nobody has seen them work; probe before you trust them.

Examples: `--model gpt-4o`, `--model ollama:qwen2.5-coder:14b`, `--model openrouter:vendor/model-name`, `--model compatible:whatever-your-server-calls-it`. OpenRouter and OpenAI-compatible servers are how to reach providers that have no row of their own.

Workers and reviewers can use different models, from different providers: `--model` builds, `--review-model` judges. A cheaper model that builds with a stronger one that reviews is a reasonable combination.

#### Checking that a model works

```bash
cargo run --release -- --probe gpt-4o claude-sonnet-5-5 ollama:llama3.2
```

For each model, `--probe` makes one plain request and then one tool call, in which the model has to call a tool with a list argument and repeat a code that only the tool knows. It uses no files and runs no commands, and costs a few hundred tokens per model. A model that fails the tool call cannot be used, because workers and reviewers do everything through tools. The exit code is 0 only if every model passed.

Every run also starts with one small request to each model it will use, so a mistyped model name or a dead key stops the run before any work is done.

rig's own log lines are held at `warn` because at `info` it prints every tool call with its full arguments and result; set `RUST_LOG=info` to see them.

Settings are read from the environment, `.env.local` and `.env` (in that order of precedence): the variables in the table above, `DEFAULT_MODEL`, `WORKING_DIRECTORY` and `LOG_LEVEL` (`RUST_LOG` overrides it).

## The TUI

| Key | Action |
| --- | --- |
| `1`-`4`, `Tab` | Switch between Tasks, Logs, HITL and Stats |
| `↑` `↓` | Select a task; its description, summary, error and feedback show on the right |
| `a` / `r` (HITL) | Approve, or reject with typed feedback for the worker |
| `m` | Menu: brave mode, model, max steps. Changes apply to agents started afterwards |
| `PgUp` `PgDn` `Esc` (Logs) | Scroll, return to following |
| `q`, `Ctrl-C` | Quit. Progress is saved; continue with `--resume` |

Stats shows the token counts reported by the provider, including how many input tokens came from its prompt cache, and the result of the design check.

## MCP server

`clankistrator --mcp-server` speaks MCP over stdio and offers three tools:

- `parse_markdown_design`: split a Markdown file into sections.
- `plan_tasks`: produce the validated task graph for a design document. This calls the model.
- `project_status`: read the saved state of a run from its output directory.

The server does not execute tasks, run commands or write files.

## Writing a design document

Any Markdown works; the whole file is given to the planner and to every agent, so keep it to what matters. The examples in `docs/` are a reasonable shape: goals, features, technology choices, data types, and a roadmap. Name the language and libraries explicitly. Interactive programs are fine as a target, but agents can only verify them through builds and tests, so ask for testable logic.

## Limitations

- How strict the gate is depends on the reviewing model. On the bundled calculator example, one model as reviewer passed a program that never set up the terminal and did not handle Enter, while a stronger one, given the same code and the same checks, reported both with the lines to prove it. If results look too good, set `--review-model` to the strongest model you have. The structure forces it to answer item by item with evidence, but a weak model can still call something met that is not. Agents also cannot run an interactive program, so anything that only shows when you use it (a TUI that draws wrongly, say) is caught only if the reviewer spots it in the code; such items are reported as "could not be verified" and are yours to check. Run the result yourself before you rely on it.
- Agents share one working directory with no locking. Tasks that run in parallel can overwrite each other's files. The reviewer usually catches the resulting breakage, but a plan with fewer parallel tasks is more reliable.
- The reviewer has no write tool, but it can run commands, and a command can modify files.
- Token counts only include agent runs that finished; a run that errored or hit its turn limit is not counted.
- The test suite drives the orchestrator with scripted model replies; it makes no provider calls. Which providers have been run against the real services is listed in the provider table. A run that mixes providers (one building, another reviewing) has been done only as a standalone `--check`, not as a full run.
- Changing the model in the TUI affects the next pass, not agents already running.

## Development

```bash
cargo test
cargo clippy --all-targets
```

```text
src/
├── main.rs            CLI, logging setup, headless and TUI drivers
├── context.rs         Tasks, statuses, graph validation, saved state
├── llm.rs             Provider selection and the single call into rig
├── agents/            Worker and reviewer prompts, reply parsing
├── orchestrator/      Planning, the run loop, system prompts
├── tools/             Sandboxed file tools and command execution
├── tui/               Terminal UI
├── parser/            Markdown section parser (used by the MCP server)
└── mcp.rs             MCP server
docs/                  Example design documents
```

## License

MIT. See `LICENSE`.
