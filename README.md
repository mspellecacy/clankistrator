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

A model request that is turned away for a passing reason (rate limit, overload, a dropped connection) is sent again up to three times with growing waits, so the agent keeps the turns it has already made. If the agent fails all the same, it is started again up to twice before the failure counts against the task. A task gets three retries, counted across failed attempts and failed reviews. After that it is marked failed, and every task that depends on it is marked failed too, so a run always ends. State is saved to `<output>/.clankistrator/state.json` after every task result, and `--resume` continues from it.

## Safety

Agents write files and run programs on your machine. Know what the guard rails are and are not:

- File tools are confined to the output directory. Absolute paths outside it, `..` escapes and symlinks leading out are refused.
- Commands run with the output directory as the working directory, with no stdin, with your provider API keys removed from their environment, and are killed (with their child processes) after `--command-timeout` seconds. Process-group cleanup is implemented for Unix only.
- Choose an output directory that is not inside another project (the examples use `../sumcli`, next to this repository). Build tools and version control search parent directories for their project files. Clankistrator refuses a `cargo` command that would fall through to a `Cargo.toml` outside the output directory, stops `git` from seeing an enclosing repository, and warns at startup when the output directory is nested in a project, but other tools (npm, go, pytest) are not guarded.
- **Commands in an ordinary run are not sandboxed.** (Those of `run_agent` are: see the last point.) They are ordinary processes running as you, and a program can touch anything you can. Clankistrator starts no shell itself, but a model can ask for `sh -c "..."` or `python3 -c "..."` and gets it. Building generated code also runs its build scripts. Use a container or a throwaway account if that matters to you.
- **What a command prints goes to the model's provider.** That includes anything it reads outside the output directory, and your `.env.local` is an ordinary file a command could read. In one live run a reviewer wanted a dependency's source, ran `find /` over the whole machine until the timeout five minutes later, and then read files under `~/.cargo/registry`. In the 13 saved run logs, one other command pointed outside its project: a `grep` over `/root/.cargo/registry/src`, a path that does not exist here. None touched `.env.local` or anything else under the home directory. Reviewers, the advisor and the command tool are now told to stay inside the project; that is an instruction to the model, not a barrier.
- **Reviewers and the advisor are not given the file-writing tools** (`write_file` and `edit_file`), but they can run commands, and nothing stops a command from changing a file.
- Without `--brave` nothing counts as done until you approve it, but the work has already been written and run by then. Approval gates progress, not execution.
- **`run_agent` (the MCP server with `--delegate`) works in a project you already have,** so more is at stake than in an output directory. Its agents run on the models you name on the server's command line and no others, and their file tools refuse environment files (`.env`, `.envrc`, `.env.*` other than `.env.example`, `.env.sample` and `.env.template`). Their commands run in a jail built with [bubblewrap](https://github.com/containers/bubblewrap) (Linux only), and a `--delegate` server does not start where it cannot build one unless you pass `--no-jail`. In the jail:
  - the project is the only part of your files that is there, read-only unless the call gave `write_access`;
  - your home directory is empty apart from the paths you name with `--allow-read`, so there are no SSH keys, no CLI logins and no other projects;
  - environment files in the project cannot be opened, and the paths you name with `--hide` are empty (the file tools refuse those too);
  - there is no network unless you pass `--allow-network`, no Docker socket, and none of the session's sockets (SSH agent, keyring, D-Bus);
  - the environment is a short list of harmless variables, and the server's own environment, which holds the provider keys, cannot be read through `/proc`;
  - with `write_access`, `.git/hooks`, `.git/config` and `.claude/settings.local.json` stay read-only, and the file tools refuse to write them: they run code later, outside the jail, and never show in a diff. If `.claude/settings.local.json` does not exist, the whole `.claude` directory is read-only to commands, because a file that is not there cannot be covered. That list is not everything that runs later: `node_modules`, build output, a submodule's own git directory and every other path git ignores stay writable, and none of them shows in a diff either. Give `write_access` only where you would let that model loose with a shell in that one directory.

  What the jail does not do: it does not decide what the model's provider gets to see (everything the agent reads in the project goes there), it finds environment files by a bounded walk (six levels deep, skipping `.git`, `node_modules`, `target` and other build directories), and it leaves the rest of the machine outside your home directory readable, as any local user could read it. With `--allow-network`, commands reach everything your machine reaches, including services on localhost. If the client is Claude Code, the project's permission rules and hooks still apply only to Claude's own tool calls; for an agent's commands the jail stands in for them. With `--no-jail`, commands are as unguarded as in an ordinary run: `sh -c 'cat .env.production'` works, and so does `git push`.

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
| `-m, --model <MODEL>` | `$DEFAULT_MODEL`, else `gpt-6.1-sol` | Model, see below |
| `--headless` | off | Log output instead of the TUI; approvals on stdin |
| `-b, --brave` | off | Approve reviewed tasks automatically |
| `-S, --max-steps <N>` | 50 | Orchestration passes that ran or reviewed a task; 0 is unlimited. Waiting for approval costs nothing |
| `--max-turns <N>` | 40 | Model turns (tool-call rounds) per worker attempt |
| `--command-timeout <SECS>` | 300 | Time limit for a command started by an agent |
| `--review-model <MODEL[,MODEL]>` | the advisor, else `--model` | Model for reviews and the design check. A stronger model here is the cheapest way to a stricter gate. With several, each task is reviewed by the first one from another provider than its builder |
| `--workers <MODEL,MODEL>` | only `--model` | Models that may build, cheapest first. See "An advisor and several worker models" |
| `--advisor-model <MODEL>` | none | A stronger model that plans, assigns tasks to the worker models, advises after a failed attempt, and reviews unless `--review-model` is set |
| `--fix-rounds <N>` | 2 | How many times unmet design requirements are turned into a fix task |
| `--skip-design-check` | off | Do not check the finished project against the design |
| `--no-prompt-cache` | off | Do not ask Anthropic to cache prompts between an agent's turns |
| `--check` | off | Only check the project in `--output` against `--design` and print the findings. Exits 1 if anything is unmet |
| `--resume` | off | Continue the run saved in the output directory. Interrupted and failed tasks are queued again |
| `--probe <MODEL>...` | | Test the given models (one request and one tool call each) instead of running |
| `--mcp-server` | off | Run as an MCP server instead (see below) |
| `--delegate` | off | With `--mcp-server`: also offer `run_agent`, which runs agents in a project for the MCP client (see below) |
| `--env-file <FILE>` | none | Read settings from this file instead of `.env.local` and `.env` in the current directory |
| `--no-jail` | off | With `--delegate`: run agents' commands as ordinary processes instead of in a bubblewrap jail. Without it, the server does not start where it cannot build a jail |
| `--allow-network` | off | With `--delegate`: let jailed commands use the network |
| `--allow-read <PATH,PATH>` | none | With `--delegate`: absolute paths outside the project that jailed commands may read, such as a toolchain under your home directory |
| `--hide <PATH,PATH>` | none | With `--delegate`: paths in the project, relative to its root, that agents may not see |

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
| OpenRouter | `openrouter:` | `OPENROUTER_API_KEY` (or `OPENROUTER_KEY`) | yes: probe with two free models, and a full run with a free model as builder (see below) |
| OpenAI-compatible server | `compatible:` | `OPENAI_COMPATIBLE_BASE_URL` (for example `http://localhost:1234/v1`); `OPENAI_COMPATIBLE_API_KEY` if the server wants one | no |

"Run live" says whether that path has been exercised against the real service by the maintainers. The rows marked "no" compile and are wired up the same way, but nobody has seen them work; probe before you trust them. "Probe only" means a request and a tool call went through, and no full run has been made.

Examples: `--model gpt-6.1-sol`, `--model ollama:qwen2.5-coder:14b`, `--model openrouter:vendor/model-name`, `--model compatible:whatever-your-server-calls-it`. OpenRouter and OpenAI-compatible servers are how to reach providers that have no row of their own.

Model names go out of date. Providers retire models on a schedule (OpenAI shuts down the `gpt-4o` snapshots on 2027-01-14, for instance), and a name that works today may be a provider's oldest and weakest offering. Check the provider's model list when choosing, and probe: every run starts with a request to each model it will use, so a retired name stops the run at once rather than failing halfway.

Workers and reviewers can use different models, from different providers: `--model` builds, `--review-model` judges. A cheaper model that builds with a stronger one that reviews is a reasonable combination.

`--review-model` takes several models, separated by commas. The first does the design check. Each task is then reviewed by the first of them that comes from a different provider than the model that built it, so with `--review-model claude-sonnet-5-5,gpt-6.1-sol` work built by an OpenAI model is judged by Claude and work built by a Claude model is judged by OpenAI. The point is that models of one family tend to share blind spots; it also spreads the requests over two accounts. If no reviewer is from another provider, the first one reviews. Seen live in one run with three worker models and two reviewers: Claude Haiku built the scaffold and `gpt-6.1-sol` reviewed it, two OpenAI models built the rest and Claude Sonnet reviewed that, and the design check ended with nothing unmet. That is one review by the second reviewer, of an easy task that passed, and how strict a gate each model makes is something to find out per model (see Limitations).

#### An advisor and several worker models

One strong model can lead a run while cheaper ones do the building:

```bash
cargo run --release -- --design design.md --output ../my-project \
  --advisor-model claude-sonnet-5-5 \
  --workers openrouter:cohere/north-mini-code:free,gpt-6-luna,claude-sonnet-5-5
```

`--workers` lists the models that may build, **cheapest first**. Only the models you name are used; having a key for a provider does not put its models in the pool. `--advisor-model` is the lead. It never writes the project's files. What each part does:

- **Planning and assignment.** The advisor writes the plan and gives each task to one of the worker models. A model's name may mean nothing to it, so it is told what Clankistrator knows: the order, which models are free and how many free requests are left today, and how each model's attempts have fared so far in the run. It is asked for the cheapest model that can do each task well, to spread tasks between models it considers equal, and to give a one-sentence reason for each choice, which the log prints next to the assignment.
- **Stepping up.** A task that fails twice on its model moves to the next one in the list. This works without an advisor too. A task still gets four attempts in all, whatever models make them.
- **Advice after a failure.** When an attempt fails (the reviewer found something unmet, or the worker gave up), the advisor reads the project (it has no file-writing tools) and writes guidance for the next attempt: the cause and the exact change. It can also move the task to another worker model. It is consulted before every retry, and before the first new attempt at a task that an earlier run gave up on (`--resume`). Its answers are kept in `.clankistrator/transcripts/<task>.advice.*.md`.
- **A model that stops working** during a run (a used-up free allowance, a revoked key) is left out for the rest of the run, and its task goes to the next model without losing an attempt. If no worker model is left, the remaining tasks fail; building never moves to a model that was not listed as a worker. A rate limit or an overloaded provider does not count as "stopped working".
- **Fix tasks** from the design check go to the last model in the list, because they are what the earlier work missed.
- **Reviews** are done by the advisor unless `--review-model` names another model, or several (see above).

The run summary lists calls, tokens and passed and failed attempts per model, so you can see where the work went and what was paid for. (For a project started with an older version, the per-model lines begin with the first resumed run, while the totals cover everything.) Each consultation is a multi-turn run of your most expensive model; that is the price of the cheaper models' failures. A review whose only problem is that the reviewer did not show its evidence is not treated as a failure of the worker: no advice is asked and the task stays where it is.

Not built: a tool that lets a worker ask the advisor a question in the middle of its own attempt. The advisor is consulted by Clankistrator at fixed points, not by the workers.

Run live so far: a full build of the bundled calculator design with one advisor and two worker models, in which the advisor planned two tasks and gave one to each worker, reviewed them, and the fix task from the design check went to the stronger worker. In a second run the cheapest worker was a free OpenRouter model with 12 of the day's requests left; the advisor was told so and gave both tasks to the paid worker. Nothing failed review in either run. In a third run the only worker was `gpt-4o`, a model its provider has deprecated, and the advisor was consulted three times. Its advice named the missing tests after the first task failed review, and that task passed on the next attempt. For the second task it named the removed `ratatui` type that stopped the build, and the next attempt compiled; that task still failed after four attempts, the last of which had no fresh advice (at the time the advisor was consulted at most twice) and left a one-line compile error. When the run was resumed, the advisor looked first and named the line and the change, and `gpt-4o` fixed it in one attempt; the design check then ended with nothing unmet. Stepping up and fallback are still covered only by tests against scripted models and have not been seen in a real run.

#### Checking that a model works

```bash
cargo run --release -- --probe gpt-6.1-sol claude-sonnet-5-5 ollama:llama3.2
```

For each model, `--probe` makes one plain request and then one tool call, in which the model has to call a tool with a list argument and repeat a code that only the tool knows. It uses no files and runs no commands, and costs a few hundred tokens per model. A model that fails the tool call cannot be used, because workers and reviewers do everything through tools. The exit code is 0 only if every model passed.

Every run also starts with one small request to each model it will use, so a mistyped model name or a dead key stops the run before any work is done.

#### Free models on OpenRouter

OpenRouter offers some models at no charge; their names end in `:free`, for example `--model openrouter:cohere/north-mini-code:free`. The list changes, so take names from [openrouter.ai/models](https://openrouter.ai/models) (filter by price) and probe them. On 2026-11-10 `cohere/north-mini-code:free` and `nvidia/nemotron-3-ultra-550b-a55b:free` passed the probe, and `google/gemma-4-31b-it:free` was turned away as rate-limited by its upstream provider.

What to know before relying on them:

- **The allowance is counted in requests.** An account that has never bought credits gets 50 free-model requests a day and 20 a minute; one that has bought 10 credits gets 1,000 a day. Every turn of an agent is one request, and a task takes a worker and a reviewer several turns each, so 50 requests do not go far. Measured: building the bundled `docs/sumcli_design.md` (three small tasks, each passing review first time) took 38 free-model requests with `cohere/north-mini-code:free` as builder and another provider as reviewer, about 12 per task. A project of that size fits in a day's allowance; one rejected task may already not. A run says at the start how many OpenRouter reports are left for the day, and its summary says how many it sent. Using the free model only to build (`--model`) and another provider to review (`--review-model`) halves the requests.
- **Clankistrator spaces requests to free models** about 3 seconds apart, across all agents, to stay under the per-minute limit. Runs are slower for it.
- **When the day's allowance is used up,** a run refuses to start, and a run already under way fails its remaining tasks quickly and ends with exit code 1. (OpenRouter answers a used-up day and a busy minute with the same error, so after a rate-limit error Clankistrator asks OpenRouter for the day's counter.) The state is saved; `--resume` after the counter resets, at midnight UTC, continues it.
- **To make sure nothing is ever billed,** set `OPENROUTER_FREE_ONLY=true`. Any `openrouter:` model whose name does not end in `:free` is then refused before a request is sent.
- **Free endpoints may keep and train on what you send.** The prompts contain your design document and the generated code. Check the model's page and your OpenRouter privacy settings if that matters for the project.
- **Free models are small or heavily shared.** Passing the probe shows that tool calls work, not that the model can build your design. A stronger `--review-model` still tells you honestly whether it did.

rig's own log lines are held at `warn` because at `info` it prints every tool call with its full arguments and result; set `RUST_LOG=info` to see them.

Settings are read from the environment, `.env.local` and `.env` (in that order of precedence): the variables in the table above, `OPENROUTER_FREE_ONLY`, `DEFAULT_MODEL`, `WORKING_DIRECTORY` and `LOG_LEVEL` (`RUST_LOG` overrides it). None of the key variables is passed on to the commands that agents run.

## The TUI

| Key | Action |
| --- | --- |
| `1`-`4`, `Tab` | Switch between Tasks, Logs, HITL and Stats |
| `↑` `↓` | Select a task; its description, summary, error and feedback show on the right |
| `a` / `r` (HITL) | Approve, or reject with typed feedback for the worker |
| `m` | Menu: brave mode, model, max steps. Changes apply to agents started afterwards. The model set here is `--model`; with `--workers`, the worker models stay as given |
| `PgUp` `PgDn` `Esc` (Logs) | Scroll, return to following |
| `q`, `Ctrl-C` | Quit. Progress is saved; continue with `--resume` |

Stats shows the token counts reported by the provider, including how many input tokens came from its prompt cache, and the result of the design check.

## MCP server

`clankistrator --mcp-server` speaks MCP over stdio and offers three tools:

- `parse_markdown_design`: split a Markdown file into sections.
- `plan_tasks`: produce the validated task graph for a design document. This calls the model.
- `project_status`: read the saved state of a run from its output directory.

Started like this, the server does not execute tasks, run commands or write files.

### Running agents for an MCP client

With `--delegate` the server offers a fourth tool, `run_agent`, which does run commands and change files. It lets the client (Claude Code, say) hand a piece of work in an existing repository to a model of your choosing: one agent, one prompt, a fresh context, and its final reply back. There is no design document, no plan and no review; the client is the orchestrator.

```bash
clankistrator --mcp-server --delegate \
  --env-file ~/dev/clankistrator/.env.local \
  --workers gpt-6-luna,claude-sonnet-5-5 \
  --max-turns 80 --command-timeout 900 \
  --allow-read ~/.nvm/versions/node/v25.2.0 \
  --hide supabase/.temp,tmp/private
```

| Argument | Meaning |
| --- | --- |
| `directory` | Absolute path of the project. It must be the root of the repository or of a worktree, because commands cannot see a repository above the directory they run in. It is never created |
| `prompt` | The task |
| `agent` | The agent's standing instructions: a Markdown (`.md`) file, or a bare name looked up as `.claude/agents/<name>.md` in the directory. Front matter is ignored |
| `model` | One of the server's models. Defaults to the first |
| `write_access` | Also give the agent `write_file` and `edit_file`. Off by default |
| `report_path` | A file inside the directory for the agent's report. If the agent does not write it, its final reply is saved there |
| `max_turns` | Model turns for this call, instead of the server's `--max-turns` |

What to know:

- **The models are yours to choose, not the client's.** An agent runs only on a model named with `--model`, `--workers`, `--review-model` or `--advisor-model` when the server was started. `$DEFAULT_MODEL` and the built-in fallback do not count, and a `--delegate` server with no model named does not start. What an agent reads goes to its model's provider, so this list is also the list of providers that may see the project.
- **Settings do not come from the current directory.** A client usually starts the server inside the project the agents will work on, where `.env.local` is the project's own: reading it would bill the project's provider key and pass its settings on to every command. A `--delegate` server reads the environment and `--env-file`, nothing else.
- **Agent files written for Claude Code mostly work as they are.** The agent is told which tools it has here and which of Claude Code's they stand for. It has no skills, slash commands, subagents or MCP tools, and is told to stop and say so if its instructions need one, so an agent that only wraps `/code-review` cannot be delegated.
- **Nothing is written to the project** except what the agent writes and the report: no `.clankistrator/` directory, no transcripts.
- **A failed run** (a provider error, or all turns used without a final reply) writes no report and leaves an earlier one alone. The files may be half changed; the error says so.
- **One agent, started once.** A request that a provider turns away for a passing reason is sent again, as everywhere. An agent that fails is not started again: a second one would begin with an empty context on files the first had half changed. An agent run ends after 30 minutes at the latest.
- **Stopping.** When the client cancels the call (MCP's `notifications/cancelled`) or goes away (its end of stdin closes), the agent is stopped and the commands it started are killed. While an agent works, the server keeps answering other requests and, if the client asked for progress, sends it a progress notification every minute, so that a long run is not taken for a dead one. Whether Claude Code sends that cancellation when you interrupt it, and whether it asks for progress, has not been observed; the server's log (stderr) says both for each run.
- **Commands are jailed** (see Safety). A toolchain installed under your home directory, such as node from nvm or cargo from rustup, is not there until you name it with `--allow-read`; a command that needs it fails with "No such file or directory". What the jail allows is settled on the server's command line, and no argument of `run_agent` loosens it. A `report_path` under a hidden path is refused.
- **Run live on 2026-11-11, from Claude Code.** A reviewer agent file written for Claude Code ran unchanged against a private repository on `claude-haiku-5-5` and on `gpt-6-luna`, and both returned a report in the format the file asks for. Those two runs were before the jail existed, inside a bubblewrap wrapper around the whole server. With the jail, two canary agents on `claude-haiku-5-5` then tried what it forbids. Read-only: creating a file, reading environment files, listing the home directory, Docker, the network and the server's environment through `/proc` all failed, while `git diff` and `node` worked and a 25-second command ran to its end. With `write_access`, in a scratch repository: writing files and `git commit` worked, and `.git/hooks`, `.git/config` and `.claude/settings.local.json` could not be written by a command or by `write_file`. A worktree has only been run in tests, and `--allow-network` not through an agent at all: only its mounts were tried by hand, where a name resolved and an HTTPS request went through.

Agents also get `edit_file`, which replaces one piece of text in a file. Workers in an ordinary run have it too.

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
├── delegate.rs        One agent on one prompt in an existing project (run_agent)
├── orchestrator/      Planning, the run loop, system prompts
├── tools/             Sandboxed file tools and command execution
├── tui/               Terminal UI
├── parser/            Markdown section parser (used by the MCP server)
└── mcp.rs             MCP server
docs/                  Example design documents
```

## License

MIT. See `LICENSE`.
