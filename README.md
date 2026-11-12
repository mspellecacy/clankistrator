# Clankistrator

Clankistrator is an MCP server with one tool, `run_agent`. It lets an MCP client (Claude Code, say) hand a piece of work in an existing repository to a model of your choosing, from any of several providers: one agent, one prompt, a fresh context, and its final reply back. The agent's commands run in a jail. There is no plan and no review; the client is the orchestrator.

It is a Rust binary built on [rig](https://github.com/0xPlaygrounds/rig), and it speaks MCP (JSON-RPC) over stdio.

Until November 2026 this project also turned a design document into a project with a planner, worker and reviewer agents and a terminal UI. That was removed; the last commit with it is `d7f9429`.

## Getting started

Requirements: [Rust](https://rustup.rs/) 1.88 or newer, [bubblewrap](https://github.com/containers/bubblewrap) (Linux only; see Safety), and an API key for a provider, or a local Ollama. Bubblewrap 0.11.0 is the version this has been run with; `scratch` needs its overlay support, and the server checks for that when a call asks for it.

```bash
git clone https://github.com/mspellecacy/clankistrator.git
cd clankistrator
cargo build --release
cp .env.example .env.local    # then put your keys in it
```

Check that the models you mean to use work from this machine:

```bash
target/release/clankistrator --env-file .env.local --probe gpt-6-luna claude-sonnet-5-5
```

Then add the server to your MCP client. For Claude Code:

```bash
claude mcp add-json --scope local clankistrator '{
  "command": "/path/to/clankistrator/target/release/clankistrator",
  "args": [
    "--env-file", "/path/to/clankistrator/.env.local",
    "--models", "gpt-6-luna,claude-sonnet-5-5",
    "--max-turns", "80", "--command-timeout", "900",
    "--allow-read", "/home/you/.nvm/versions/node/v25.2.0",
    "--hide", "tmp/private",
    "--usage-log", "/home/you/.local/state/clankistrator/usage.jsonl"
  ]
}'
```

A run makes real, billed model calls. An agent sends its whole conversation again on every turn, so input tokens dominate; prompt caching takes most of the cost out of the repeats where the provider supports it.

## The `run_agent` tool

| Argument | Meaning |
| --- | --- |
| `directory` | Absolute path of the project. It must be the root of the repository or of a worktree, because commands cannot see a repository above the directory they run in. It is never created |
| `prompt` | The task |
| `agent` | The agent's standing instructions: a Markdown (`.md`) file, or a bare name looked up as `.claude/agents/<name>.md` in the directory. Front matter is ignored |
| `model` | One of the server's models. Defaults to the first |
| `write_access` | Also give the agent `write_file` and `edit_file`. Off by default |
| `scratch` | Let the agent's commands change files in a throwaway copy of the project, deleted when the run ends. Needs the jail; cannot be combined with `write_access` |
| `report_path` | A file inside the directory for the agent's report. If the agent does not write it, its final reply is saved there |
| `max_turns` | Model turns for this call, instead of the server's `--max-turns` |

The result is JSON: the model, the reply, where the report went and who wrote it, how many seconds the run took, and the token counts the provider reported (input as one total, with the part read from the cache beside it).

What to know:

- **The models are yours to choose, not the client's.** An agent runs only on a model named with `--models` (or `--model`) when the server was started, and a server with no model named does not start. What an agent reads goes to its model's provider, so this list is also the list of providers that may see the project.
- **Settings do not come from the current directory.** A client usually starts the server inside the project the agents will work on, where `.env.local` is the project's own: reading it would bill the project's provider key and pass its settings on to every command. The server reads the environment and `--env-file`, nothing else.
- **Agent files written for Claude Code mostly work as they are.** The agent is told which tools it has here and which of Claude Code's they stand for. It has no skills, slash commands, subagents or MCP tools, and is told to stop and say so if its instructions need one, so an agent that only wraps `/code-review` cannot be run this way.
- **Nothing is written to the project** except what the agent writes and the report.
- **A failed run** (a provider error, or all turns used without a final reply) writes no report and leaves an earlier one alone. The files may be half changed; the error says so.
- **One agent, started once.** A request that a provider turns away for a passing reason (rate limit, overload, a dropped connection) is sent again up to three times with growing waits. An agent that fails is not started again: a second one would begin with an empty context on files the first had half changed. An agent run ends after 30 minutes at the latest, or after `--run-timeout` minutes.
- **Stopping.** When the client cancels the call (MCP's `notifications/cancelled`) or goes away (its end of stdin closes), the agent is stopped and the commands it started are killed. While an agent works, the server keeps answering other requests and, if the client asked for progress, sends it a progress notification every minute, so that a long run is not taken for a dead one. Claude Code does send that cancellation: on 2026-11-11 a run it had moved to the background was stopped from Claude Code, and its command was killed, its scratch layer deleted and the run logged as cancelled. Whether it asks for progress has not been observed; the server's log (stderr) says so for each run.
- **`scratch` is for an agent that has to change code to test it,** such as a reviewer running a mutation check. Its commands see the project through an overlay whose writes go to a directory of their own, so a change stays for the agent's later commands and never reaches the project; nothing of yours is bound writable, a worktree's repository included. The agent gets no file-writing tools, and `read_file` goes on showing the real file, which it is told. Commands in a scratch run take turns. The layer is deleted when the run ends, is cancelled or times out; a server that is killed outright leaves a `clankistrator-run-*` directory in the temporary directory. Where `/tmp` is kept in memory, set `TMPDIR` for the server to a directory on disk.
- **An environment file that git tracks shows as modified inside the jail** (` M` in `git status`), because the jail covers it with an unreadable file. It is not refused, as a tracked hidden path is, so tell an agent that stages files about it, or stop tracking the file.
- **`/tmp` belongs to the run.** Each command is a jail of its own, so a jailed agent gets a `/tmp` that lasts from one command to the next, where it can keep notes, snapshots and scripts. It is a directory only you can read, and it is deleted with the run.
- **A hidden path that git tracks is refused,** with its name, before any model is called. Inside the jail git would report its files as deleted, which an agent may stage or put in its report, and hiding it protects nothing: the content is in the repository's history.
- **`--usage-log` keeps a line per run:** time, directory, agent, model, access (`read`, `write` or `scratch`), outcome (`ok`, `failed` or `cancelled`), seconds and token counts. It never holds a prompt, a reply or an error text, and a line that cannot be written does not fail the run.
- **Commands are jailed** (see Safety). A toolchain installed under your home directory, such as node from nvm or cargo from rustup, is not there until you name it with `--allow-read`; a command that needs it fails with "No such file or directory". What the jail allows is settled on the server's command line, and no argument of `run_agent` loosens it. A `report_path` under a hidden path is refused.

## Safety

Agents read a project you already have, run programs on your machine and, if a call asks, change files. Know what the guard rails are and are not:

- Agents run on the models you name on the server's command line and no others.
- File tools are confined to the project directory. Absolute paths outside it, `..` escapes and symlinks leading out are refused. They also refuse environment files (`.env`, `.envrc`, `.env.*` other than `.env.example`, `.env.sample` and `.env.template`).
- Commands get no stdin and are killed, with their child processes, after `--command-timeout` seconds. The server starts no shell itself.
- **Commands run in a jail built with bubblewrap** (Linux only). The server does not start where it cannot build one, unless you pass `--no-jail`. In the jail:
  - the project is the only part of your files that is there, read-only unless the call gave `write_access`;
  - your home directory is empty apart from the paths you name with `--allow-read`, so there are no SSH keys, no CLI logins and no other projects;
  - environment files in the project cannot be opened, and the paths you name with `--hide` are empty (the file tools refuse those too);
  - there is no network unless you pass `--allow-network`, no Docker socket, and none of the session's sockets (SSH agent, keyring, D-Bus);
  - the environment is a short list of harmless variables, and the server's own environment, which holds the provider keys, cannot be read through `/proc`;
  - with `write_access`, `.git/hooks`, `.git/config` and `.claude/settings.local.json` stay read-only, and the file tools refuse to write them: they run code later, outside the jail, and never show in a diff. If `.claude/settings.local.json` does not exist, the whole `.claude` directory is read-only to commands, because a file that is not there cannot be covered. That list is not everything that runs later: `node_modules`, build output, a submodule's own git directory and every other path git ignores stay writable, and none of them shows in a diff either. Give `write_access` only where you would let that model loose with a shell in that one directory.
- **What the jail does not do.** It does not decide what the model's provider gets to see: everything the agent reads in the project goes there. It finds environment files by a bounded walk (six levels deep, skipping `.git`, `node_modules`, `target` and other build directories). It leaves the rest of the machine outside your home directory readable, as any local user could read it. With `--allow-network`, commands reach everything your machine reaches, including services on localhost. If the client is Claude Code, the project's permission rules and hooks still apply only to Claude's own tool calls; for an agent's commands the jail stands in for them.
- **With `--no-jail`, commands are ordinary processes running as you.** `sh -c 'cat .env.production'` works, and so does `git push`.

## Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--models <MODEL,MODEL>` | none | The only models an agent may run on, the default first. `--workers` is accepted as another name for it |
| `-m, --model <MODEL>` | none | One more model, placed before `--models` |
| `--env-file <FILE>` | none | Read settings (API keys, `LOG_LEVEL`) from this file. Without it they come from the environment only |
| `--max-turns <N>` | 40 | Model turns (tool-call rounds) an agent gets, unless a call asks for another number |
| `--command-timeout <SECS>` | 300 | Time limit for a command started by an agent |
| `--run-timeout <MINUTES>` | 30 | How long one agent run (all of its turns) may take |
| `--no-prompt-cache` | off | Do not ask Anthropic to cache prompts between an agent's turns |
| `--no-jail` | off | Run agents' commands as ordinary processes instead of in a bubblewrap jail. Without it, the server does not start where it cannot build a jail |
| `--allow-network` | off | Let jailed commands use the network |
| `--allow-read <PATH,PATH>` | none | Absolute paths outside the project that jailed commands may read, such as a toolchain under your home directory |
| `--hide <PATH,PATH>` | none | Paths in the project, relative to its root, that agents may not see. A call is refused if git tracks anything under one of them |
| `--usage-log <FILE>` | none | Append one JSON line per agent run to this file. It must be absolute and outside the projects agents work in |
| `--probe <MODEL>...` | | Test the given models (one request and one tool call each) instead of serving |

`--mcp-server --delegate` is accepted and ignored, so a command line written when the server had other modes still works. `--mcp-server` without `--delegate` is refused: it used to start a server that could not run a command.

Logs go to stderr. `LOG_LEVEL` sets the level (default `info`) and `RUST_LOG` overrides it. rig's own log lines are held at `warn` because at `info` it prints every tool call with its full arguments and result.

## Models and providers

A model is named as `prefix:model`. For three providers the prefix can be left out: names starting with `claude` go to Anthropic, names starting with `gemini` go to Gemini, and any other bare name goes to OpenAI. The rest are only ever chosen by prefix.

| Provider | Prefix | Needs | Run live |
| --- | --- | --- | --- |
| OpenAI | `openai:` (optional) | `OPENAI_API_KEY` | yes: probe and agent runs |
| Anthropic | `anthropic:` (optional) | `ANTHROPIC_API_KEY` | yes: probe and agent runs |
| Gemini | `gemini:` (optional) | `GEMINI_API_KEY` | no |
| Ollama | `ollama:` | a running server; `OLLAMA_BASE_URL` if it is not on `http://localhost:11434` | no |
| OpenRouter | `openrouter:` | `OPENROUTER_API_KEY` (or `OPENROUTER_KEY`) | probe only, with two free models |
| OpenAI-compatible server | `compatible:` | `OPENAI_COMPATIBLE_BASE_URL` (for example `http://localhost:1234/v1`); `OPENAI_COMPATIBLE_API_KEY` if the server wants one | no |

"Run live" says whether that path has been exercised against the real service by the maintainers. The rows marked "no" compile and are wired up the same way, but nobody has seen them work; probe before you trust them. "Probe only" means a request and a tool call went through, and no agent run has been made through `run_agent`.

Model names go out of date. Providers retire models on a schedule, and a name that works today may be a provider's oldest and weakest offering. Check the provider's model list when choosing, and probe.

### Checking that a model works

```bash
clankistrator --env-file .env.local --probe gpt-6.1-sol claude-sonnet-5-5 ollama:llama3.2
```

For each model, `--probe` makes one plain request and then one tool call, in which the model has to call a tool with a list argument and repeat a code that only the tool knows. It uses no files and runs no commands, and costs a few hundred tokens per model. A model that fails the tool call cannot be used, because agents do everything through tools. The exit code is 0 only if every model passed.

### Free models on OpenRouter

OpenRouter offers some models at no charge; their names end in `:free`. The list changes, so take names from [openrouter.ai/models](https://openrouter.ai/models) (filter by price) and probe them.

- **The allowance is counted in requests.** An account that has never bought credits gets 50 free-model requests a day and 20 a minute; one that has bought 10 credits gets 1,000 a day. Every turn of an agent is one request.
- **Requests to free models are spaced** about 3 seconds apart, across all agents, to stay under the per-minute limit.
- **When the day's allowance is used up,** the agent run fails and says so. OpenRouter answers a used-up day and a busy minute with the same error, so after a rate-limit error the server asks OpenRouter for the day's counter.
- **To make sure nothing is ever billed,** set `OPENROUTER_FREE_ONLY=true`. Any `openrouter:` model whose name does not end in `:free` is then refused before a request is sent.
- **Free endpoints may keep and train on what you send.** An agent's prompts contain your project's code.

## What has been run live

- **From Claude Code, 2026-11-11.** A reviewer agent file written for Claude Code ran unchanged against a private repository on `claude-haiku-5-5` and on `gpt-6-luna`, and both returned a report in the format the file asks for. Those two runs were before the jail existed, inside a bubblewrap wrapper around the whole server.
- **The jail, 2026-11-11.** Two canary agents on `claude-haiku-5-5` tried what it forbids. Read-only: creating a file, reading environment files, listing the home directory, Docker, the network and the server's environment through `/proc` all failed, while `git diff` and `node` worked and a 25-second command ran to its end. With `write_access`, in a scratch repository: writing files and `git commit` worked, and `.git/hooks`, `.git/config` and `.claude/settings.local.json` could not be written by a command or by `write_file`.
- **Not yet through an agent:** a worktree (tests only), and `--allow-network` (only its mounts were tried by hand, where a name resolved and an HTTPS request went through).

## Limitations

- Linux only for the jail. Elsewhere the server starts only with `--no-jail`.
- Token counts only include agent runs that finished; a run that errored or hit its turn limit is not counted.
- The test suite drives the server with scripted model replies; it makes no provider calls.
- An agent cannot run an interactive program, and it has no way to ask the client a question mid-run.

## Development

```bash
cargo test
cargo clippy --all-targets
```

```text
src/
├── main.rs            Command line, settings, the jail policy, the probe report
├── mcp.rs             The MCP server and the usage log
├── delegate.rs        One agent on one prompt in an existing project
├── llm.rs             Provider selection, retries, and the single call into rig
├── probe.rs           The model check behind --probe
└── tools/             File tools, command execution, and the jail
integrations/          Files other tools need to use this server
```

## License

MIT. See `LICENSE`.
