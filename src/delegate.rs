//! Run one agent on a prompt in a project that already exists.
//!
//! No plan and no review: whoever calls it (the MCP server's `run_agent`
//! tool) decides what the agent is for, and gets its final reply back, saved
//! to a report file if one was named.
//!
//! Nothing is written to the project except what the agent itself writes and
//! that report: a state directory would show up as an untracked path in
//! somebody's repository.

use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::llm::{AgentRequest, Llm, LlmError, TokenUsage, check_credentials};
use crate::tools::{JailPolicy, Sandbox, ToolAccess, Writes};

/// Where a project keeps the agent definitions that can be named without a path.
const AGENTS_DIR: &str = ".claude/agents";

/// What the caller asks for.
#[derive(Debug, Clone)]
pub struct Delegation {
    /// The root of the project, which must exist.
    pub directory: PathBuf,
    /// A Markdown file with the agent's standing instructions: a path, or a
    /// bare name looked up as `.claude/agents/<name>.md` in the project.
    pub agent: Option<String>,
    pub prompt: String,
    pub model: String,
    /// Whether the agent gets the tools that write and edit files.
    pub write_access: bool,
    /// Whether the agent's commands may change files in a throwaway copy of
    /// the project. Needs a jail, and excludes `write_access`.
    pub scratch: bool,
    /// File, inside the project, that is to hold the agent's report.
    pub report_path: Option<String>,
    pub max_turns: usize,
    pub command_timeout: Duration,
    /// The jail the agent's commands run in. `None` runs them as ordinary processes.
    pub jail: Option<JailPolicy>,
    /// Paths in the project, relative to its root, that the agent may not see.
    pub hidden: Vec<PathBuf>,
}

/// A delegation that has passed every check that needs no model call.
#[derive(Debug)]
pub struct Prepared {
    request: AgentRequest,
    report: Option<Report>,
    summary: Summary,
}

/// What a run was, for a usage record: no prompt and no reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Summary {
    pub directory: PathBuf,
    /// The agent as the caller named it, if it named one.
    pub agent: Option<String>,
    pub model: String,
    /// "read", "write" or "scratch".
    pub access: &'static str,
}

impl Prepared {
    pub fn summary(&self) -> &Summary {
        &self.summary
    }
}

#[derive(Debug)]
struct Report {
    path: PathBuf,
    /// The path as the caller wrote it, for messages.
    given: String,
}

#[derive(Debug, PartialEq, Eq)]
pub struct SavedReport {
    pub path: String,
    /// True if the agent wrote the file itself; false if its final reply was saved there.
    pub written_by_agent: bool,
}

#[derive(Debug)]
pub struct Outcome {
    pub model: String,
    pub reply: String,
    pub usage: TokenUsage,
    pub report: Option<SavedReport>,
}

/// Stops the run's commands however the run ends. rig runs each tool call in
/// a task of its own, so a command would otherwise outlive a cancelled run
/// and go on changing files with nobody waiting for it.
struct StopCommands(Sandbox);

impl Drop for StopCommands {
    fn drop(&mut self) {
        self.0.stop_commands();
        self.0.discard_run_files();
    }
}

/// The hidden paths that hold files git tracks. Hiding one makes git report
/// its files as deleted inside the jail, where an agent may stage or report
/// them, and hides nothing: their content is in the repository's history.
fn tracked_hidden(root: &Path, hidden: &[PathBuf]) -> Vec<String> {
    if hidden.is_empty() || !root.join(".git").exists() {
        return Vec::new();
    }
    let mut tracked = Vec::new();
    for path in hidden {
        // This is the server's own git call in somebody's repository, so the
        // repository's configuration gets to start no program through it.
        let listed = std::process::Command::new("git")
            .arg("-C")
            .arg(root)
            .args(["-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null", "ls-files", "-z", "--"])
            .arg(path)
            .stdin(std::process::Stdio::null())
            .output();
        match listed {
            Ok(output) if output.status.success() => {
                let files = output.stdout.split(|byte| *byte == 0).filter(|name| !name.is_empty()).count();
                if files > 0 {
                    tracked.push(format!("{} ({files} tracked file{})", path.display(), if files == 1 { "" } else { "s" }));
                }
            }
            // No git, or not a repository git can read: nothing to report deleted.
            _ => {}
        }
    }
    tracked
}

/// The text of an agent definition without its YAML front matter (the
/// `---` block that names and describes the agent for another harness).
pub fn strip_front_matter(text: &str) -> &str {
    let text = text.trim_start_matches('\u{feff}');
    let Some(rest) = text.strip_prefix("---") else { return text };
    let Some(rest) = rest.strip_prefix("\r\n").or_else(|| rest.strip_prefix('\n')) else { return text };
    let mut offset = 0;
    for line in rest.split_inclusive('\n') {
        offset += line.len();
        if line.trim_end() == "---" {
            return &rest[offset..];
        }
    }
    text
}

/// Where an agent's instructions are. Only Markdown files are accepted: the
/// whole file goes to the model's provider, and a caller's slip must not send
/// an environment file or a key there.
fn agent_file(directory: &Path, agent: &str) -> Result<PathBuf, String> {
    let given = Path::new(agent);
    if given.extension().is_some_and(|ext| ext == "md") {
        Ok(directory.join(given))
    } else if given.components().count() == 1 {
        Ok(directory.join(AGENTS_DIR).join(format!("{agent}.md")))
    } else {
        Err(format!("agent {agent:?} is neither a bare name nor the path of a Markdown (.md) file"))
    }
}

fn read_agent(directory: &Path, agent: &str) -> Result<String, String> {
    let file = agent_file(directory, agent.trim())?;
    let text = std::fs::read_to_string(&file)
        .map_err(|e| format!("could not read the agent file {}: {e}", file.display()))?;
    let body = strip_front_matter(&text).trim();
    if body.is_empty() {
        return Err(format!("the agent file {} has no instructions in it", file.display()));
    }
    Ok(body.to_string())
}

/// What the agent is told about where it is. Agent files are usually written
/// for another harness, so this names the tools that exist here and says what
/// to do about the ones that do not.
fn rules(
    write_access: bool,
    scratch: bool,
    report: Option<&str>,
    command_timeout: Duration,
    jail: Option<&JailPolicy>,
) -> String {
    let mut text = String::from(
        "You are working in an existing project, through Clankistrator. Your instructions may have been \
         written for a different tool; this is what you have here.\n\n# Your tools\n\n",
    );
    text.push_str(if write_access {
        "- read_file, list_files, execute_command, write_file and edit_file.\n\
         - Change an existing file with edit_file, which replaces one piece of text. Use write_file for a \
         new file. Do not write a whole existing file out again to change a few lines of it.\n"
    } else {
        "- read_file, list_files and execute_command. You have no tool that writes files, and you must not \
         change files through commands either, unless your instructions explicitly ask for a change.\n"
    });
    if let Some(jail) = jail {
        text.push_str(&format!(
            "- Commands run in a jail. In it the project is {}, there is {}, and nothing else on this machine \
             is yours to read: the home directory is empty and environment files cannot be opened. \
             `Read-only file system`, `Permission denied` or a file that is missing there is the jail at \
             work, not a fault. Do not look for a way around it. If it keeps you from what you were asked \
             to do, say so in your reply. /tmp is yours: what you put there stays for your later commands \
             and is deleted when you finish.\n",
            if write_access {
                "writable"
            } else if scratch {
                "a throwaway copy"
            } else {
                "read-only"
            },
            if jail.network { "a network" } else { "no network" },
        ));
    }
    if scratch {
        text.push_str(
            "- Your commands work on a throwaway copy of the project. They may change files in it, for \
             example to break a line and see whether a test notices. A change stays for your later commands \
             and is thrown away when you finish: nothing a command writes reaches the real project, so there \
             is nothing to restore. read_file and list_files show the real project, without your changes; to \
             see a file as your commands left it, print it with a command. Run one command at a time.\n",
        );
    }
    text.push_str(&format!(
        "- Paths are relative to the project root. Paths outside the project are refused, and so are \
         environment files (.env and its variants): do not try to read those in any other way, and never \
         print credentials.\n\
         - execute_command starts a program directly, without a shell, in the working_dir you pass ('.' is \
         the project root). When you need a pipe, a redirect or '&&', run `sh` with the args \
         [\"-c\", \"<command line>\"].\n\
         - Commands get no input and are killed after {} seconds. Never start a server, a watcher or \
         anything interactive.\n\
         - Search with a command (`git grep -n`, `rg` or `grep -rn`). read_file cuts a file off after \
         100,000 bytes; for a large file, print the lines you need (`sed -n '120,200p' <file>`).\n\
         - Stay inside the project. Do not search or read the rest of the machine.\n\n\
         # Instructions written for another tool\n\n\
         Where your instructions name Bash, Read, Edit, Write, Grep or Glob, use the tool above that does \
         the same thing. There are no skills, slash commands, subagents or MCP tools here. If your \
         instructions depend on one (\"/code-review\", for example), do not imitate it: stop, and say in \
         your reply which one you needed.\n\n\
         # Reporting\n\n\
         Your final reply goes back to whoever started you. Nobody can answer a question while you work, \
         so if you cannot go on, say why in that reply.\n",
        command_timeout.as_secs()
    ));
    match (report, write_access) {
        (Some(path), true) => text.push_str(&format!(
            "Your report belongs in `{path}`. Write it there with write_file before your final reply. If \
             the file is not there when you finish, your final reply is saved to it instead.\n"
        )),
        (Some(path), false) => text.push_str(&format!(
            "Your report belongs in `{path}`, which you cannot write yourself. Make your final reply the \
             complete report, even if your instructions ask for a one-line reply: it is saved to that path \
             for you.\n"
        )),
        (None, _) => {}
    }
    text
}

/// Check a delegation and turn it into a request. Everything that can be
/// refused without calling a model is refused here, so that a caller with a
/// mistyped path hears about it at once.
pub fn prepare(delegation: Delegation) -> Result<Prepared, String> {
    let Delegation {
        directory,
        agent,
        prompt,
        model,
        write_access,
        scratch,
        report_path,
        max_turns,
        command_timeout,
        jail,
        hidden,
    } = delegation;

    if prompt.trim().is_empty() {
        return Err("the prompt is empty".to_string());
    }
    if !directory.is_absolute() {
        return Err(format!("directory {} must be an absolute path", directory.display()));
    }
    // Checked first because a sandbox creates its root, and a mistyped path
    // must not become a new directory.
    if !directory.is_dir() {
        return Err(format!("directory {} does not exist", directory.display()));
    }
    if scratch && write_access {
        return Err("scratch and write_access cannot be combined: scratch throws every change away".to_string());
    }
    if scratch && jail.is_none() {
        return Err(
            "scratch needs the jail, and this server runs commands without one (--no-jail): \
             a command's writes would reach the project"
                .to_string(),
        );
    }
    let mut sandbox = Sandbox::new(&directory, command_timeout)
        .map_err(|e| format!("could not use directory {}: {e}", directory.display()))?
        .hiding_env_files()
        .guarding_unseen_code()
        .hiding(&hidden)?;
    let tracked = tracked_hidden(sandbox.root(), &hidden);
    if !tracked.is_empty() {
        return Err(format!(
            "this server hides paths that git tracks in {}: {}. An agent would see those files as deleted \
             and could stage or report that, and hiding them protects nothing, because their content is in \
             the repository's history. Take them off --hide, or stop tracking them.",
            sandbox.root().display(),
            tracked.join(", ")
        ));
    }
    let writes = match (write_access, scratch) {
        (true, _) => Writes::Project,
        (false, true) => Writes::Scratch,
        (false, false) => Writes::Refused,
    };
    if let Some(policy) = jail.clone() {
        sandbox = sandbox.jailed(policy, writes)?;
    }
    // Commands cannot see a repository above the directory they run in, so a
    // subdirectory of a repository would leave the agent without git.
    if !sandbox.root().join(".git").exists()
        && let Some(above) = sandbox.enclosing_projects().iter().find(|marker| marker.ends_with(".git"))
    {
        return Err(format!(
            "directory {} is inside the git repository at {} but is not its root. \
             Pass the root of the repository, or of a worktree.",
            sandbox.root().display(),
            above.parent().unwrap_or(above).display()
        ));
    }
    check_credentials(&model).map_err(|e| format!("{e:#}"))?;

    let report = match report_path.as_deref().map(str::trim).filter(|path| !path.is_empty()) {
        Some(given) => Some(Report {
            path: sandbox.resolve_for_write(given).map_err(|e| format!("report_path: {e}"))?,
            given: given.to_string(),
        }),
        None => None,
    };

    let mut preamble =
        rules(write_access, scratch, report.as_ref().map(|r| r.given.as_str()), command_timeout, jail.as_ref());
    let summary = Summary {
        directory: sandbox.root().to_path_buf(),
        agent: agent.as_deref().map(str::trim).filter(|agent| !agent.is_empty()).map(str::to_string),
        model: model.clone(),
        access: match writes {
            Writes::Project => "write",
            Writes::Scratch => "scratch",
            Writes::Refused => "read",
        },
    };
    if let Some(agent) = agent.as_deref().filter(|agent| !agent.trim().is_empty()) {
        preamble.push_str("\n# Your instructions\n\n");
        preamble.push_str(&read_agent(sandbox.root(), agent)?);
        preamble.push('\n');
    }

    Ok(Prepared {
        request: AgentRequest {
            model,
            preamble,
            prompt,
            tools: if write_access { ToolAccess::Full } else { ToolAccess::Inspect },
            sandbox,
            max_turns: max_turns.max(1),
        },
        report,
        summary,
    })
}

/// Run a prepared delegation to its end.
///
/// The agent is started once. A single request that a provider turns away
/// for a passing reason is sent again further down (see `llm.rs`); an agent
/// that fails is not started again, because the second one would begin with
/// an empty context on files the first had half changed, and the caller would
/// never know. The caller decides whether to try again.
///
/// On failure nothing is written: a report file from an earlier run stays as
/// it is, and the error says that the agent may have left work half done.
pub async fn run(llm: &dyn Llm, prepared: Prepared) -> Result<Outcome, String> {
    let Prepared { request, report, summary: _ } = prepared;
    let model = request.model.clone();
    let root = request.sandbox.root().to_path_buf();
    let _stop = StopCommands(request.sandbox.clone());
    let before = report.as_ref().map(|report| std::fs::read(&report.path).ok());

    tracing::info!("Agent ({model}) starting in {}", root.display());
    let reply = llm.run(request).await.map_err(|e| {
        let cause = match e {
            LlmError::MaxTurns(turns) => format!(
                "it used all {turns} turns without a final reply (max_turns raises the limit for one call, \
                 --max-turns for the server)"
            ),
            LlmError::Other(e) => format!("{e:#}"),
        };
        format!(
            "The agent did not finish: {cause}. It wrote no report, and it may have left files in {} \
             half changed; look at the tree before running anything else on it.",
            root.display()
        )
    })?;
    tracing::info!("Agent ({model}) finished");

    let report = match report {
        None => None,
        Some(report) => {
            let after = std::fs::read(&report.path).ok();
            let written_by_agent = after.is_some() && Some(&after) != before.as_ref();
            if !written_by_agent {
                let saved = report
                    .path
                    .parent()
                    .map_or(Ok(()), std::fs::create_dir_all)
                    .and_then(|_| std::fs::write(&report.path, &reply.text));
                if let Err(e) = saved {
                    return Err(format!(
                        "The agent finished, but its report could not be saved to {}: {e}. Its reply was:\n\n{}",
                        report.given, reply.text
                    ));
                }
            }
            Some(SavedReport { path: report.given, written_by_agent })
        }
    };

    Ok(Outcome { model, reply: reply.text, usage: reply.usage, report })
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::llm::{LlmFuture, LlmReply};

    type Act = dyn Fn(&AgentRequest) -> Result<String, String> + Send + Sync;

    /// A model that does what the test says and remembers what it was asked.
    pub(crate) struct ActingLlm {
        act: Box<Act>,
        pub(crate) requests: Mutex<Vec<AgentRequest>>,
    }

    impl ActingLlm {
        pub(crate) fn new(act: impl Fn(&AgentRequest) -> Result<String, String> + Send + Sync + 'static) -> Self {
            Self { act: Box::new(act), requests: Mutex::new(Vec::new()) }
        }

        pub(crate) fn replying(reply: &'static str) -> Self {
            Self::new(move |_| Ok(reply.to_string()))
        }

        fn request(&self) -> AgentRequest {
            self.requests.lock().unwrap().last().expect("the model was called").clone()
        }
    }

    impl Llm for ActingLlm {
        fn run<'a>(&'a self, request: AgentRequest) -> LlmFuture<'a> {
            self.requests.lock().unwrap().push(request.clone());
            Box::pin(async move {
                (self.act)(&request)
                    .map(|text| LlmReply {
                        text,
                        usage: TokenUsage { input_tokens: 10, cached_input_tokens: 4, output_tokens: 2 },
                    })
                    .map_err(|e| LlmError::Other(anyhow::anyhow!(e)))
            })
        }
    }

    /// A project directory with an agent definition in it.
    pub(crate) fn project() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(AGENTS_DIR)).unwrap();
        std::fs::write(
            dir.path().join(AGENTS_DIR).join("reviewer.md"),
            "---\nname: reviewer\ndescription: \"Reviews a change\"\n---\nYou review a change you did not write.\n",
        )
        .unwrap();
        dir
    }

    pub(crate) fn delegation(dir: &Path) -> Delegation {
        Delegation {
            directory: dir.to_path_buf(),
            agent: Some("reviewer".into()),
            prompt: "- **Diff:** `git diff abc123`".into(),
            model: "ollama:test".into(),
            write_access: false,
            scratch: false,
            report_path: None,
            max_turns: 7,
            command_timeout: Duration::from_secs(90),
            jail: None,
            hidden: Vec::new(),
        }
    }

    #[test]
    fn front_matter_is_removed_and_nothing_else() {
        assert_eq!(strip_front_matter("---\nname: a\n---\nBody\n"), "Body\n");
        assert_eq!(strip_front_matter("---\r\nname: a\r\n---\r\nBody"), "Body");
        assert_eq!(strip_front_matter("No front matter\n---\nrule\n"), "No front matter\n---\nrule\n");
        assert_eq!(strip_front_matter("---\nnever closed\n"), "---\nnever closed\n");
        assert_eq!(strip_front_matter("--- not a block\n---\n"), "--- not a block\n---\n");
    }

    #[tokio::test]
    async fn the_agent_file_and_the_prompt_reach_the_model_with_the_right_tools() {
        let dir = project();
        let llm = ActingLlm::replying("done");

        let outcome = run(&llm, prepare(delegation(dir.path())).unwrap()).await.unwrap();
        assert_eq!(outcome.reply, "done");
        assert_eq!(outcome.model, "ollama:test");
        assert_eq!(outcome.usage.input_tokens, 10);
        assert_eq!(outcome.report, None);

        let request = llm.request();
        assert_eq!(request.prompt, "- **Diff:** `git diff abc123`");
        assert!(request.preamble.contains("You review a change you did not write."));
        assert!(!request.preamble.contains("description:"), "front matter is not part of the instructions");
        assert!(request.preamble.contains("killed after 90 seconds"));
        assert!(request.preamble.contains("You have no tool that writes files"));
        assert_eq!(request.tools, ToolAccess::Inspect);
        assert_eq!(request.max_turns, 7);
        assert_eq!(request.sandbox.root(), dir.path().canonicalize().unwrap());
        assert!(request.sandbox.resolve(".env.production").is_err(), "environment files are hidden");

        let mut writer = delegation(dir.path());
        writer.write_access = true;
        writer.agent = None;
        run(&llm, prepare(writer).unwrap()).await.unwrap();
        let request = llm.request();
        assert_eq!(request.tools, ToolAccess::Full);
        assert!(request.preamble.contains("edit_file"));
        assert!(!request.preamble.contains("# Your instructions"));
    }

    #[test]
    fn a_jailed_agent_is_told_so_and_gets_a_jail_with_the_hidden_paths() {
        let dir = project();
        let mut d = delegation(dir.path());
        d.hidden = vec![PathBuf::from("tmp/private")];
        let plain = prepare(d.clone()).unwrap().request;
        assert!(plain.sandbox.jail().is_none());
        assert!(!plain.preamble.contains("jail"));
        assert!(plain.sandbox.resolve("tmp/private/notes.md").is_err(), "hidden with or without a jail");
        assert!(plain.sandbox.resolve_for_write(".git/hooks/pre-commit").is_err());

        d.jail = Some(JailPolicy::default());
        let jailed = prepare(d.clone()).unwrap().request;
        assert!(jailed.sandbox.jail().is_some());
        assert!(jailed.preamble.contains("Commands run in a jail. In it the project is read-only, there is no network"));

        d.write_access = true;
        d.jail = Some(JailPolicy { network: true, ..Default::default() });
        let writer = prepare(d.clone()).unwrap().request;
        assert!(writer.preamble.contains("In it the project is writable, there is a network"));

        d.hidden = vec![PathBuf::from("../outside")];
        assert!(prepare(d).unwrap_err().contains("hidden path"));
    }

    #[test]
    fn scratch_needs_a_jail_and_excludes_write_access() {
        let dir = project();
        let mut d = delegation(dir.path());
        d.scratch = true;
        assert!(prepare(d.clone()).unwrap_err().contains("scratch needs the jail"));

        d.jail = Some(JailPolicy::default());
        d.write_access = true;
        assert!(prepare(d.clone()).unwrap_err().contains("cannot be combined"));

        if crate::tools::jail::probe().is_err() {
            return; // no bubblewrap here: the overlay cannot be tried
        }
        d.write_access = false;
        let prepared = prepare(d.clone()).unwrap();
        assert_eq!(prepared.summary().access, "scratch");
        assert_eq!(prepared.summary().agent.as_deref(), Some("reviewer"));
        assert_eq!(prepared.request.tools, ToolAccess::Inspect, "scratch gives no file-writing tools");
        assert!(prepared.request.preamble.contains("In it the project is a throwaway copy"));
        assert!(prepared.request.preamble.contains("read_file and list_files show the real project"));

        d.scratch = false;
        let plain = prepare(d).unwrap();
        assert_eq!(plain.summary().access, "read");
        assert!(!plain.request.preamble.contains("throwaway"));
    }

    #[test]
    fn a_hidden_path_that_git_tracks_is_refused_by_name() {
        let dir = project();
        let git = |args: &[&str]| {
            std::process::Command::new("git")
                .args(args)
                .current_dir(dir.path())
                .output()
                .is_ok_and(|out| out.status.success())
        };
        std::fs::create_dir_all(dir.path().join("linked")).unwrap();
        std::fs::create_dir_all(dir.path().join("scratchpad")).unwrap();
        std::fs::write(dir.path().join("linked/project.json"), "{}").unwrap();
        std::fs::write(dir.path().join("linked/url"), "x").unwrap();
        std::fs::write(dir.path().join("scratchpad/notes.md"), "untracked").unwrap();
        if !git(&["init", "-q"]) || !git(&["add", "linked"]) {
            return; // git is not installed
        }

        let mut d = delegation(dir.path());
        d.hidden = vec![PathBuf::from("scratchpad"), PathBuf::from("linked")];
        let error = prepare(d.clone()).unwrap_err();
        assert!(error.contains("linked (2 tracked files)"), "{error}");
        assert!(!error.contains("scratchpad"), "an untracked path may be hidden: {error}");

        d.hidden = vec![PathBuf::from("scratchpad"), PathBuf::from("not-there")];
        assert!(prepare(d).is_ok());
    }

    #[test]
    fn an_agent_is_found_by_name_or_by_path() {
        let dir = project();
        std::fs::create_dir_all(dir.path().join("docs")).unwrap();
        std::fs::write(dir.path().join("docs/helper.md"), "Help with the docs.\n").unwrap();
        std::fs::write(dir.path().join("empty.md"), "---\nname: empty\n---\n").unwrap();
        let with_agent = |agent: &str| {
            let mut d = delegation(dir.path());
            d.agent = Some(agent.to_string());
            prepare(d)
        };

        assert!(with_agent("docs/helper.md").unwrap().request.preamble.contains("Help with the docs."));
        let absolute = dir.path().join("docs/helper.md");
        assert!(with_agent(absolute.to_str().unwrap()).unwrap().request.preamble.contains("Help with the docs."));
        assert!(with_agent("nobody").unwrap_err().contains("could not read the agent file"));
        // Whatever is named goes to the provider whole, so it has to be an agent file.
        std::fs::write(dir.path().join(".env.production"), "KEY=secret").unwrap();
        for not_an_agent in ["./.env.production", "/etc/passwd", "docs/helper"] {
            assert!(with_agent(not_an_agent).unwrap_err().contains("Markdown"), "{not_an_agent}");
        }
        assert!(with_agent("empty.md").unwrap_err().contains("no instructions"));
    }

    #[test]
    fn what_cannot_work_is_refused_before_any_model_call() {
        let dir = project();
        let refused = |change: &dyn Fn(&mut Delegation)| {
            let mut d = delegation(dir.path());
            change(&mut d);
            prepare(d).unwrap_err()
        };

        assert!(refused(&|d| d.prompt = "  ".into()).contains("prompt is empty"));
        assert!(refused(&|d| d.directory = PathBuf::from("relative/dir")).contains("absolute"));
        let missing = dir.path().join("typo");
        assert!(refused(&|d| d.directory = missing.clone()).contains("does not exist"));
        assert!(!missing.exists(), "a mistyped directory is not created");
        assert!(refused(&|d| d.report_path = Some("../report.md".into())).contains("report_path"));
        assert!(refused(&|d| d.report_path = Some(".env.local".into())).contains("report_path"));
        // A model without credentials. (Skipped where a developer's real key is exported.)
        if std::env::var("ANTHROPIC_API_KEY").is_err() {
            assert!(refused(&|d| d.model = "claude-test".into()).contains("ANTHROPIC_API_KEY"));
        }
    }

    #[test]
    fn a_subdirectory_of_a_repository_is_refused_but_its_root_and_a_worktree_are_not() {
        let dir = project();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::create_dir_all(dir.path().join("packages/web")).unwrap();
        std::fs::create_dir_all(dir.path().join("worktrees/fix")).unwrap();
        // A worktree marks its root with a `.git` file, not a directory.
        std::fs::write(dir.path().join("worktrees/fix/.git"), "gitdir: ../../.git/worktrees/fix\n").unwrap();
        let at = |sub: &str| {
            let mut d = delegation(dir.path());
            d.directory = dir.path().join(sub);
            d.agent = None;
            prepare(d)
        };

        assert!(at("").is_ok());
        assert!(at("worktrees/fix").is_ok());
        let error = at("packages/web").unwrap_err();
        assert!(error.contains("is not its root"), "{error}");
    }

    #[tokio::test]
    async fn the_reply_is_saved_as_the_report_unless_the_agent_wrote_it() {
        let dir = project();
        let mut d = delegation(dir.path());
        d.report_path = Some("tmp/run/review-r1.md".into());
        let file = dir.path().join("tmp/run/review-r1.md");

        // A read-only agent cannot write it: its reply is the report.
        let llm = ActingLlm::replying("VERDICT: clean");
        let prepared = prepare(d.clone()).unwrap();
        assert!(prepared.request.preamble.contains("Make your final reply the complete report"));
        let outcome = run(&llm, prepared).await.unwrap();
        assert_eq!(
            outcome.report,
            Some(SavedReport { path: "tmp/run/review-r1.md".into(), written_by_agent: false })
        );
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "VERDICT: clean");

        // An agent that writes the file and replies with one line keeps its file.
        d.write_access = true;
        let llm = ActingLlm::new(|request| {
            let path = request.sandbox.resolve_for_write("tmp/run/review-r1.md").unwrap();
            std::fs::write(path, "VERDICT: 2 findings\n1. ...\n").unwrap();
            Ok("2 findings, tmp/run/review-r1.md".to_string())
        });
        let prepared = prepare(d.clone()).unwrap();
        assert!(prepared.request.preamble.contains("Write it there with write_file"));
        let outcome = run(&llm, prepared).await.unwrap();
        assert!(outcome.report.unwrap().written_by_agent);
        assert_eq!(outcome.reply, "2 findings, tmp/run/review-r1.md");
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "VERDICT: 2 findings\n1. ...\n");

        // A report left by an earlier run is replaced by this run's reply, not mistaken for it.
        let llm = ActingLlm::replying("VERDICT: clean again");
        let outcome = run(&llm, prepare(d).unwrap()).await.unwrap();
        assert!(!outcome.report.unwrap().written_by_agent);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "VERDICT: clean again");
    }

    #[tokio::test]
    async fn a_failed_run_leaves_the_report_alone_and_says_the_tree_may_be_half_changed() {
        let dir = project();
        std::fs::write(dir.path().join("report.md"), "from an earlier round").unwrap();
        let mut d = delegation(dir.path());
        d.report_path = Some("report.md".into());

        let llm = ActingLlm::new(|_| Err("401 Unauthorized".to_string()));
        let error = run(&llm, prepare(d.clone()).unwrap()).await.unwrap_err();
        assert!(error.contains("401 Unauthorized"), "{error}");
        assert!(error.contains("half changed"), "{error}");
        assert_eq!(std::fs::read_to_string(dir.path().join("report.md")).unwrap(), "from an earlier round");
        assert_eq!(llm.requests.lock().unwrap().len(), 1);

        // Not even a passing failure starts the agent again: the second agent
        // would find the first one's half-finished work and not know it.
        let llm = ActingLlm::new(|_| Err("503 Service Unavailable: overloaded".to_string()));
        let error = run(&llm, prepare(d.clone()).unwrap()).await.unwrap_err();
        assert!(error.contains("overloaded"), "{error}");
        assert_eq!(llm.requests.lock().unwrap().len(), 1, "the agent is started exactly once");

        struct OutOfTurns;
        impl Llm for OutOfTurns {
            fn run<'a>(&'a self, request: AgentRequest) -> LlmFuture<'a> {
                Box::pin(async move { Err(LlmError::MaxTurns(request.max_turns)) })
            }
        }
        let error = run(&OutOfTurns, prepare(d).unwrap()).await.unwrap_err();
        assert!(error.contains("used all 7 turns"), "{error}");
    }

    #[tokio::test]
    async fn the_runs_commands_are_stopped_when_it_ends_and_when_it_is_cancelled() {
        let dir = project();

        let llm = ActingLlm::replying("done");
        run(&llm, prepare(delegation(dir.path())).unwrap()).await.unwrap();
        assert!(llm.request().sandbox.commands_stopped());

        /// Never answers, like a model in the middle of a long run.
        struct Hanging(Mutex<Option<Sandbox>>);
        impl Llm for Hanging {
            fn run<'a>(&'a self, request: AgentRequest) -> LlmFuture<'a> {
                *self.0.lock().unwrap() = Some(request.sandbox.clone());
                Box::pin(std::future::pending())
            }
        }
        let llm = Arc::new(Hanging(Mutex::new(None)));
        let prepared = prepare(delegation(dir.path())).unwrap();
        let task = tokio::spawn({
            let llm = llm.clone();
            async move { run(llm.as_ref(), prepared).await }
        });
        let sandbox = loop {
            if let Some(sandbox) = llm.0.lock().unwrap().clone() {
                break sandbox;
            }
            tokio::task::yield_now().await;
        };
        assert!(!sandbox.commands_stopped());
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert!(sandbox.commands_stopped(), "a cancelled run does not leave a command running");
    }
}
