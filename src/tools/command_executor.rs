use std::collections::VecDeque;
use std::path::Path;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;

use crate::tools::{CommandScope, Sandbox, ToolError};

/// Per stream, the first and last this-many bytes of output are kept.
const OUTPUT_HEAD_BYTES: usize = 4_000;
const OUTPUT_TAIL_BYTES: usize = 8_000;

#[derive(Deserialize, Serialize)]
pub struct ExecuteCommandArgs {
    pub command: String,
    #[serde(default, deserialize_with = "args_list")]
    pub args: Vec<String>,
    /// Directory to run in, relative to the project root.
    #[serde(default = "project_root")]
    pub working_dir: String,
}

/// Models occasionally send `"args": "test"` instead of `["test"]`. Failing the
/// tool call over that costs a turn, so a plain string is split on whitespace.
fn args_list<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Vec<String>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Args {
        List(Vec<String>),
        Line(String),
    }
    Ok(match Option::<Args>::deserialize(deserializer)? {
        Some(Args::List(list)) => list,
        Some(Args::Line(line)) => line.split_whitespace().map(str::to_string).collect(),
        None => Vec::new(),
    })
}

fn project_root() -> String {
    ".".to_string()
}

/// cargo subcommands that do not need an existing manifest.
const CARGO_WITHOUT_MANIFEST: &[&str] =
    &["new", "init", "help", "version", "install", "uninstall", "search", "login", "logout"];

/// What a command call returns once its agent's run has been stopped.
const STOPPED: &str = "Command not run: this agent's run has been stopped.";

pub struct ExecuteCommand {
    pub sandbox: Sandbox,
}

/// Reads a stream to the end while holding on to only its head and tail.
async fn read_capped(mut stream: impl AsyncRead + Unpin) -> String {
    let mut head: Vec<u8> = Vec::new();
    let mut tail: VecDeque<u8> = VecDeque::new();
    let mut dropped: usize = 0;
    let mut buf = [0u8; 8192];
    loop {
        let n = match stream.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        for &byte in &buf[..n] {
            if head.len() < OUTPUT_HEAD_BYTES {
                head.push(byte);
            } else {
                if tail.len() == OUTPUT_TAIL_BYTES {
                    tail.pop_front();
                    dropped += 1;
                }
                tail.push_back(byte);
            }
        }
    }
    let mut out = String::from_utf8_lossy(&head).into_owned();
    if dropped > 0 {
        out.push_str(&format!("\n[... {dropped} bytes of output omitted ...]\n"));
    }
    let tail: Vec<u8> = tail.into();
    out.push_str(&String::from_utf8_lossy(&tail));
    out
}

/// Kill the command's whole process group, so that e.g. the binary started by
/// `cargo run` dies along with cargo.
#[cfg(unix)]
pub(crate) fn kill_process_group(pid: Option<u32>) {
    if let Some(pid) = pid.and_then(|p| i32::try_from(p).ok()) {
        // SAFETY: plain syscall; the pid is the id of a group we created with process_group(0).
        unsafe {
            libc::killpg(pid, libc::SIGKILL);
        }
    }
}

#[cfg(not(unix))]
pub(crate) fn kill_process_group(_pid: Option<u32>) {}

/// Process groups of commands that are running right now.
static RUNNING: Mutex<Vec<u32>> = Mutex::new(Vec::new());

/// Kill every command an agent currently has running, including what those
/// commands started. Called when the program is interrupted or quits, because
/// the commands live in their own process groups and would otherwise survive.
pub fn kill_running_commands() {
    let pids: Vec<u32> = RUNNING.lock().map(|mut running| running.drain(..).collect()).unwrap_or_default();
    for pid in pids {
        kill_process_group(Some(pid));
    }
}

/// Registers a running command, with the program and with its sandbox, and
/// kills its group when dropped, which also covers the future running the
/// command being dropped mid-command.
struct RunningCommand(Option<u32>, Arc<CommandScope>);

impl RunningCommand {
    fn new(pid: Option<u32>, sandbox: &Sandbox) -> Self {
        let scope = sandbox.commands().clone();
        if let Some(pid) = pid {
            if let Ok(mut running) = RUNNING.lock() {
                running.push(pid);
            }
            if let Ok(mut pids) = scope.pids.lock() {
                pids.push(pid);
            }
        }
        Self(pid, scope)
    }
}

impl Drop for RunningCommand {
    fn drop(&mut self) {
        let Some(pid) = self.0 else { return };
        if let Ok(mut pids) = self.1.pids.lock() {
            pids.retain(|p| *p != pid);
        }
        // Only kill what is still registered; kill_running_commands may have been here already.
        let registered = RUNNING
            .lock()
            .map(|mut running| {
                let before = running.len();
                running.retain(|p| *p != pid);
                running.len() != before
            })
            .unwrap_or(false);
        if registered {
            kill_process_group(Some(pid));
        }
    }
}

impl ExecuteCommand {
    /// cargo looks for `Cargo.toml` in every parent directory. When the output
    /// directory sits inside another Rust project (this repository, say), a
    /// `cargo test` with no manifest of its own would silently build and test
    /// that outer project and report its results. Refuse instead, and tell the
    /// model how to fix it.
    fn cargo_outside_project(&self, args: &ExecuteCommandArgs, cwd: &Path) -> Option<String> {
        let program = Path::new(&args.command).file_name()?.to_str()?;
        if program != "cargo" {
            return None;
        }
        let subcommand = args.args.iter().find(|a| !a.starts_with('-') && !a.starts_with('+'))?;
        if CARGO_WITHOUT_MANIFEST.contains(&subcommand.as_str())
            || args.args.iter().any(|a| a.starts_with("--manifest-path"))
        {
            return None;
        }

        let root = self.sandbox.root();
        let mut dir = cwd;
        loop {
            if dir.join("Cargo.toml").is_file() {
                return None;
            }
            if dir == root {
                break;
            }
            dir = dir.parent()?;
        }

        let nested: Vec<String> = std::fs::read_dir(root)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|entry| entry.path().join("Cargo.toml").is_file())
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect();
        let hint = if nested.is_empty() {
            "Create the project in the project root with `cargo init` (not `cargo new <name>`, which makes a subdirectory).".to_string()
        } else {
            format!(
                "A Cargo.toml exists in the subdirectory {}: the project was created one level too deep. \
                 Either move its files up into the project root, or run cargo there by setting working_dir.",
                nested.join(", ")
            )
        };
        Some(format!(
            "Command not run: there is no Cargo.toml in '{}' or above it inside the project, \
             so cargo would have picked up an unrelated project outside. {hint}",
            args.working_dir
        ))
    }

    fn build_command(&self, args: &ExecuteCommandArgs, cwd: &Path) -> Command {
        let mut cmd = match self.sandbox.jail() {
            // The jail starts from an empty environment, so it has no credentials to remove.
            Some(jail) => jail.command(self.sandbox.root(), self.sandbox.hidden(), &args.command, &args.args, cwd),
            None => {
                let mut cmd = Command::new(&args.command);
                cmd.args(&args.args).current_dir(cwd);
                // Credentials that generated code and its build scripts have no business seeing.
                for key in crate::llm::secret_env_vars() {
                    cmd.env_remove(key);
                }
                cmd
            }
        };
        cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);
        // Keep git from discovering a repository that encloses the output
        // directory; otherwise an agent's `git add`/`git commit` lands there.
        if let Some(parent) = self.sandbox.root().parent() {
            cmd.env("GIT_CEILING_DIRECTORIES", parent);
        }
        #[cfg(unix)]
        cmd.process_group(0);
        cmd
    }

    async fn run(&self, args: &ExecuteCommandArgs) -> String {
        let timeout = self.sandbox.command_timeout;
        let cwd = match self.sandbox.resolve(&args.working_dir) {
            Ok(dir) if dir.is_dir() => dir,
            Ok(_) => return format!("Command not run: working_dir '{}' is not an existing directory.", args.working_dir),
            Err(e) => return format!("Command not run: {e}"),
        };
        if let Some(refusal) = self.cargo_outside_project(args, &cwd) {
            return refusal;
        }
        if self.sandbox.commands_stopped() {
            return STOPPED.to_string();
        }
        // In a scratch run, commands share one overlay and take turns.
        let _turn = match self.sandbox.jail().and_then(|jail| jail.turn()) {
            Some(turn) => Some(turn.lock().await),
            None => None,
        };
        if self.sandbox.commands_stopped() {
            return STOPPED.to_string();
        }
        let mut cmd = self.build_command(args, &cwd);

        let mut child = match cmd.spawn() {
            Ok(child) => child,
            Err(e) => {
                return format!("Failed to start command {} {:?}: {}", args.command, args.args, e);
            }
        };
        let pid = child.id();
        let running = RunningCommand::new(pid, &self.sandbox);
        // Stopped between the check above and the registration: nobody else will kill it.
        if self.sandbox.commands_stopped() {
            kill_process_group(pid);
        }
        let stdout = tokio::spawn(read_capped(child.stdout.take().expect("stdout is piped")));
        let stderr = tokio::spawn(read_capped(child.stderr.take().expect("stderr is piped")));

        let status = tokio::time::timeout(timeout, child.wait()).await;
        // Whether it exited or timed out, make sure nothing it started lingers on
        // holding the output pipes open.
        drop(running);
        if status.is_err() {
            let _ = child.kill().await;
        }

        let collect = |handle: tokio::task::JoinHandle<String>| async move {
            match tokio::time::timeout(Duration::from_secs(5), handle).await {
                Ok(Ok(text)) => text,
                _ => "[output could not be collected]".to_string(),
            }
        };
        let stdout = collect(stdout).await;
        let stderr = collect(stderr).await;

        let headline = match status {
            Ok(Ok(status)) if status.success() => "Command succeeded.".to_string(),
            Ok(Ok(status)) => match status.code() {
                Some(code) => format!("Command failed with exit code {code}."),
                None => "Command was terminated by a signal.".to_string(),
            },
            Ok(Err(e)) => format!("Failed to wait for command: {e}"),
            Err(_) => format!(
                "Command timed out after {} seconds and was killed. Commands must finish on their own: \
                 do not start interactive programs, TUIs, servers or watchers. Use build and test commands instead.",
                timeout.as_secs()
            ),
        };
        format!("{headline}\nStdout:\n{stdout}\nStderr:\n{stderr}")
    }
}

impl Tool for ExecuteCommand {
    const NAME: &'static str = "execute_command";
    type Error = ToolError;
    type Args = ExecuteCommandArgs;
    type Output = String;

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        tracing::info!("Executing command in '{}': {} {:?}", args.working_dir, args.command, args.args);
        Ok(self.run(&args).await)
    }

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: format!(
                "Run a program inside the project and return its exit status and output. \
                 There is no shell: pass the program and its arguments separately (no pipes, redirects or '&&'). \
                 The command gets no input and is killed after {} seconds, so never start anything interactive \
                 or long-running (no TUIs, servers or watchers). Use it to build and test, e.g. command 'cargo', args ['test']. \
                 Keep to the project directory: do not search or read the rest of the machine.",
                self.sandbox.command_timeout.as_secs()
            ),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The program to run (e.g. 'cargo')."
                    },
                    "args": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Arguments for the program (e.g. ['check'])."
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Directory to run in, relative to the project root. Use '.' for the project root."
                    }
                },
                "required": ["command", "args", "working_dir"]
            }),
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::time::Instant;

    fn tool(timeout: Duration) -> (tempfile::TempDir, ExecuteCommand) {
        let dir = tempfile::tempdir().unwrap();
        let sandbox = Sandbox::new(dir.path().join("out"), timeout).unwrap();
        (dir, ExecuteCommand { sandbox })
    }

    fn args(command: &str, args: &[&str]) -> ExecuteCommandArgs {
        ExecuteCommandArgs {
            command: command.into(),
            args: args.iter().map(|a| a.to_string()).collect(),
            working_dir: ".".into(),
        }
    }

    #[test]
    fn args_may_arrive_as_a_list_a_string_or_not_at_all() {
        let parse = |json: &str| serde_json::from_str::<ExecuteCommandArgs>(json).unwrap().args;
        assert_eq!(parse(r#"{"command":"cargo","args":["test","--","--nocapture"]}"#), ["test", "--", "--nocapture"]);
        // Seen live: `"args": "test"` made the tool call fail.
        assert_eq!(parse(r#"{"command":"cargo","args":"test"}"#), ["test"]);
        assert_eq!(parse(r#"{"command":"cargo","args":"build  --release"}"#), ["build", "--release"]);
        assert!(parse(r#"{"command":"ls","args":null}"#).is_empty());
        assert!(parse(r#"{"command":"ls"}"#).is_empty());
        assert_eq!(serde_json::from_str::<ExecuteCommandArgs>(r#"{"command":"ls"}"#).unwrap().working_dir, ".");
    }

    #[tokio::test]
    async fn runs_in_the_sandbox_root() {
        let (_dir, tool) = tool(Duration::from_secs(10));
        let out = tool.run(&args("pwd", &[])).await;
        assert!(out.starts_with("Command succeeded."), "{out}");
        assert!(out.contains(tool.sandbox.root().to_str().unwrap()), "{out}");
    }

    #[tokio::test]
    async fn working_dir_selects_a_subdirectory_inside_the_sandbox() {
        let (_dir, tool) = tool(Duration::from_secs(10));
        std::fs::create_dir_all(tool.sandbox.root().join("backend")).unwrap();

        let mut pwd = args("pwd", &[]);
        pwd.working_dir = "backend".into();
        let out = tool.run(&pwd).await;
        assert!(out.contains(tool.sandbox.root().join("backend").to_str().unwrap()), "{out}");

        for bad in ["..", "/tmp", "missing-dir"] {
            pwd.working_dir = bad.into();
            let out = tool.run(&pwd).await;
            assert!(out.starts_with("Command not run"), "{bad}: {out}");
        }
    }

    #[tokio::test]
    async fn cargo_never_falls_through_to_a_project_outside_the_sandbox() {
        // The temp dir plays the part of an enclosing Rust project.
        let (dir, tool) = tool(Duration::from_secs(10));
        std::fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"outer\"\n").unwrap();
        let root = tool.sandbox.root().to_path_buf();

        let out = tool.run(&args("cargo", &["test"])).await;
        assert!(out.starts_with("Command not run"), "{out}");
        assert!(out.contains("cargo init"), "{out}");

        // The situation from a real run: `cargo new sumcli` made a nested project.
        std::fs::create_dir_all(root.join("sumcli/src")).unwrap();
        std::fs::write(root.join("sumcli/Cargo.toml"), "").unwrap();
        let out = tool.run(&args("cargo", &["+stable", "build", "--release"])).await;
        assert!(out.starts_with("Command not run"), "{out}");
        assert!(out.contains("subdirectory sumcli"), "{out}");

        // Allowed: a manifest in the working directory or between it and the root...
        let mut nested = args("cargo", &["test"]);
        nested.working_dir = "sumcli/src".into();
        assert!(tool.cargo_outside_project(&nested, &root.join("sumcli/src")).is_none());
        // ...an explicit manifest, commands that need none, and other programs.
        for ok in [
            args("cargo", &["build", "--manifest-path", "sumcli/Cargo.toml"]),
            args("cargo", &["init", "--name", "x"]),
            args("cargo", &["--version"]),
            args("ls", &["test"]),
        ] {
            assert!(tool.cargo_outside_project(&ok, &root).is_none(), "{:?}", ok.args);
        }
        std::fs::write(root.join("Cargo.toml"), "").unwrap();
        assert!(tool.cargo_outside_project(&args("cargo", &["test"]), &root).is_none());
    }

    #[tokio::test]
    async fn git_does_not_see_a_repository_enclosing_the_sandbox() {
        let (dir, tool) = tool(Duration::from_secs(10));
        if std::process::Command::new("git").arg("init").arg("-q").arg(dir.path()).status().is_err() {
            return; // git is not installed
        }
        let out = tool.run(&args("git", &["rev-parse", "--show-toplevel"])).await;
        assert!(out.contains("failed"), "git must not find the outer repository: {out}");
        assert!(!out.contains(&format!("Stdout:\n{}", dir.path().display())), "{out}");
    }

    #[tokio::test]
    async fn reports_failure_and_missing_programs() {
        let (_dir, tool) = tool(Duration::from_secs(10));
        let out = tool.run(&args("sh", &["-c", "echo oops >&2; exit 3"])).await;
        assert!(out.contains("exit code 3"), "{out}");
        assert!(out.contains("oops"), "{out}");

        let out = tool.run(&args("definitely-not-a-real-program-xyz", &[])).await;
        assert!(out.starts_with("Failed to start command"), "{out}");
    }

    #[tokio::test]
    async fn hanging_commands_are_killed_with_their_children() {
        let (_dir, tool) = tool(Duration::from_secs(1));
        let started = Instant::now();
        // The grandchild keeps the pipes open; only killing the group ends it.
        let out = tool.run(&args("sh", &["-c", "echo started; sleep 60 & sleep 60"])).await;
        assert!(started.elapsed() < Duration::from_secs(10), "took {:?}", started.elapsed());
        assert!(out.contains("timed out after 1 seconds"), "{out}");
        assert!(out.contains("started"), "partial output should be kept: {out}");
    }

    #[tokio::test]
    async fn stdin_is_closed() {
        let (_dir, tool) = tool(Duration::from_secs(10));
        let out = tool.run(&args("cat", &[])).await;
        assert!(out.starts_with("Command succeeded."), "{out}");
    }

    #[tokio::test]
    async fn output_is_capped() {
        let (_dir, tool) = tool(Duration::from_secs(20));
        let out = tool.run(&args("sh", &["-c", "head -c 500000 /dev/zero | tr '\\0' 'a'; echo THE-END"])).await;
        assert!(out.len() < 20_000, "output was {} bytes", out.len());
        assert!(out.contains("bytes of output omitted"), "{}", &out[..200]);
        assert!(out.contains("THE-END"));
    }

    #[test]
    fn api_keys_are_not_passed_to_commands() {
        let (_dir, tool) = tool(Duration::from_secs(10));
        let cmd = tool.build_command(&args("env", &[]), tool.sandbox.root());
        let removed: Vec<_> = cmd
            .as_std()
            .get_envs()
            .filter(|(_, value)| value.is_none())
            .map(|(key, _)| key.to_string_lossy().into_owned())
            .collect();
        let secrets = crate::llm::secret_env_vars();
        assert!(secrets.contains(&"OPENAI_API_KEY") && secrets.contains(&"OPENROUTER_KEY"));
        for key in secrets {
            assert!(removed.iter().any(|k| k == key), "{key} must be removed from the command's environment");
        }
        assert_eq!(cmd.as_std().get_current_dir(), Some(tool.sandbox.root()));
    }

    #[tokio::test]
    async fn commands_die_when_the_program_shuts_down_or_the_agent_is_cancelled() {
        let (dir, tool) = tool(Duration::from_secs(60));
        let marker = dir.path().join("out/pid");
        let script = "sleep 60 & echo $! > pid; wait";

        // Cancelled agent: the future running the command is dropped.
        let call = tokio::spawn(async move { tool.run(&args("sh", &["-c", script])).await });
        let grandchild = loop {
            if let Ok(text) = std::fs::read_to_string(&marker)
                && let Ok(pid) = text.trim().parse::<i32>()
            {
                break pid;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        };
        // SAFETY: signal 0 only checks that the process exists.
        let alive = |pid: i32| unsafe { libc::kill(pid, 0) == 0 };
        assert!(alive(grandchild));
        call.abort();
        let _ = call.await;
        for _ in 0..100 {
            if !alive(grandchild) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(!alive(grandchild), "the process started by the command must not outlive it");
        assert!(RUNNING.lock().unwrap().iter().all(|p| *p as i32 != grandchild));
    }

    /// rig runs a tool call in a task of its own, so the command is not
    /// dropped with the agent: it has to be stopped through its sandbox.
    #[tokio::test]
    async fn stopping_a_sandbox_kills_its_commands_and_refuses_new_ones() {
        let (dir, tool) = tool(Duration::from_secs(60));
        let sandbox = tool.sandbox.clone();
        let (_other_dir, other) = self::tool(Duration::from_secs(60));
        let marker = dir.path().join("out/pid");

        let call = tokio::spawn(async move {
            let output = tool.run(&args("sh", &["-c", "sleep 60 & echo $! > pid; wait"])).await;
            (tool, output)
        });
        let grandchild = loop {
            if let Ok(text) = std::fs::read_to_string(&marker)
                && let Ok(pid) = text.trim().parse::<i32>()
            {
                break pid;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        };
        // SAFETY: signal 0 only checks that the process exists.
        let alive = |pid: i32| unsafe { libc::kill(pid, 0) == 0 };
        assert!(alive(grandchild));

        sandbox.stop_commands();
        let (tool, output) = tokio::time::timeout(Duration::from_secs(10), call)
            .await
            .expect("the command ends when its sandbox is stopped, long before its timeout")
            .unwrap();
        assert!(output.contains("terminated by a signal"), "{output}");
        for _ in 0..100 {
            if !alive(grandchild) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(!alive(grandchild), "what the command started dies with it");

        assert_eq!(tool.run(&args("echo", &["hi"])).await, STOPPED);
        assert!(
            other.run(&args("echo", &["hi"])).await.contains("Command succeeded"),
            "another agent's sandbox is not affected"
        );
    }

    /// A jailed tool for a project under a pretend home directory:
    /// `<dir>/home/dev/project`. `None` where bubblewrap cannot build a jail.
    fn jailed(
        writes: crate::tools::Writes,
        timeout: Duration,
        hidden: &[&str],
        readable: &[&str],
    ) -> Option<(tempfile::TempDir, ExecuteCommand)> {
        if let Err(why) = crate::tools::jail::probe() {
            eprintln!("skipped: {why}");
            return None;
        }
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().canonicalize().unwrap().join("home");
        let root = home.join("dev/project");
        for sub in [".git/hooks", ".claude", "private", "nested"] {
            std::fs::create_dir_all(root.join(sub)).unwrap();
        }
        std::fs::create_dir_all(home.join("tools")).unwrap();
        for (file, text) in [
            (home.join("secret.txt"), "HOME-SECRET"),
            (home.join("tools/hello"), "TOOLCHAIN"),
            (root.join("source.txt"), "SOURCE"),
            (root.join(".env.local"), "KEY=ROOT-ENV"),
            (root.join("nested/.env.production"), "KEY=NESTED-ENV"),
            (root.join("private/notes.txt"), "PRIVATE-NOTES"),
            (root.join(".git/config"), "[core]\n"),
        ] {
            std::fs::write(file, text).unwrap();
        }
        let policy = crate::tools::JailPolicy {
            network: false,
            readable: readable.iter().map(|path| home.join(path)).collect(),
            home: Some(home),
        };
        let hidden: Vec<std::path::PathBuf> = hidden.iter().map(std::path::PathBuf::from).collect();
        let sandbox =
            Sandbox::new(&root, timeout).unwrap().hiding(&hidden).unwrap().jailed(policy, writes).unwrap();
        Some((dir, ExecuteCommand { sandbox }))
    }

    use crate::tools::Writes;

    fn sh(script: &str) -> ExecuteCommandArgs {
        args("sh", &["-c", script])
    }

    #[tokio::test]
    async fn a_jailed_command_sees_the_project_and_nothing_else_of_the_users() {
        let Some((_dir, tool)) = jailed(Writes::Refused, Duration::from_secs(20), &["private"], &["tools"]) else { return };

        let out = tool.run(&sh("cat source.txt && cat ../../tools/hello")).await;
        assert!(out.contains("SOURCE") && out.contains("TOOLCHAIN"), "{out}");

        // The home directory is empty, apart from what was allowed and the way to the project.
        let out = tool.run(&sh("cat ../../secret.txt 2>/dev/null; echo HOME=$(ls ../.. | tr '\\n' ' ')")).await;
        assert!(!out.contains("HOME-SECRET"), "{out}");
        assert!(out.contains("HOME=dev tools\n"), "{out}");

        // Environment files, wherever they are, and hidden paths.
        let script = "cat .env.local nested/.env.production private/notes.txt; grep -rn KEY . ; echo PRIVATE=[$(ls -A private)]";
        let out = tool.run(&sh(script)).await;
        for secret in ["ROOT-ENV", "NESTED-ENV", "PRIVATE-NOTES"] {
            assert!(!out.contains(secret), "{secret}: {out}");
        }
        assert!(out.contains("PRIVATE=[]"), "{out}");

        // Read-only, and nothing of this process: not its environment, its /proc entry or its network.
        let out = tool.run(&sh("echo x > new.txt")).await;
        assert!(out.contains("Read-only file system"), "{out}");
        assert!(!tool.sandbox.root().join("new.txt").exists());
        let out = tool.run(&sh("env; echo PIDS=$(ls /proc | grep -c '^[0-9]'); echo NICS=$(grep -c : /proc/net/dev)")).await;
        assert!(!out.contains("CARGO_PKG_NAME"), "the environment of this test must not leak in: {out}");
        assert!(out.contains("NICS=1"), "only the loopback interface: {out}");
        let pids: usize = out.split("PIDS=").nth(1).unwrap().lines().next().unwrap().parse().unwrap();
        assert!(pids < 10, "a private process namespace: {out}");
    }

    #[tokio::test]
    async fn a_writable_jail_lets_commands_change_the_project_but_not_what_runs_unseen() {
        let Some((dir, tool)) = jailed(Writes::Project, Duration::from_secs(20), &[], &[]) else { return };
        let root = tool.sandbox.root().to_path_buf();

        let out = tool.run(&sh("echo changed > source.txt && echo new > new.txt && mkdir -p .git/objects/ab")).await;
        assert!(out.starts_with("Command succeeded."), "{out}");
        assert_eq!(std::fs::read_to_string(root.join("new.txt")).unwrap(), "new\n");
        assert_eq!(std::fs::read_to_string(root.join("source.txt")).unwrap(), "changed\n");

        for script in [
            "echo evil > .git/hooks/pre-commit",
            "echo evil >> .git/config",
            "echo evil > .claude/settings.local.json",
        ] {
            let out = tool.run(&sh(script)).await;
            assert!(out.contains("Read-only file system"), "{script}: {out}");
        }
        // Nor by moving the directory above them aside and putting a writable one in its place.
        for script in [
            "mv .git .git-aside && mkdir -p .git/hooks && echo evil > .git/hooks/pre-commit",
            "mv .claude .claude-aside && mkdir .claude && echo evil > .claude/settings.local.json",
            "rm -rf .git/hooks && mkdir .git/hooks && echo evil > .git/hooks/pre-commit",
        ] {
            let out = tool.run(&sh(script)).await;
            assert!(out.starts_with("Command failed"), "{script}: {out}");
        }
        assert!(!root.join(".git-aside").exists() && !root.join(".claude-aside").exists());
        assert!(!root.join(".git/hooks/pre-commit").exists());
        assert!(!root.join(".claude/settings.local.json").exists());
        assert_eq!(std::fs::read_to_string(root.join(".git/config")).unwrap(), "[core]\n");

        // The same, where the local settings file is already there: it is covered itself.
        std::fs::write(root.join(".claude/settings.local.json"), "{}").unwrap();
        let script = "echo evil > .claude/settings.local.json; mv .claude .claude-aside; echo ok > .claude/agent.md";
        let out = tool.run(&sh(script)).await;
        assert!(out.contains("Read-only file system"), "{out}");
        assert_eq!(std::fs::read_to_string(root.join(".claude/settings.local.json")).unwrap(), "{}");
        assert!(!root.join(".claude-aside").exists());
        assert!(root.join(".claude/agent.md").exists(), "the rest of .claude can still be changed");

        // Still no way to the rest of the home directory, to read or to write.
        let out = tool.run(&sh("cat ../../secret.txt; echo x > ../../planted.txt; cat .env.local")).await;
        assert!(!out.contains("HOME-SECRET") && !out.contains("ROOT-ENV"), "{out}");
        assert!(!dir.path().join("home/planted.txt").exists());
    }

    /// Whether the loop a jailed command started is still appending to `beat`.
    async fn still_beating(beat: &std::path::Path) -> bool {
        let size = || std::fs::metadata(beat).map(|meta| meta.len()).unwrap_or(0);
        let before = size();
        tokio::time::sleep(Duration::from_millis(400)).await;
        size() != before
    }

    const BEAT: &str = "while :; do echo x >> beat; sleep 0.05; done";

    /// Processes in the jail leave our process group (`--new-session`) and
    /// have pids of their own namespace, so the kill has to reach them
    /// through bubblewrap.
    #[tokio::test]
    async fn a_jailed_command_that_times_out_dies_with_everything_it_started() {
        let Some((_dir, tool)) = jailed(Writes::Project, Duration::from_secs(1), &[], &[]) else { return };
        let beat = tool.sandbox.root().join("beat");
        let out = tool.run(&sh(&format!("({BEAT}) & {BEAT}"))).await;
        assert!(out.contains("timed out after 1 seconds"), "{out}");
        assert!(beat.exists(), "the command ran");
        assert!(!still_beating(&beat).await, "nothing the command started is left running");
    }

    #[tokio::test]
    async fn stopping_a_sandbox_kills_its_jailed_commands() {
        let Some((_dir, tool)) = jailed(Writes::Project, Duration::from_secs(60), &[], &[]) else { return };
        let sandbox = tool.sandbox.clone();
        let beat = sandbox.root().join("beat");
        let call = tokio::spawn(async move { tool.run(&sh(&format!("({BEAT}) & {BEAT}"))).await });
        while !beat.exists() {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(still_beating(&beat).await);

        sandbox.stop_commands();
        tokio::time::timeout(Duration::from_secs(10), call).await.expect("the command ends when stopped").unwrap();
        assert!(!still_beating(&beat).await, "nothing the command started is left running");
    }

    #[tokio::test]
    async fn git_works_in_a_jailed_worktree() {
        if crate::tools::jail::probe().is_err() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().canonicalize().unwrap();
        let git = |cwd: &std::path::Path, args: &[&str]| {
            std::process::Command::new("git")
                .args(["-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "init.defaultBranch=main"])
                .args(args)
                .current_dir(cwd)
                .output()
                .is_ok_and(|out| out.status.success())
        };
        std::fs::create_dir_all(base.join("main")).unwrap();
        if !git(&base.join("main"), &["init", "-q"])
            || !git(&base.join("main"), &["commit", "-q", "--allow-empty", "-m", "first commit"])
            || !git(&base.join("main"), &["worktree", "add", "-q", "../tree"])
        {
            return; // git is not installed, or too old for worktrees
        }
        let policy = crate::tools::JailPolicy { home: Some(base.join("home")), ..Default::default() };
        let sandbox = Sandbox::new(base.join("tree"), Duration::from_secs(20)).unwrap().jailed(policy, Writes::Refused).unwrap();
        let out = ExecuteCommand { sandbox }.run(&args("git", &["log", "--oneline"])).await;
        assert!(out.starts_with("Command succeeded.") && out.contains("first commit"), "{out}");
    }

    #[test]
    fn a_project_that_contains_the_home_directory_cannot_be_jailed() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let policy = crate::tools::JailPolicy { home: Some(root.join("me")), ..Default::default() };
        let error = Sandbox::new(&root, Duration::from_secs(1)).unwrap().jailed(policy, Writes::Refused).unwrap_err();
        assert!(error.contains("home directory"), "{error}");
    }

    #[tokio::test]
    async fn scratch_writes_are_seen_by_later_commands_and_never_reach_the_project() {
        let Some((dir, tool)) = jailed(Writes::Scratch, Duration::from_secs(20), &["private"], &[]) else { return };
        let root = tool.sandbox.root().to_path_buf();
        let scratch = tool.sandbox.jail().unwrap().run_dir().to_path_buf();
        assert!(scratch.join("upper").is_dir());
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&scratch).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o700, "the layer holds copies of project files: only its owner may read it");
        }

        let script = "echo mutated > source.txt && echo new > new.txt && rm nested/.env.production; \
                      echo evil > .git/hooks/pre-commit; echo x > ../../planted.txt; echo done";
        let out = tool.run(&sh(script)).await;
        assert!(out.contains("done"), "{out}");
        // A later command works on what the first one left.
        let out = tool.run(&sh("cat source.txt new.txt .git/hooks/pre-commit")).await;
        assert!(out.contains("mutated") && out.contains("new") && out.contains("evil"), "{out}");
        // What is hidden from a read-only run is hidden here too.
        let out = tool.run(&sh("cat .env.local private/notes.txt ../../secret.txt; echo PRIVATE=[$(ls -A private)]")).await;
        for secret in ["ROOT-ENV", "PRIVATE-NOTES", "HOME-SECRET"] {
            assert!(!out.contains(secret), "{secret}: {out}");
        }
        assert!(out.contains("PRIVATE=[]"), "{out}");

        // The project itself is as it was.
        assert_eq!(std::fs::read_to_string(root.join("source.txt")).unwrap(), "SOURCE");
        assert_eq!(std::fs::read_to_string(root.join("nested/.env.production")).unwrap(), "KEY=NESTED-ENV");
        assert!(!root.join("new.txt").exists());
        assert!(!root.join(".git/hooks/pre-commit").exists());
        assert!(!dir.path().join("home/planted.txt").exists());

        // The layer goes when the run says so, unreadable overlay work directory included.
        tool.sandbox.discard_run_files();
        assert!(!scratch.exists(), "the scratch directory is deleted");
    }

    /// Every command is a jail of its own, so an empty /tmp per command would
    /// forget the notes and snapshots an agent keeps between its steps.
    #[tokio::test]
    async fn tmp_lasts_from_one_command_to_the_next_and_goes_with_the_run() {
        for writes in [Writes::Refused, Writes::Project, Writes::Scratch] {
            let Some((dir, tool)) = jailed(writes, Duration::from_secs(20), &[], &[]) else { return };
            let run_dir = tool.sandbox.jail().unwrap().run_dir().to_path_buf();
            let out = tool.run(&sh("echo snapshot > /tmp/note && mkdir -p /tmp/deep/er && echo ok")).await;
            assert!(out.contains("ok"), "{writes:?}: {out}");
            let out = tool.run(&sh("cat /tmp/note; ls /tmp | tr '\\n' ' '")).await;
            assert!(out.contains("snapshot"), "{writes:?}: {out}");
            // It is this run's alone: not the machine's /tmp, and not the project.
            assert!(!out.contains("clankistrator-run-"), "the run's own directory is not visible from inside: {out}");
            assert!(run_dir.join("tmp/note").exists());
            assert!(!dir.path().join("home/dev/project/note").exists());

            tool.sandbox.discard_run_files();
            assert!(!run_dir.exists(), "{writes:?}: deleted with the run");
        }
    }

    #[tokio::test]
    async fn scratch_commands_take_turns() {
        let Some((_dir, tool)) = jailed(Writes::Scratch, Duration::from_secs(30), &[], &[]) else { return };
        let tool = std::sync::Arc::new(tool);
        let step = |name: &'static str| {
            let tool = tool.clone();
            tokio::spawn(async move {
                tool.run(&sh(&format!("echo {name}-start >> order; sleep 0.4; echo {name}-end >> order"))).await
            })
        };
        let (first, second) = (step("a"), step("b"));
        for out in [first.await.unwrap(), second.await.unwrap()] {
            assert!(out.starts_with("Command succeeded."), "{out}");
        }
        let out = tool.run(&sh("tr '\\n' ' ' < order")).await;
        assert!(
            out.contains("a-start a-end b-start b-end") || out.contains("b-start b-end a-start a-end"),
            "one command finished before the other began: {out}"
        );
    }

    #[tokio::test]
    async fn a_scratch_layer_is_deleted_after_a_timeout_and_when_the_sandbox_goes() {
        let Some((_dir, tool)) = jailed(Writes::Scratch, Duration::from_secs(1), &[], &[]) else { return };
        let root = tool.sandbox.root().to_path_buf();
        let scratch = tool.sandbox.jail().unwrap().run_dir().to_path_buf();
        let out = tool.run(&sh(&format!("({BEAT}) & {BEAT}"))).await;
        assert!(out.contains("timed out after 1 seconds"), "{out}");
        assert!(scratch.join("upper/beat").exists(), "the command wrote into the layer");
        assert!(!still_beating(&scratch.join("upper/beat")).await, "nothing is left running");
        assert!(!root.join("beat").exists());

        // Nobody called discard_run_files: dropping the last sandbox is enough.
        drop(tool);
        assert!(!scratch.exists(), "the scratch directory is deleted with the sandbox");
    }

    /// What a cancelled run does: stop the commands, then delete the layer at
    /// once, while the command that was writing into it is still dying.
    #[tokio::test]
    async fn a_scratch_layer_is_deleted_under_a_command_that_is_still_running() {
        let Some((_dir, tool)) = jailed(Writes::Scratch, Duration::from_secs(60), &[], &[]) else { return };
        let sandbox = tool.sandbox.clone();
        let scratch = sandbox.jail().unwrap().run_dir().to_path_buf();
        let beat = scratch.join("upper/beat");
        let call = tokio::spawn(async move { tool.run(&sh(&format!("({BEAT}) & {BEAT}"))).await });
        while !beat.exists() {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        sandbox.stop_commands();
        sandbox.discard_run_files();
        tokio::time::timeout(Duration::from_secs(10), call).await.expect("the command ends when stopped").unwrap();
        tokio::time::sleep(Duration::from_secs(1)).await;
        assert!(!scratch.exists(), "the layer does not come back, and nothing is left writing into it");
        assert!(!sandbox.root().join("beat").exists());
    }

    #[tokio::test]
    async fn a_scratch_run_in_a_worktree_cannot_write_to_the_repository_it_belongs_to() {
        if crate::tools::jail::probe().is_err() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().canonicalize().unwrap();
        let git = |cwd: &std::path::Path, args: &[&str]| {
            std::process::Command::new("git")
                .args(["-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "init.defaultBranch=main"])
                .args(args)
                .current_dir(cwd)
                .output()
                .is_ok_and(|out| out.status.success())
        };
        std::fs::create_dir_all(base.join("main")).unwrap();
        if !git(&base.join("main"), &["init", "-q"])
            || !git(&base.join("main"), &["commit", "-q", "--allow-empty", "-m", "first commit"])
            || !git(&base.join("main"), &["worktree", "add", "-q", "../tree"])
        {
            return; // git is not installed, or too old for worktrees
        }
        let policy = crate::tools::JailPolicy { home: Some(base.join("home")), ..Default::default() };
        let sandbox =
            Sandbox::new(base.join("tree"), Duration::from_secs(20)).unwrap().jailed(policy, Writes::Scratch).unwrap();
        let tool = ExecuteCommand { sandbox };
        let out = tool.run(&args("git", &["log", "--oneline"])).await;
        assert!(out.starts_with("Command succeeded.") && out.contains("first commit"), "{out}");
        let planted = base.join("main/.git/planted");
        let out = tool.run(&sh(&format!("echo x > {}", planted.display()))).await;
        assert!(out.contains("Read-only file system"), "{out}");
        assert!(!planted.exists());
    }
}
