use rig::providers::openai;
use rig::completion::Prompt;
use std::env;

#[tokio::main]
async fn main() {
    let key = env::var("OPENAI_API_KEY").unwrap_or_else(|_| "stub".to_string());
    let client = openai::Client::new(&key);
    let agent = client.agent("gpt-5").build();
    let res = agent.prompt("hello").await;
    println!("{:?}", res);
}
