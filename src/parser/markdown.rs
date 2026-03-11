use pulldown_cmark::{Event, Parser, Tag, TagEnd};
use std::collections::HashMap;

pub struct MarkdownSection {
    pub title: String,
    pub content: String,
}

pub struct MarkdownParser;

impl MarkdownParser {
    pub fn parse_sections(markdown: &str) -> Vec<MarkdownSection> {
        let mut sections = Vec::new();
        let mut current_title = String::new();
        let mut current_content = String::new();
        let mut in_heading = false;
        
        let parser = Parser::new(markdown);

        for event in parser {
            match event {
                Event::Start(Tag::Heading { .. }) => {
                    if !current_title.is_empty() || !current_content.is_empty() {
                        sections.push(MarkdownSection {
                            title: current_title.trim().to_string(),
                            content: current_content.trim().to_string(),
                        });
                    }
                    current_title.clear();
                    current_content.clear();
                    in_heading = true;
                }
                Event::End(TagEnd::Heading(_)) => {
                    in_heading = false;
                }
                Event::Text(text) => {
                    if in_heading {
                        current_title.push_str(&text);
                    } else {
                        current_content.push_str(&text);
                    }
                }
                Event::Code(code) => {
                    if in_heading {
                        current_title.push_str(&code);
                    } else {
                        current_content.push_str(&format!("`{}`", code));
                    }
                }
                Event::SoftBreak | Event::HardBreak => {
                    current_content.push('\n');
                }
                Event::Start(Tag::Link { .. }) | Event::End(TagEnd::Link) => {
                    // Links are just text inside
                }
                Event::Start(Tag::Emphasis) | Event::End(TagEnd::Emphasis) => {
                    current_content.push('*');
                }
                Event::Start(Tag::Strong) | Event::End(TagEnd::Strong) => {
                    current_content.push_str("**");
                }
                Event::Start(Tag::CodeBlock(_)) => {
                    current_content.push_str("\n```\n");
                }
                Event::End(TagEnd::CodeBlock) => {
                    current_content.push_str("\n```\n");
                }
                Event::Start(Tag::List(_)) | Event::End(TagEnd::List(_)) => {
                    current_content.push('\n');
                }
                Event::Start(Tag::Item) => {
                    current_content.push_str("- ");
                }
                Event::End(TagEnd::Item) => {
                    current_content.push('\n');
                }
                _ => {}
            }
        }

        if !current_title.is_empty() || !current_content.is_empty() {
            sections.push(MarkdownSection {
                title: current_title,
                content: current_content.trim().to_string(),
            });
        }

        sections
    }

    pub fn to_map(sections: Vec<MarkdownSection>) -> HashMap<String, String> {
        sections
            .into_iter()
            .map(|s| (s.title.to_lowercase(), s.content))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sections() {
        let markdown = r#"# Vision
This is the vision.

## Features
- Feature 1
- Feature 2

### Infrastructure
Some infrastructure details.
"#;
        let sections = MarkdownParser::parse_sections(markdown);
        assert_eq!(sections.len(), 3);
        assert_eq!(sections[0].title, "Vision");
        assert_eq!(sections[0].content, "This is the vision.");
        assert_eq!(sections[1].title, "Features");
        assert!(sections[1].content.contains("- Feature 1"));
        assert!(sections[1].content.contains("- Feature 2"));
        assert_eq!(sections[2].title, "Infrastructure");
        assert_eq!(sections[2].content, "Some infrastructure details.");
    }
}
