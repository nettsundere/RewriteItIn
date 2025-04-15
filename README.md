# Rewrite It In

A code conversion tool that uses the DeepSeek v3 LLM or LM Studio API to rewrite code from one programming/markup language to another while maintaining structure and functionality and following your principles.

## Features

- Converts code between programming/markup languages using LLM API
- Supports batch processing of multiple files
- Maintains directory structure in output
- Smart file format filtering
- Customizable conversion principles
- API key authentication support
- Progress tracking and error reporting

## How to run
```bash
git clone https://github.com/nettsundere/rewrite-it-in.git
cd rewrite-it-in
...

```bash
Usage: dotnet run [options]
Options:
  --source <path>        Source file or directory
  --source-lang <lang>   Source language
  --target <path>        Target directory
  --target-lang <lang>   Target language
  --server <url>         API server URL
  --model <model-name>   LLM Model (like deepseek-chat)
  --principles <text>    (Optional) Conversion principles
  --api-key <key>        (Optional) API key for authentication
```

### Example (rewrite this tool itself to JAVA)
```bash
dotnet run --source . --source-lang F# --target ../test/w12 --target-lang JAVA --server https://api.deepseek.com --principles "Create standard JAVA project structure" --model "deepseek-reasoner"  --api-key REDACTED
```

## License
MIT, see [LICENSE](https://github.com/nettsundere/RewriteItIn/blob/master/LICENSE) 
