# Rewrite It In

A code conversion tool that uses the DeepSeek v3 LLM or LM Studio API to rewrite code from one programming/markup language to another while maintaining structure and functionality and following your principles.

## Features

- Converts code between programming/markup languages using LLM API
- Supports batch processing of multiple files
- Maintains directory structure in output
- Customizable conversion principles
- Concurrency by default
- API key authentication support
- Progress tracking and error reporting

## How to run
```bash
git clone https://github.com/nettsundere/rewrite-it-in.git
cd rewrite-it-in
```

```bash
Options:
  --source <path>                Source file/directory
  --source-lang <lang>           Source language
  --target <path>                Target directory
  --target-lang <lang>           Target language
  --server <url>                 API server URL
  --model <model>                LLM Model name
  --batch-size <n>               (Optional) Files per batch (default: 3)
  --rewrite-concurrency <n>      (Optional) Rewrite phase concurrency (default: 3)
  --max-tokens <n>               (Optional) Max response tokens (default: 8192)
  --temperature <f>              (Optional) LLM temperature (default: 1,3)
  --file-size-limit-bytes <n>    (Optional) Max file size in bytes (default: 120000)
  --http-retries <n>             (Optional) Conversion retries (default: 3)
  --timeout-minutes <n>          (Optional) LLM API Timeout in minutes (default: 25)
  --principles <text>            (Optional) Conversion principles
  --api-key <key>                (Optional) API key

```

### Example (rewrite this tool itself to JAVA)
```bash
dotnet run --source . --source-lang F# --target ../test/w12 --target-lang JAVA --server https://api.deepseek.com --principles "Create standard JAVA project structure" --model "deepseek-reasoner"  --api-key REDACTED
```

## License
MIT, see [LICENSE](https://github.com/nettsundere/RewriteItIn/blob/master/LICENSE) 
