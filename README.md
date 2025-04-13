# Rewrite It In

A code conversion tool that uses the DeepSeek v3 LLM to rewrite code from one programming/markup language to another while maintaining structure and functionality and following your principles.

It also supports local LM Studio Server API with a bit of modification.

It sends your data to the server specified.

There is no guarantee.

## Features

- Converts code between programming/markup languages using LLM API
- Supports batch processing of multiple files
- Maintains directory structure in output
- Optional file format filtering
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
  --principles <text>    (Optional) Conversion principles
  --api-key <key>        (Optional) API key for authentication
  --use-json-schema      (Optional) Use Json Schema

### Example
```bash
dotnet run --source  ../nettsundere.github.io --source-lang HTML --target ../test/w2 --target-lang HTML --server https://api.deepseek.com --principles "Fix the paths, make it look modern, add missing things" --api-key SAMPLE 
```

## License
MIT, see [LICENSE](https://github.com/nettsundere/RewriteItIn/blob/master/LICENSE) 
