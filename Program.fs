module Main

open System
open System.Buffers
open System.IO
open System.Net
open System.Net.Http
open System.Numerics
open System.Text
open System.Text.Json
open System.Text.Json.Serialization
open System.Threading
open System.Threading.Tasks
open Microsoft.FSharp.Control
open RewriteItIn.Concurrency

type ChatMessage = {
    role: string
    content: string
}

type ChatRequest = {
    model: string
    messages: ChatMessage list
    temperature: float
    max_tokens: int
    stream: bool
}

type CodeFile = {
    path: string
    content: string
}

type Options = {
    SourcePath: string
    SourceLang: string
    TargetPath: string
    TargetLang: string
    ServerUrl: string
    Model: string
    BatchSize: int
    MaxTokens: int
    Temperature: float
    FileSizeLimitBytes: int
    HttpRetries: int
    RewriteConcurrency: int
    TimeoutMinutes: int
    Principles: string option
    ApiKey: string option
}

let defaultOptions : Options = {
    TargetLang = ""
    SourceLang = ""
    TargetPath = ""
    SourcePath = ""
    ServerUrl = ""
    Model = ""
    MaxTokens = 8192
    Temperature = 1.3
    FileSizeLimitBytes = 120_000
    RewriteConcurrency = 3
    HttpRetries = 3
    TimeoutMinutes = 25
    Principles = None
    ApiKey = None
    BatchSize = 3
}

let isBinaryFile (filePath: string) =
    try
        use fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read)
        use memory = MemoryPool<byte>.Shared.Rent(1024)
        let span = memory.Memory.Span
        let bytesRead = fs.Read(span)
        
        if bytesRead = 0 then false
        else
            let isBinary = 
                // Check for text file markers first
                match span with
                | _ when bytesRead >= 3 && span[0] = 0xEFuy && span[1] = 0xBBuy && span[2] = 0xBFuy -> false // UTF-8 BOM
                | _ when bytesRead >= 2 ->
                    match span[0], span[1] with
                    | 0xFEuy, 0xFFuy | 0xFFuy, 0xFEuy -> false // UTF-16 BOMs
                    | _ ->
                        // Check for binary markers
                        let mutable i = 0
                        let mutable foundBinary = false
                        
                        while not foundBinary && i < bytesRead do
                            let c = span[i]
                            if c = 0uy then
                                foundBinary <- true
                            elif c > 0x7Fuy then
                                let leadingOnes = BitOperations.LeadingZeroCount(uint32 (~~~(int c)) <<< 24)
                                if leadingOnes < 2 || leadingOnes > 4 then
                                    foundBinary <- true
                                elif i + leadingOnes > bytesRead then
                                    foundBinary <- true
                                else
                                    let mutable j = 1
                                    while not foundBinary && j < leadingOnes do
                                        if span[i + j] &&& 0xC0uy <> 0x80uy then
                                            foundBinary <- true
                                        j <- j + 1
                                    i <- i + leadingOnes - 1
                            i <- i + 1
                        
                        foundBinary
                | _ -> false
            isBinary
    with
    | _ -> true
    
let readSourceFiles (options: Options) (allowedFormats: string Set option) =
    let ensureTrailingSlash (path: string) =
        if not (path.EndsWith(Path.DirectorySeparatorChar)) then path + string Path.DirectorySeparatorChar
        else path

    let sourceUri = 
        Path.GetFullPath(options.SourcePath)
        |> ensureTrailingSlash
        |> fun p -> Uri(p, UriKind.Absolute)

    let toUri (path: string) =
        Path.GetFullPath(path)
        |> fun p -> Uri(p, UriKind.Absolute)

    let getRelativePath (filePath: string) =
        let fileUri = toUri filePath
        if not (sourceUri.IsBaseOf(fileUri)) then
            failwithf $"Attempted to access file outside source directory: %s{filePath}"
        sourceUri.MakeRelativeUri(fileUri).ToString().Replace('/', Path.DirectorySeparatorChar)

    let hasAllowedExtension (filePath: string) (formats: string Set) =
        let fileExt = Path.GetExtension(filePath).ToLowerInvariant()
        formats |> Set.exists (fun ext -> 
            let normalizedExt = ext.ToLowerInvariant()
            fileExt = normalizedExt)

    let extensionAllowed (filePath: string) =
        match allowedFormats with
        | Some formats -> hasAllowedExtension filePath formats
        | None -> true

    let readFileContent filePath =
        { 
            path = if Directory.Exists options.SourcePath 
                   then getRelativePath filePath 
                   else Path.GetFileName(filePath)
            content = File.ReadAllText(filePath) 
        }

    let isNotTooBig file = FileInfo(file).Length < options.FileSizeLimitBytes
    
    let processDirectory () =
        try
            let allowed =
                Directory.GetFiles(sourceUri.LocalPath, "*", SearchOption.AllDirectories)
                |> Array.filter (isBinaryFile >> not)
                |> Array.filter extensionAllowed
            
            let (sizeOk, sizeNotOk) =
                allowed
                |> Array.partition isNotTooBig
            
            if not (Array.isEmpty sizeNotOk) then
                printfn $"Following files are too big to process according to the current file size limit (%d{options.FileSizeLimitBytes} bytes):"
                sizeNotOk
                |> Array.iter (fun file -> printfn $"- %s{file}")

            sizeOk
                |> Array.map readFileContent
                |> Array.toList
                |> List.randomShuffle
                
        with
        | :? UnauthorizedAccessException -> []
        | :? DirectoryNotFoundException -> []

    let processSingleFile () =
        let fileUri = toUri options.SourcePath
        if not (sourceUri.IsBaseOf(fileUri)) then
            []
        elif isBinaryFile options.SourcePath || not (extensionAllowed options.SourcePath) then 
            []
        else 
            [readFileContent options.SourcePath]

    if Directory.Exists sourceUri.LocalPath then
        processDirectory ()
    elif File.Exists sourceUri.LocalPath then
        processSingleFile ()
    else
        []

let createFormatQueryPayload (options: Options) =
    let systemMessage = {
        role = "system"
        content = "Keep the dot, strip whitespace. Do not add anything else, just formats. Response must be newline (\\n) separated list of formats. End with the newline.
        Sample:
.format1
.format2
.exe
.cs"
    }

    let userMessage = {
        role = "user"
        content = $"List all file extensions are used when writing apps using {options.SourceLang} OR {options.TargetLang}, ignore binary files. Also, these are the conversion principles as specified by the User, use them to figure out more formats: " + (options.Principles |> Option.defaultValue "None")
    }

    {
        model = options.Model
        messages = systemMessage :: [ userMessage ] 
        temperature = options.Temperature
        max_tokens = options.MaxTokens
        stream = false
    }

let createRewritePayload (options: Options) (files: CodeFile list) =
    let principlesPart = 
        options.Principles 
        |> Option.map (fun p -> $"Follow these mandatory principles: {p}.")
        |> Option.defaultValue ""
        
    let systemMessage = {
        role = "system"
        content =
            $"You have to rewrite code in {options.TargetLang} using the source in {options.SourceLang} as example." + principlesPart + "User input is JSON. Always respond with complete, unabridged answers. What you respond with is the final result. Do not simplify for brevity.
            Respond using the following format (zero or more files, feel free to change filenames, or create new files, or change them again if not finished before), provide no comments besides it:
````relative/path/to/write.cs
the source code to write to file write.cs
````
````relative/path/to/write2.cs
the source code to write to file write2.cs
````
"
    }
    
    let userContent = JsonSerializer.Serialize({| files = files |})

    let userMessage = {
        role = "user"
        content = userContent
    }

    {
        model = options.Model
        messages = systemMessage :: [ userMessage ]
        temperature = options.Temperature
        max_tokens = options.MaxTokens
        stream = false
    }
    
module HttpClientSingleton =    
    let private createClient() =
        let client = new HttpClient()
        client.Timeout <- Timeout.InfiniteTimeSpan
        client
        
    let Instance = lazy(createClient())

exception BatchTooBigException

let rec sendRequest (options: Options) (payload: ChatRequest) = async { 
    let url = $"{options.ServerUrl}/v1/chat/completions"
    
    let jsOptions = JsonSerializerOptions()
    jsOptions.DefaultIgnoreCondition <- JsonIgnoreCondition.WhenWritingNull
    
    let json = JsonSerializer.Serialize(payload, jsOptions)
    
    use content = new StringContent(json, Encoding.UTF8, "application/json")
    use request = new HttpRequestMessage(HttpMethod.Post, url)
    request.Content <- content
    
    match options.ApiKey with
    | Some key -> 
        request.Headers.Authorization <- 
            Headers.AuthenticationHeaderValue("Bearer", key)
    | None -> ()
    
    printfn $"Sending request to: {url}"

    try
        let client = HttpClientSingleton.Instance.Value
        use cts = new CancellationTokenSource(TimeSpan.FromMinutes(options.TimeoutMinutes))
        let! response = client.SendAsync(request, cts.Token) |> Async.AwaitTask
        
        let! responseContent = response.Content.ReadAsStringAsync() |> Async.AwaitTask
        
        if response.StatusCode = HttpStatusCode.BadRequest then
            raise BatchTooBigException
        
        if not response.IsSuccessStatusCode then
            raise (HttpRequestException($"Request failed with status code {response.StatusCode}: {responseContent}"))

        return responseContent
    with
    | :? BatchTooBigException as ex ->
        return! raise ex
    | :? TaskCanceledException when options.HttpRetries > 0 ->
        let newRetryAttempts = options.HttpRetries - 1
        printfn $"Retrying request to: {url}, {newRetryAttempts} attempts left"
        return! sendRequest { options with HttpRetries = newRetryAttempts } payload
    | ex ->
        return! raise ex  // Re-throw any other unexpected exceptions
}
    
let getMessage (rawResponse: string) =
    try
        use jsonDoc = JsonDocument.Parse(rawResponse)
        let root = jsonDoc.RootElement
        
        match root.TryGetProperty("choices") with
        | (true, choices) when choices.ValueKind = JsonValueKind.Array && choices.GetArrayLength() > 0 ->
            let firstChoice = choices[0]
            
            match firstChoice.TryGetProperty("message") with
            | (true, message) ->
                match message.TryGetProperty("content"), message.TryGetProperty("role") with
                | (true, content), (true, role) ->
                    Some { content = content.GetString(); role = role.GetString() } 
                | _ -> 
                    printfn "Invalid response: missing content or role"
                    None
            | _ -> 
                printfn "Invalid response: missing message"
                None
        | _ -> 
            printfn "Invalid response: missing choices or choices is empty"
            None
    with ex ->
        printfn $"Error parsing response: {ex.Message}"
        None
        
let parseSetResponse (message: ChatMessage option) =
    message   
    |> Option.map (fun message ->
        message.content.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
        |> Array.map (fun s -> s.Trim())
        |> Array.filter (System.String.IsNullOrWhiteSpace >> not)
        |> Set.ofArray
    )
    
let parseRewriteResponse (message: ChatMessage option) =
    message
    |> Option.bind (fun message ->
        let lines = message.content.Split('\n')
        let mutable currentPath = None
        let mutable currentContent = []
        let mutable files = []
        for line in lines do
            match currentPath with
            | None ->
                let trimmedLine = line.TrimStart()
                if trimmedLine.StartsWith("````") then
                    let pathPart = trimmedLine.[4..].Trim()
                    currentPath <- Some(pathPart)
                    currentContent <- []
            | Some(path) ->
                let trimmedLine = line.Trim()
                if trimmedLine.StartsWith("````") then
                    let content = currentContent |> List.rev |> String.concat Environment.NewLine
                    files <- { path = path; content = content } :: files
                    currentPath <- None
                    currentContent <- []
                else
                    currentContent <- line :: currentContent
        Some (List.rev files)
    )


let parseCommandLine (args: string[]) =
    let rec parseInternal (args: string list) (options: Options) =
        match args with
        | [] -> options
        | "--source" :: src :: rest -> parseInternal rest { options with SourcePath = src }
        | "--source-lang" :: lang :: rest -> parseInternal rest { options with SourceLang = lang }
        | "--target" :: tgt :: rest -> parseInternal rest { options with TargetPath = tgt }
        | "--target-lang" :: lang :: rest -> parseInternal rest { options with TargetLang = lang }
        | "--server" :: url :: rest -> parseInternal rest { options with ServerUrl = url }
        | "--model" :: model :: rest -> parseInternal rest { options with Model = model }
        | "--batch-size" :: size :: rest -> parseInternal rest { options with BatchSize = int size }
        | "--rewrite-concurrency" :: concurrency :: rest -> parseInternal rest { options with RewriteConcurrency = int concurrency }
        | "--max-tokens" :: tokens :: rest -> parseInternal rest { options with MaxTokens = int tokens }
        | "--temperature" :: temp :: rest -> parseInternal rest { options with Temperature = float temp }
        | "--file-size-limit-bytes" :: limit :: rest -> parseInternal rest { options with FileSizeLimitBytes = int limit }
        | "--http-retries" :: retries :: rest -> parseInternal rest { options with HttpRetries = int retries }
        | "--timeout-minutes" :: timeoutMinutes :: rest -> parseInternal rest { options with TimeoutMinutes = int timeoutMinutes }
        | "--principles" :: principles :: rest -> parseInternal rest { options with Principles = Some principles }
        | "--api-key" :: key :: rest -> parseInternal rest { options with ApiKey = Some key }
        | x :: _ -> failwith $"Unknown option: {x}"
    
    parseInternal (args |> Array.toList) defaultOptions

let queryRelevantFormatsAsync (options: Options) = async {
    printfn "\n=== Querying relevant file formats ==="
    let payload = createFormatQueryPayload options
    let! response = sendRequest options payload
    let message = getMessage response
    return parseSetResponse message
}

let printBatch list =
    printfn $"=== Batch ==="
    list 
    |> List.iter (fun file -> printfn $"  - {file.path}")

exception NoValidConversionException
let rewriteBatchAsync (options: Options) (files: CodeFile list)  = async {
    printfn $"\nProcessing {files.Length} files..."

    let payload = createRewritePayload options files
    let! response = sendRequest options payload
    let message = getMessage response
    match parseRewriteResponse message with
    | Some results ->
        printfn $"Successfully converted {results.Length} files"
        return results
    | None ->
        printfn "No valid conversion returned for these files"
        return! raise NoValidConversionException      
}

let private fileWriteLock = obj()

let writeOutputFile (targetPath: string) (file: CodeFile) =
    lock fileWriteLock (fun () ->
        let targetDir = Path.GetFullPath(targetPath)
        let targetDirNormalized =
            if not (targetDir.EndsWith(Path.DirectorySeparatorChar.ToString())) then
                targetDir + string Path.DirectorySeparatorChar
            else
                targetDir
        let combinedPath = Path.Combine(targetDir, file.path)
        let fullFilePath = Path.GetFullPath(combinedPath)
        let targetUri = Uri(targetDirNormalized)
        let fileUri = Uri(fullFilePath)
        if not (targetUri.IsBaseOf(fileUri)) then
            raise (ArgumentException $"The file path '%s{fullFilePath}' is outside the target directory '%s{targetDir}'.")
        else
            Directory.CreateDirectory(Path.GetDirectoryName(fullFilePath)) |> ignore
            File.WriteAllText(fullFilePath, file.content)
    )

let rec rewriteFilesInBatchesAsync (options: Options) (files: CodeFile list) = async {
    if List.isEmpty files then
        return 0
    else
        let batches = files |> List.chunkBySize options.BatchSize

        let tasks = 
            batches
            |> List.map (fun batch ->
                async {
                    printBatch batch
                    let rec processBatch (currentOptions: Options) (currentBatch: CodeFile list) = async {
                        try 
                            let! results = rewriteBatchAsync currentOptions currentBatch
                            results |> List.iter (writeOutputFile options.TargetPath)
                            return results.Length
                        with
                        | :? BatchTooBigException | NoValidConversionException as ex when currentOptions.BatchSize > 1 ->
                            printfn $"Error processing batch: {ex}. Retrying with smaller batch size."
                            let newBatchSize = currentOptions.BatchSize / 2
                            let smallerBatches = currentBatch |> List.chunkBySize newBatchSize
                            let! counts = 
                                smallerBatches
                                |> List.map (processBatch { currentOptions with BatchSize = newBatchSize })
                                |> Async.Parallel
                            return counts |> Array.sum
                        | ex ->
                            printfn $"Error processing batch: {ex.Message}. Skipping."
                            return 0
                    }
                    return! processBatch options batch
                }
            )

        let! results = 
            tasks
            |> runThrottled options.RewriteConcurrency

        return results |> Seq.sum
}

let runConversion (options: Options) (history: ChatMessage List) = async {
    try
        printfn "\n=== Starting file formats phase ==="

        printfn $"Determining relevant file formats for {options.SourceLang} -> {options.TargetLang} conversion"
        let! relevantFormatsList = queryRelevantFormatsAsync options
        let relevantFormats = match relevantFormatsList with Some f -> String.Join(", ", f) | None -> "All files"
        printfn $"Relevant formats: {relevantFormats}"

        printfn $"Reading source files from %s{options.SourcePath}"
        let allSourceFiles = readSourceFiles options relevantFormatsList
        printfn $"Found {allSourceFiles.Length} relevant files to process"

        Directory.CreateDirectory(options.TargetPath) |> ignore
        
        printfn "\n=== Starting rewrite phase ==="
        let! successCount = rewriteFilesInBatchesAsync options allSourceFiles
        
        printfn $"\nConversion complete. Successfully converted {successCount} of {allSourceFiles.Length} source files"
        return history
    with
    | :? OperationCanceledException as ex ->
        Console.ForegroundColor <- ConsoleColor.Red
        printfn $"\nTask cancelled: {ex.Message}"
        Console.ResetColor()
        return history
    | ex ->
        printfn $"\nError: %s{ex.Message}"
        return history
}

[<EntryPoint>]
let main argv =
    try
        if argv.Length = 0 then
            printfn "Usage: dotnet run [options]"
            printfn "Options:"
            printfn $"  --source <path>                Source file/directory"
            printfn $"  --source-lang <lang>           Source language"
            printfn $"  --target <path>                Target directory"
            printfn $"  --target-lang <lang>           Target language"
            printfn $"  --server <url>                 API server URL"
            printfn $"  --model <model>                LLM Model name"
            printfn $"  --batch-size <n>               (Optional) Files per batch (default: {defaultOptions.BatchSize})"
            printfn $"  --rewrite-concurrency <n>      (Optional) Rewrite phase concurrency (default: {defaultOptions.RewriteConcurrency})"
            printfn $"  --max-tokens <n>               (Optional) Max response tokens (default: {defaultOptions.MaxTokens})"
            printfn $"  --temperature <f>              (Optional) LLM temperature (default: {defaultOptions.Temperature})"
            printfn $"  --file-size-limit-bytes <n>    (Optional) Max file size in bytes (default: {defaultOptions.FileSizeLimitBytes})"
            printfn $"  --http-retries <n>             (Optional) Conversion retries (default: {defaultOptions.HttpRetries})"
            printfn $"  --timeout-minutes <n>          (Optional) LLM API Timeout in minutes (default: {defaultOptions.TimeoutMinutes})"
            printfn $"  --principles <text>            (Optional) Conversion principles"
            printfn $"  --api-key <key>                (Optional) API key"
            1
        else
            let options = parseCommandLine argv
            
            if String.IsNullOrEmpty(options.SourcePath) then failwith "Source path is required"
            if String.IsNullOrEmpty(options.SourceLang) then failwith "Source language is required"
            if String.IsNullOrEmpty(options.TargetPath) then failwith "Target path is required"
            if String.IsNullOrEmpty(options.TargetLang) then failwith "Target language is required"
            if String.IsNullOrEmpty(options.ServerUrl) then failwith "Server URL is required"
            if String.IsNullOrEmpty(options.Model) then failwith "LLM Model is required"

            runConversion options [] |> Async.RunSynchronously |> ignore
            0
    with
    | ex ->
        printfn $"Error: {ex.Message}"
        2