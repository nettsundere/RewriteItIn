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
open Microsoft.FSharp.Control

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
    HistoryLimit: int
    FileSizeLimitBytes: int
    ConversionRetries: int
    TimeoutMinutes: int
    Principles: string option
    ApiKey: string option
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

let shuffle (rng: Random) (list: 'a list) =
    let arr = List.toArray list
    for i in [ 0 .. arr.Length - 1 ] do
        let j = rng.Next(i, arr.Length)
        let tmp = arr.[i]
        arr.[i] <- arr.[j]
        arr.[j] <- tmp
    Array.toList arr
    
let readSourceFiles (options: Options) (allowedFormats: string list option) =
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

    let hasAllowedExtension (filePath: string) (formats: string list) =
        let fileExt = Path.GetExtension(filePath).ToLowerInvariant()
        formats |> List.exists (fun ext -> 
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
            Directory.GetFiles(sourceUri.LocalPath, "*", SearchOption.AllDirectories)
            |> Array.filter (isBinaryFile >> not)
            |> Array.filter extensionAllowed
            |> Array.filter isNotTooBig
            |> Array.map readFileContent
            |> Array.toList
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

let combineMessagesForHistory (systemMessage: ChatMessage) (userMessage: ChatMessage) (history: ChatMessage list) =
    userMessage :: ( history |> List.filter (fun message -> message.role <> "system") ) @ [ systemMessage ]
    
let captureAssistantReplyInHistory (maybeNewMessage: ChatMessage option) (payloadMessages: ChatMessage list) =
    match maybeNewMessage with
    | Some value -> value :: payloadMessages
    | None -> payloadMessages

let createFormatQueryPayload (options: Options) (history: ChatMessage list) =
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
        content = $"List all file extensions are used when writing apps using {options.SourceLang} OR {options.TargetLang}, ignore binary files. Also, these are the conversion principles as specified by the User: " + (options.Principles |> Option.defaultValue "None")
    }

    {
        model = options.Model
        messages = combineMessagesForHistory systemMessage userMessage history 
        temperature = options.Temperature
        max_tokens = options.MaxTokens
        stream = false
    }

let createAnalysisPayload (options: Options) (files: CodeFile list) (history: ChatMessage list) =
    let systemMessage = {
        role = "system"
        content = "Analyze these code files and identify which ones are important when rewriting it to another language.
        You have to Respond with a newline (\\n) separated list of such filenames. Provide no other data, just filenames.
        Add a newline to the end of a list. Input is JSON, { files = [ { filename, source }, ... ] }"
    }

    let userContent = 
        JsonSerializer.Serialize({| files = files |}, JsonSerializerOptions(WriteIndented = true))

    let userMessage = {
        role = "user"
        content = userContent
    }

    {
        model = options.Model
        messages = combineMessagesForHistory systemMessage userMessage history 
        temperature = options.Temperature
        max_tokens = options.MaxTokens
        stream = false
    }

let createRewritePayload (options: Options) (files: CodeFile list) (history: ChatMessage list)=
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
        messages = combineMessagesForHistory systemMessage userMessage history 
        temperature = options.Temperature
        max_tokens = options.MaxTokens
        stream = false
    }
    
let truncateHistoryShuffle (messages: ChatMessage list) (historyLimit: int) =
    match messages with
    | [] -> []
    | _ ->
        if historyLimit <= 0 then
            []
        else
            let userMessagesFun = (fun m -> m.role = "user")
            let systemMessage = List.last messages
            let conversations = List.take (messages.Length - 1) messages
            let fromFirstAssistant = List.skipWhile userMessagesFun conversations
            let pairs = fromFirstAssistant |> List.chunkBySize 2
            let shuffledPairs = shuffle (Random()) pairs 
            let maxPairs = (historyLimit - 1) / 2
            let keptPairs = shuffledPairs |> List.truncate maxPairs
            let unansweredMessages = List.takeWhile userMessagesFun conversations
            unansweredMessages @ (keptPairs |> List.concat) @ [ systemMessage ]

module HttpClientSingleton =    
    let private createClient() =
        let client = new HttpClient()
        client.Timeout <- Timeout.InfiniteTimeSpan
        client
        
    let Instance = lazy(createClient())

let sendRequest (options: Options) (payload: ChatRequest) =
    let rec retryWithReducedHistory (currentLimit: int) =
        async { 
            let url = $"{options.ServerUrl}/v1/chat/completions"
            
            let jsOptions = JsonSerializerOptions()
            jsOptions.DefaultIgnoreCondition <- JsonIgnoreCondition.WhenWritingNull
            
            let orderedPayload = { payload with messages = truncateHistoryShuffle payload.messages currentLimit |> List.rev }
            let json = JsonSerializer.Serialize(orderedPayload, jsOptions)
            
            use content = new StringContent(json, Encoding.UTF8, "application/json")
            use request = new HttpRequestMessage(HttpMethod.Post, url)
            request.Content <- content
            
            match options.ApiKey with
            | Some key -> 
                request.Headers.Authorization <- 
                    Headers.AuthenticationHeaderValue("Bearer", key)
            | None -> ()
            
            printfn $"Sending request to: {url} with history limit: {currentLimit}"

            try
                let client = HttpClientSingleton.Instance.Value
                use cts = new CancellationTokenSource(TimeSpan.FromMinutes(options.TimeoutMinutes))
                let! response = client.SendAsync(request, cts.Token) |> Async.AwaitTask
                
                if response.StatusCode = HttpStatusCode.BadRequest && currentLimit >= 2 then
                    let newLimit = currentLimit / 2
                    let! badRequestMessage = response.Content.ReadAsStringAsync() |> Async.AwaitTask
                    printfn $"Bad request received {badRequestMessage}, reducing history limit from {currentLimit} to {newLimit}"
                    return! retryWithReducedHistory newLimit
                elif not response.IsSuccessStatusCode then
                    let! errorContent = response.Content.ReadAsStringAsync() |> Async.AwaitTask
                    return failwith $"Request failed with status {response.StatusCode}: {errorContent}"
                else
                    let! responseContent = response.Content.ReadAsStringAsync() |> Async.AwaitTask
                    printfn $"Received response: {responseContent}"
                    return responseContent
            with
            | ex when currentLimit >= 2 ->
                let newLimit = currentLimit / 2
                printfn $"Request failed {ex.Message}, reducing history limit from {currentLimit} to {newLimit}"
                return! retryWithReducedHistory newLimit
            | ex ->
                return raise ex
        }
    
    retryWithReducedHistory options.HistoryLimit

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
        
let parseListResponse (message: ChatMessage option) =
    message   
    |> Option.map (fun message ->
        message.content.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
        |> Array.map (fun s -> s.Trim())
        |> Array.filter (System.String.IsNullOrWhiteSpace >> not)
        |> Array.toList
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

let writeOutputFile (targetPath: string) (file: CodeFile) =
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
        | "--max-tokens" :: tokens :: rest -> parseInternal rest { options with MaxTokens = int tokens }
        | "--temperature" :: temp :: rest -> parseInternal rest { options with Temperature = float temp }
        | "--history-limit" :: limit :: rest -> parseInternal rest { options with HistoryLimit = int limit }
        | "--file-size-limit-bytes" :: limit :: rest -> parseInternal rest { options with FileSizeLimitBytes = int limit }
        | "--conversion-retries" :: retries :: rest -> parseInternal rest { options with ConversionRetries = int retries }
        | "--timeout-minutes" :: timeoutMinutes :: rest -> parseInternal rest { options with TimeoutMinutes = int timeoutMinutes }
        | "--principles" :: principles :: rest -> parseInternal rest { options with Principles = Some principles }
        | "--api-key" :: key :: rest -> parseInternal rest { options with ApiKey = Some key }
        | x :: _ -> failwith $"Unknown option: {x}"
    
    let defaultOptions = {
        SourcePath = ""
        SourceLang = ""
        TargetPath = ""
        TargetLang = ""
        ServerUrl = ""
        Model = ""
        BatchSize = 30
        MaxTokens = 8192
        Temperature = 1.3
        HistoryLimit = 25
        FileSizeLimitBytes = 1024 * 50_000
        ConversionRetries = 3
        TimeoutMinutes = 25
        Principles = None
        ApiKey = None
    }
    
    parseInternal (args |> Array.toList) defaultOptions

let queryRelevantFormatsWithHistoryAsync (options: Options) (history: ChatMessage list) = async {
    printfn "\n=== Querying relevant file formats ==="
    let payload = createFormatQueryPayload options history
    let! response = sendRequest options payload
    let message = getMessage response
    return (parseListResponse message, captureAssistantReplyInHistory message payload.messages)
}

let printBatch list =
    printfn $"=== Batch ==="
    list 
    |> List.iter (fun file -> printfn $"  - {file.path}")

let filterImportantFilesWithHistoryAsync (options: Options) (allFiles: CodeFile list) (history: ChatMessage List) = async {
    printfn "\n=== Filtering important files ==="
    let batchSize = options.BatchSize
    let batches = 
        allFiles 
        |> List.chunkBySize batchSize

    let mutable importantFiles = Set.empty
    let mutable filteringHistory = history
    
    for batch in batches do
        printfn $"Analyzing batch of {batch.Length} files..."
        printBatch batch
        let payload = createAnalysisPayload options batch filteringHistory
        let! response = sendRequest options payload
        let message = getMessage response
        filteringHistory <- captureAssistantReplyInHistory message payload.messages

        match parseListResponse message with
        | Some results ->
            importantFiles <- Set.union importantFiles (results |> Set.ofList)
            printfn $"Found {results.Length} important files in this batch (Total: {importantFiles.Count})"
        | None -> 
            printfn "No important files identified in this batch"
        
    let files =
        allFiles
        |> List.filter (fun file -> Set.contains file.path importantFiles)
        
    return (files, filteringHistory)
}

let rewriteFilesWithHistoryAsync (options: Options) (files: CodeFile list) (history: ChatMessage list)  = async {
    printfn $"\nProcessing {files.Length} files..."
    try
        let payload = createRewritePayload options files history
        let! response = sendRequest options payload
        let message = getMessage response
        match parseRewriteResponse message with
        | Some results ->
            printfn $"Successfully converted {results.Length} files"
            return (Some results, captureAssistantReplyInHistory message payload.messages)
        | None ->
            printfn "No valid conversion returned for these files"
            return (None, history)
    with ex ->
        printfn $"Error processing files: {ex.Message}, {ex.StackTrace}"
        return (None, history)
}


let rewriteBatchWithRetryAsync (options: Options) (batch: CodeFile list) (history: ChatMessage list) (maxRetries: int) = async {
    let rec retry attemptsRemaining = async {
        try
            let! (results, newHistory) = rewriteFilesWithHistoryAsync options batch history
            return (results, newHistory)
        with ex ->
            if attemptsRemaining > 0 then
                printfn $"Retrying batch ({maxRetries - attemptsRemaining + 1}/{maxRetries})..."
                return! retry (attemptsRemaining - 1)
            else
                printfn $"Failed to rewrite batch after {maxRetries} attempts: {ex.Message}"
                return (None, history)
    }
    
    return! retry maxRetries
}

let rewriteFilesInBatchesWithRetryAsync (options: Options) (files: CodeFile list) (initialHistory: ChatMessage list) (maxRetries: int) = async {
    let batchSize = options.BatchSize
    let batches = files |> List.chunkBySize batchSize
    let mutable successCount = 0
    let mutable currentHistory = initialHistory
    
    for batch in batches do
        printBatch batch
        let! (results, newHistory) = rewriteBatchWithRetryAsync options batch currentHistory maxRetries
        currentHistory <- newHistory
        match results with
        | Some rewrittenFiles ->
            rewrittenFiles |> List.iter (fun result ->
                writeOutputFile options.TargetPath result
                successCount <- successCount + 1
            )
        | None -> ()
    
    return (successCount, currentHistory)
}

let runConversion (options: Options) (history: ChatMessage List) = async {
    try
        printfn $"Determining relevant file formats for {options.SourceLang} -> {options.TargetLang} conversion"
        let! (relevantFormatsList, formatsHistory) = queryRelevantFormatsWithHistoryAsync options history
        let relevantFormats = match relevantFormatsList with Some f -> String.Join(", ", f) | None -> "All files"
        printfn $"Relevant formats: {relevantFormats}"

        printfn $"Reading source files from %s{options.SourcePath}"
        let allSourceFiles = readSourceFiles options relevantFormatsList
        printfn $"Found {allSourceFiles.Length} relevant files to process"

        let! (importantFiles, importantFilesHistory) = filterImportantFilesWithHistoryAsync options allSourceFiles formatsHistory
        printfn $"Identified {importantFiles.Length} important files to convert"

        Directory.CreateDirectory(options.TargetPath) |> ignore
        
        printfn "\n=== Starting rewrite phase ==="
        let! successCount, _ = rewriteFilesInBatchesWithRetryAsync options importantFiles importantFilesHistory options.ConversionRetries
        
        printfn $"\nConversion complete. Successfully converted {successCount} of {importantFiles.Length} important files"
        return 0
    with
    | :? OperationCanceledException as ex ->
        Console.ForegroundColor <- ConsoleColor.Red
        printfn $"\nTask cancelled: {ex.Message}"
        Console.ResetColor()
        return 1
    | ex ->
        printfn $"\nError: %s{ex.Message}"
        return 1
}

[<EntryPoint>]
let main argv =
    try
        if argv.Length = 0 then
            printfn "Usage: dotnet run [options]"
            printfn "Options:"
            printfn "  --source <path>          Source file/directory"
            printfn "  --source-lang <lang>     Source language"
            printfn "  --target <path>          Target directory"
            printfn "  --target-lang <lang>     Target language"
            printfn "  --server <url>           API server URL"
            printfn "  --model <model>          LLM Model name"
            printfn "  --batch-size <n>         (Optional) Files per batch (default: 30)"
            printfn "  --max-tokens <n>         (Optional) Max response tokens (default: 8192)"
            printfn "  --temperature <f>        (Optional) AI temperature (default: 1.3)"
            printfn "  --history-limit <n>      (Optional) Conversation history limit (default: 25)"
            printfn "  --file-size-limit <n>    (Optional) Max file size in bytes (default: 51200000)"
            printfn "  --conversion-retries <n> (Optional) Conversion retries (default: 3)"
            printfn "  --timeout-minutes <n>    (Optional) LLM API Timeout in minutes (default: 25)"
            printfn "  --principles <text>      (Optional) Conversion principles"
            printfn "  --api-key <key>          (Optional) API key"
            1
        else
            let options = parseCommandLine argv
            
            if String.IsNullOrEmpty(options.SourcePath) then failwith "Source path is required"
            if String.IsNullOrEmpty(options.SourceLang) then failwith "Source language is required"
            if String.IsNullOrEmpty(options.TargetPath) then failwith "Target path is required"
            if String.IsNullOrEmpty(options.TargetLang) then failwith "Target language is required"
            if String.IsNullOrEmpty(options.ServerUrl) then failwith "Server URL is required"
            if String.IsNullOrEmpty(options.Model) then failwith "LLM Model is required"

            runConversion options [] |> Async.RunSynchronously
    with
    | ex ->
        printfn $"Error: {ex.Message}"
        2