open System
open System.Buffers
open System.Buffers.Binary
open System.IO
open System.Net.Http
open System.Numerics
open System.Text
open System.Text.Json
open System.Text.Json.Serialization

let batchSizeGlobal = 30
let maxTokensGlobal = 8_192
let globalTimeoutMinutes = 15;
let globalTemperature = 1.3

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

            isBinary || 
            // Check for binary file signatures if no markers found
            (bytesRead >= 4 && 
                let signature = BinaryPrimitives.ReadUInt32LittleEndian(span)
                match signature with
                | 0x46445025u  // %PDF
                | 0x04034B50u   // ZIP
                | 0x464C457Fu   // ELF
                | 0x4D5A9000u   // MZ (PE)
                | 0x474E5089u   // PNG
                | 0x38464947u   // GIF
                | 0x46464952u   // RIFF
                | 0xE011CFD0u -> true  // MS Office
                | _ -> false)
    with
    | _ -> true

let readSourceFiles (sourcePath: string) (allowedFormats: string list option) =
    let getRelativePath (filePath: string) =
        if Directory.Exists sourcePath then
            Path.GetRelativePath(sourcePath, filePath)
        else
            Path.GetFileName(filePath)

    let filterByFormat (files: string list) =
        match allowedFormats with
        | Some formats ->
            files |> List.filter (fun f ->
                formats |> List.exists (fun ext ->
                    f.EndsWith(ext, StringComparison.OrdinalIgnoreCase)
                ))
        | _ -> files

    if Directory.Exists sourcePath then
        Directory.GetFiles(sourcePath, "*", SearchOption.AllDirectories)
        |> Array.toList  // Convert array to list
        |> List.filter (isBinaryFile >> not)
        |> filterByFormat
        |> List.map (fun filePath -> 
            let relativePath = getRelativePath filePath
            let content = File.ReadAllText(filePath)
            { path = relativePath; content = content })
    elif File.Exists sourcePath then
        if isBinaryFile sourcePath then
            []
        else
            let shouldInclude = 
                match allowedFormats with
                | Some formats -> formats |> List.exists (fun ext -> sourcePath.EndsWith(ext, StringComparison.OrdinalIgnoreCase))
                | None -> true
            if shouldInclude then
                [ { path = getRelativePath sourcePath; content = File.ReadAllText(sourcePath) } ]
            else
                []
    else
        failwith "Source path does not exist"

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
        content = $"List all file extensions are used when writing apps using {options.SourceLang} OR {options.TargetLang}, ignore binary files. Also, these are the conversion principles as specified by the User: " + (options.Principles |> Option.defaultValue "None")

    }

    {
        model = options.Model
        messages = [systemMessage; userMessage]
        temperature = globalTemperature
        max_tokens = maxTokensGlobal
        stream = false
    }

let createAnalysisPayload (options: Options) (files: CodeFile list) =
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
        messages = [systemMessage; userMessage]
        temperature = globalTemperature
        max_tokens = maxTokensGlobal
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
            Respond using the following format (zero or more files, feel free to change filenames, or create new files), provide no comments besides it:
```relative/path/to/write.cs
the source code to write to file write.cs
```
```relative/path/to/write2.cs
the source code to write to file write2.cs
```
"
    }
    
    let userContent = JsonSerializer.Serialize({| files = files |})

    let userMessage = {
        role = "user"
        content = userContent
    }

    {
        model = options.Model
        messages = [systemMessage; userMessage]
        temperature = globalTemperature
        max_tokens = maxTokensGlobal
        stream = false
    }

let sendRequest options payload =
    async {
        use client = new HttpClient()
        client.Timeout <- TimeSpan.FromMinutes(globalTimeoutMinutes)
        let url = $"{options.ServerUrl}/v1/chat/completions"
        
        let jsOptions = JsonSerializerOptions()
        jsOptions.DefaultIgnoreCondition <- JsonIgnoreCondition.WhenWritingNull
        let json = JsonSerializer.Serialize(payload, jsOptions)
        
        let content = new StringContent(json, Encoding.UTF8, "application/json")
        
        match options.ApiKey with
        | Some key -> client.DefaultRequestHeaders.Add("Authorization", $"Bearer {key}")
        | None -> ()
        
        printfn $"Sending request to: {url}"

        let! response = client.PostAsync(url, content) |> Async.AwaitTask
        
        if not response.IsSuccessStatusCode then
            let! errorContent = response.Content.ReadAsStringAsync() |> Async.AwaitTask
            failwith $"Request failed with status {response.StatusCode}: {errorContent}"

        let! responseContent = response.Content.ReadAsStringAsync() |> Async.AwaitTask
        printfn $"Received response: {responseContent}"
        return responseContent
    }
    
let getResponseContent (rawResponse: string) =
    try
        use jsonDoc = JsonDocument.Parse(rawResponse)
        let root = jsonDoc.RootElement
        
        match root.TryGetProperty("choices") with
        | (true, choices) when choices.ValueKind = JsonValueKind.Array && choices.GetArrayLength() > 0 ->
            let firstChoice = choices[0]
            
            match firstChoice.TryGetProperty("message") with
            | (true, message) ->
                match message.TryGetProperty("content") with
                | (true, content) when content.ValueKind = JsonValueKind.String ->
                    Some(content.GetString())
                | _ -> 
                    printfn "Invalid response: missing content"
                    None
            | _ -> 
                printfn "Invalid response: missing message"
                None
        | _ -> 
            printfn "Invalid response: missing choices"
            None
    with ex ->
        printfn $"Error parsing response: {ex.Message}"
        None

let parseListResponse (rawResponse: string) =
    getResponseContent rawResponse
    |> Option.map (fun contentStr ->
        contentStr.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
        |> Array.map (fun s -> s.Trim())
        |> Array.filter (System.String.IsNullOrWhiteSpace >> not)
        |> Array.toList
    )
    
let parseRewriteResponse (rawResponse: string) =
    getResponseContent rawResponse
    |> Option.bind (fun contentStr ->
        let lines = contentStr.Split('\n')
        let mutable currentPath = None
        let mutable currentContent = []
        let mutable files = []
        for line in lines do
            match currentPath with
            | None ->
                let trimmedLine = line.TrimStart()
                if trimmedLine.StartsWith("```") then
                    let pathPart = trimmedLine.[3..].Trim()  // Remove leading ``` and trim
                    currentPath <- Some(pathPart)
                    currentContent <- []
            | Some(path) ->
                let trimmedLine = line.Trim()
                if trimmedLine.StartsWith("```") then
                    let content = currentContent |> List.rev |> String.concat Environment.NewLine
                    files <- { path = path; content = content } :: files
                    currentPath <- None
                    currentContent <- []
                else
                    currentContent <- line :: currentContent
        Some (List.rev files)
    )
        

let writeOutputFile (targetPath: string) (file: CodeFile) =
    let fullPath = Path.Combine(targetPath, file.path)
    let directory = Path.GetDirectoryName(fullPath)
    
    if not (String.IsNullOrEmpty(directory)) then
        Directory.CreateDirectory(directory) |> ignore
    
    File.WriteAllText(fullPath, file.content)
    printfn $"Written: {file.path}"

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
        Principles = None
        ApiKey = None
    }
    
    parseInternal (args |> Array.toList) defaultOptions

let queryRelevantFormats (options: Options) =
    printfn "\n=== Querying relevant file formats ==="
    let payload = createFormatQueryPayload options
    let response = sendRequest options payload |> Async.RunSynchronously
    parseListResponse response

let printBatch list =
    printfn $"=== Batch ==="
    list 
    |> List.iter (fun file -> printfn $"  - {file.path}")

let filterImportantFiles options allFiles =
    printfn "\n=== Filtering important files ==="
    let batchSize = batchSizeGlobal
    let batches = 
        allFiles 
        |> List.chunkBySize batchSize

    let mutable importantFiles = Set.empty
    for batch in batches do
        printfn $"Analyzing batch of {batch.Length} files..."
        printBatch batch
        let payload = createAnalysisPayload options batch
        let response = sendRequest options payload |> Async.RunSynchronously
        
        match parseListResponse response with
        | Some results -> 
            importantFiles <- Set.union importantFiles (results |> Set.ofList)
            printfn $"Found {results.Length} important files in this batch (Total: {importantFiles.Count})"
        | None -> 
            printfn "No important files identified in this batch"
        
    allFiles
        |> List.filter (fun file -> Set.contains file.path importantFiles)

let rewriteFiles (options: Options) (files: CodeFile list)  =
    printfn $"\nProcessing {files.Length} files..."
    try
        let payload = createRewritePayload options files
        let response = sendRequest options payload |> Async.RunSynchronously
        
        match parseRewriteResponse response with
        | Some results ->
            printfn $"Successfully converted {results.Length} files"
            Some results
        | None ->
            printfn "No valid conversion returned for these files"
            None
    with ex ->
        printfn $"Error processing files: {ex.Message}, {ex.StackTrace}"
        None

let runConversion (options: Options) =
    try
        printfn $"Determining relevant file formats for {options.SourceLang} -> {options.TargetLang} conversion"
        let relevantFormatsList = queryRelevantFormats options
        let relevantFormats = match relevantFormatsList with Some f -> String.Join(", ", f) | None -> "All files"
        printfn $"Relevant formats: {relevantFormats}"

        printfn $"Reading source files from %s{options.SourcePath}"
        let allSourceFiles = readSourceFiles options.SourcePath relevantFormatsList
        printfn $"Found {allSourceFiles.Length} relevant files to process"

        // Filter to get only important files
        let importantFiles = filterImportantFiles options allSourceFiles
        printfn $"Identified {importantFiles.Length} important files to convert"

        Directory.CreateDirectory(options.TargetPath) |> ignore
        let mutable successCount = 0

        printfn "\n=== Starting rewrite phase ==="
        let batchSize = batchSizeGlobal
        let batches = 
            importantFiles 
            |> List.chunkBySize batchSize

        for batch in batches do
            printBatch batch
            match rewriteFiles options batch with
            | Some results ->
                results |> List.iter (fun result ->
                    writeOutputFile options.TargetPath result
                    successCount <- successCount + 1
                )
            | None -> ()

        printfn $"\nConversion complete. Successfully converted {successCount} of {importantFiles.Length} important files"
        0
    with
    | :? OperationCanceledException as ex ->
        Console.ForegroundColor <- ConsoleColor.Red
        printfn $"\nTask cancelled: {ex.Message}"
        Console.ResetColor()
        1
    | ex ->
        printfn $"\nError: %s{ex.Message}"
        1

[<EntryPoint>]
let main argv =
    try
        if argv.Length = 0 then
            printfn "Usage: dotnet run [options]"
            printfn "Options:"
            printfn "  --source <path>        Source file or directory"
            printfn "  --source-lang <lang>   Source language"
            printfn "  --target <path>        Target directory"
            printfn "  --target-lang <lang>   Target language"
            printfn "  --server <url>         API server URL"
            printfn "  --model <model-name>   LLM Model (like deepseek-chat)"
            printfn "  --principles <text>    (Optional) Conversion principles"
            printfn "  --api-key <key>        (Optional) API key for authentication"
            1
        else
            let options = parseCommandLine argv
            
            if String.IsNullOrEmpty(options.SourcePath) then failwith "Source path is required"
            if String.IsNullOrEmpty(options.SourceLang) then failwith "Source language is required"
            if String.IsNullOrEmpty(options.TargetPath) then failwith "Target path is required"
            if String.IsNullOrEmpty(options.TargetLang) then failwith "Target language is required"
            if String.IsNullOrEmpty(options.ServerUrl) then failwith "Server URL is required"
            if String.IsNullOrEmpty(options.Model) then failwith "LLM Model is required"

            runConversion options
    with
    | ex ->
        printfn $"Error: {ex.Message}"
        2